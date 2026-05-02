"""``MsSwiftRolloutActor`` -- run ``swift rollout`` inside a Monarch actor.

**Conforms to:** :class:`forge.core.protocols.InferenceServerProtocol`
(``host_info`` / ``start`` / ``wait_ready`` / ``teardown``).

One Monarch actor per rollout host.  The actor wraps ms-swift's
``SwiftRolloutDeploy`` (vLLM TP=N FastAPI server) and exposes the
:class:`InferenceServerProtocol` endpoints so the driver can drive
lifecycle without ``ssh ... uvicorn ...``.

(``teardown`` and not ``stop`` because ``ActorMesh.stop`` is reserved
by Monarch's proc-mesh wrapper and would collide at spawn time.)

Why one actor per host (not per NPU)
------------------------------------
``SwiftRolloutDeploy`` internally spawns N child processes (one per
data-parallel vLLM rank) via ``multiprocessing.Pipe``.  Wrapping it in
a per-NPU SPMDActor would mean N outer processes each spawning their
own N children = N^2 processes fighting for the same N NPUs -- a
non-starter (same reasoning as
:class:`forge.actors.vllm_inference.VLLMInferenceActor`).

Why plain ``Actor`` (not ``SPMDActor``)
---------------------------------------
ms-swift owns its own multiprocessing topology (``Pipe`` + ``Process``)
and exposes its own HTTP master_port via ``get_open_port``.  Setting
``RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT`` on the outer process would
just confuse the spawned vLLM children.  Plain ``Actor`` keeps the
contract minimal.

Lifecycle::

    rollout_actor = rollout_mesh.spawn("rollout", MsSwiftRolloutActor)
    info = await rollout_actor.start.call_one(swift_args=..., port=8000)
    # info = {"host": "192.168.0.26", "port": 8000, "world_size": 4, ...}
    await rollout_actor.wait_ready.call_one(timeout_s=300.0)
    # ... trainer drives weight sync via http://info["host"]:info["port"] ...
    await rollout_actor.teardown.call_one()

The vLLM process count exposed to ``WeightSyncClient.init_communicator``
(``vllm_world_size``) is just ``vllm_data_parallel_size *
vllm_tensor_parallel_size`` from the swift_args -- the trainer fetches
this via ``GET /get_world_size/`` so the driver doesn't need to know it.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from monarch.actor import Actor, endpoint

from forge.utils.process_tree import kill_descendants

logger = logging.getLogger(__name__)


def _swift_args_to_argv(swift_args: dict[str, Any]) -> list[str]:
    """Same translation as the trainer actor; duplicated to keep modules independent."""
    argv: list[str] = []
    for key, value in swift_args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            argv.extend([flag, str(value).lower()])
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        else:
            argv.extend([flag, str(value)])
    return argv


def _primary_ip() -> str:
    """Best-effort primary-NIC IP for the actor's host.

    Used to tell the driver which address the trainer should use when
    targeting this host's rollout server.  Falls back to ``127.0.0.1``
    if hostname resolution loops back -- the driver should also accept
    a caller-supplied override (it knows the SSHJob hostname).
    """
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return "127.0.0.1"


class MsSwiftRolloutActor(Actor):
    """One ms-swift rollout server per actor instance."""

    def __init__(self) -> None:
        super().__init__()
        self._deploy = None  # SwiftRolloutDeploy instance after start()
        self._server = None  # uvicorn.Server
        self._thread: threading.Thread | None = None
        self._host: str = ""
        self._port: int = 0
        self._world_size: int = 0

    @endpoint
    def host_info(self) -> dict[str, Any]:
        """Return primary-NIC IP and hostname.

        Driver calls this BEFORE ``start`` so it knows which URL to give
        to the trainer's ``vllm_server_base_url``.  Cheap (no model load
        side-effects) so safe to invoke speculatively.
        """
        return {
            "ip": _primary_ip(),
            "hostname": socket.gethostname(),
        }

    @endpoint
    def start(
        self,
        *,
        swift_args: dict[str, Any],
        host: str = "0.0.0.0",
        port: int = 8000,
        use_modelscope: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Apply the AReaL glue, build SwiftRolloutDeploy, start uvicorn in a daemon thread.

        Idempotency: calling ``start`` twice on the same actor raises
        because we don't try to tear down + rebuild the deploy.  Use
        ``stop`` then re-spawn the actor for that.

        Args:
            swift_args: Dict of ms-swift CLI flags for ``swift rollout``.
                E.g. ``{"model": "/path", "vllm_tensor_parallel_size": 4,
                "vllm_data_parallel_size": 1, ...}``.  ``host`` / ``port``
                are appended automatically below.
            host: NIC the FastAPI server should bind to.  ``0.0.0.0`` lets
                cross-host trainers reach us; if you only need same-host
                access, ``127.0.0.1`` is fine.
            port: Port to bind.
            use_modelscope: Set ``USE_MODELSCOPE_HUB=1`` before importing
                swift -- required because SSHJob workers don't inherit the
                driver's env.
            extra_env: Same escape hatch as the trainer actor.

        Returns:
            ``{"host": str, "port": int, "world_size": int, "pid": int}``.
            ``world_size`` = ``data_parallel_size * tensor_parallel_size``
            and is what the trainer's ``WeightSyncClient`` will see via
            ``GET /get_world_size/``.

        Notes:
            * Heavy imports (``swift``, ``vllm``, ``torch_npu``) happen
              inside ``start()`` so this module stays importable on a
              CPU-only orchestration host.
            * ``install_server_patches`` MUST run before
              ``SwiftRolloutDeploy.__init__`` because the latter
              immediately calls ``_register_rl_rollout_app`` and
              ``_start_data_parallel_workers``, both of which we patch.
            * ``uvicorn.run`` blocks; we run it in a daemon thread so
              the actor's ``start`` endpoint can return after the
              child workers report ``ready`` via the lifespan handshake.
        """
        if self._deploy is not None:
            raise RuntimeError(
                "MsSwiftRolloutActor.start called twice on the same actor "
                f"(host={self._host} port={self._port}).  "
                "Spawn a fresh actor for a fresh server."
            )

        if use_modelscope:
            os.environ.setdefault("USE_MODELSCOPE_HUB", "1")
            os.environ.setdefault("MODELSCOPE_CACHE", "/root/.cache/modelscope/hub")

        os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
        os.environ.setdefault("VLLM_USE_V1", "1")
        os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "600")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if extra_env:
            os.environ.update({str(k): str(v) for k, v in extra_env.items()})

        # Apply glue BEFORE building SwiftRolloutDeploy -- the deploy ctor
        # immediately wires the FastAPI app + spawns workers, and our
        # patches need to be in place before either step.
        from forge.engines.msswift.glue import install_server_patches

        install_server_patches()

        # Compose the rollout argv: caller args + driver-supplied host/port.
        merged: dict[str, Any] = dict(swift_args)
        merged["host"] = host
        merged["port"] = int(port)
        argv = _swift_args_to_argv(merged)
        logger.info(
            "MsSwiftRolloutActor host=%s pid=%d argv=%s",
            socket.gethostname(),
            os.getpid(),
            " ".join(argv),
        )

        from swift.pipelines.infer.rollout import SwiftRolloutDeploy

        # Building SwiftRolloutDeploy:
        # 1. parses argv -> RolloutArguments
        # 2. wires FastAPI app (our patched _register_rl_rollout_app
        #    registers /areal_* routes here)
        # 3. spawns N data-parallel vLLM workers via our patched
        #    _start_data_parallel_workers (which routes through
        #    _spawn_wrapped_llm_worker so each child re-applies
        #    install_worker_patches)
        # 4. children load the model and signal "ready" via Pipe
        self._deploy = SwiftRolloutDeploy(argv)

        # ``num_connections`` = data_parallel_size when sync, 1 when async.
        # ``vllm_tensor_parallel_size`` from the args is the per-DP TP.
        # Trainer-side WeightSyncClient broadcasts to all DP*TP workers.
        args = self._deploy.args
        dp = max(1, getattr(args, "vllm_data_parallel_size", 1))
        tp = max(1, getattr(args, "vllm_tensor_parallel_size", 1))
        self._world_size = dp * tp
        self._host = host
        self._port = int(port)

        import uvicorn

        config = uvicorn.Config(
            self._deploy.app,
            host=host,
            port=int(port),
            log_level=getattr(args, "log_level", "info"),
            access_log=False,  # quieter; areal_* routes log themselves
        )
        self._server = uvicorn.Server(config)

        # uvicorn.Server.run is blocking, but it triggers the FastAPI
        # lifespan which is what waits for child workers to report ready.
        # Running it in a daemon thread lets ``start`` return promptly
        # while leaving the lifespan handshake to ``wait_ready``.
        self._thread = threading.Thread(
            target=self._server.run,
            name="msswift-rollout-uvicorn",
            daemon=True,
        )
        self._thread.start()

        return {
            "host": host,
            "port": int(port),
            "world_size": self._world_size,
            "pid": os.getpid(),
        }

    @endpoint
    def wait_ready(self, *, timeout_s: float = 300.0, interval_s: float = 2.0) -> dict:
        """Poll ``GET /health/`` until 200 OK or timeout.

        Returns ``{"ready": True, "elapsed_s": float}`` on success; raises
        ``TimeoutError`` otherwise.  Driver should call this between
        ``start`` and the trainer launch to avoid the trainer hitting a
        not-yet-listening server.
        """
        if self._port == 0:
            raise RuntimeError("MsSwiftRolloutActor.wait_ready called before start")
        url = f"http://127.0.0.1:{self._port}/health/"
        t0 = time.time()
        last_err: Exception | None = None
        while time.time() - t0 < timeout_s:
            # Catch both ``URLError`` (network) and ``HTTPError`` (4xx/5xx);
            # ms-swift returns 200 once every DP worker reports ready via
            # the FastAPI lifespan handshake, so a transient connection
            # refusal during startup is normal.
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - t0
                        logger.info(
                            "MsSwiftRolloutActor ready at %s after %.1fs",
                            url,
                            elapsed,
                        )
                        return {"ready": True, "elapsed_s": elapsed}
            except (urllib.error.URLError, OSError) as e:
                last_err = e
            # Surface any uvicorn-thread crash so we don't poll forever.
            if self._thread is not None and not self._thread.is_alive():
                raise RuntimeError(
                    "MsSwiftRolloutActor uvicorn thread died during startup; "
                    f"last health-check error: {last_err!r}"
                )
            time.sleep(interval_s)
        raise TimeoutError(
            f"MsSwiftRolloutActor not ready after {timeout_s:.0f}s; "
            f"last health-check error: {last_err!r}"
        )

    @endpoint
    def teardown(
        self,
        *,
        uvicorn_timeout_s: float = 10.0,
        term_grace_s: float = 8.0,
        kill_grace_s: float = 5.0,
    ) -> dict[str, Any]:
        """Best-effort uvicorn + recursive child-tree teardown.  Always returns.

        Named ``teardown`` (not ``stop``) because ``ActorMesh.stop`` is a
        reserved Monarch method on the proc-mesh wrapper and would
        collide at spawn time.

        Cleanup order (each step is best-effort, never raises):

        1. Flip ``uvicorn.Server.should_exit`` so the FastAPI thread
           leaves the accept loop.  Join with ``uvicorn_timeout_s``.
        2. Snapshot **every descendant** of this actor's PID via
           ``psutil`` (immediate ``llm_worker`` children + their
           ``EngineCore`` + ``Worker_TP*`` grandchildren -- the latter
           are the ones holding HCCL ports, so just terminating
           ``self._deploy.processes`` leaks ports).
        3. SIGTERM the whole tree, wait ``term_grace_s`` for graceful
           exit (vLLM has its own atexit hooks).
        4. SIGKILL survivors, wait ``kill_grace_s`` for the kernel to
           reap them.
        5. Best-effort ``proc.join(0)`` on the ``mp.Process`` handles so
           ``SwiftRolloutDeploy`` doesn't think they're still alive.

        Returns a dict with counts so the driver can log/assert no
        zombies were left behind.
        """
        # --- 1. uvicorn shutdown ---
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=uvicorn_timeout_s)
            uvicorn_joined = not self._thread.is_alive()
        else:
            uvicorn_joined = True

        # --- 2-4. recursive descendant kill ---
        terminated, killed, survivors = kill_descendants(
            os.getpid(),
            term_grace_s=term_grace_s,
            kill_grace_s=kill_grace_s,
        )

        # --- 5. join immediate mp.Process handles so SwiftRolloutDeploy's
        # lifespan doesn't try to wait on them again on a future call.
        joined_workers = 0
        if self._deploy is not None:
            for proc in getattr(self._deploy, "processes", []):
                try:
                    if proc.is_alive():
                        proc.join(timeout=0)
                    if not proc.is_alive():
                        joined_workers += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning("rollout worker join raised: %s", e)

        logger.info(
            "MsSwiftRolloutActor teardown: uvicorn_joined=%s host=%s port=%d "
            "terminated=%d killed=%d survivors=%d joined_workers=%d",
            uvicorn_joined,
            self._host,
            self._port,
            terminated,
            killed,
            survivors,
            joined_workers,
        )
        return {
            "uvicorn_joined": uvicorn_joined,
            "host": self._host,
            "port": self._port,
            "terminated": terminated,
            "killed": killed,
            "survivors": survivors,
            "joined_workers": joined_workers,
        }
