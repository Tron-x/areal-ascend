"""``MsSwiftTrainerActor`` -- run ms-swift's ``rlhf_main`` inside a Monarch actor.

**Conforms to:** :class:`forge.core.protocols.SPMDTrainerProtocol`
(Adapter Path -- ``setup_env`` + ``run`` + ``teardown``).

Phase A of the framework first principles (``.cursor/rules/framework-first-principles.mdc``):
make ms-swift swappable with TorchTitan / LlamaFactory / FSDP from the
orchestration layer's point of view.  Same SPMDActor lifecycle, same
``setup_env`` / ``run`` shape as :class:`forge.actors.titan_trainer.TitanTrainerActor`
and :class:`forge.actors.llamafactory_trainer.LlamaFactoryTrainerActor`,
just a different framework-specific body.

Translating ``torchrun --nproc_per_node=4 ... swift rlhf ...``
--------------------------------------------------------------
Yesterday's PoC ran ms-swift via::

    torchrun --nproc_per_node=4 --master_port=29501 \\
        run_swift_rlhf.py --rlhf_type grpo --model ... --use_vllm true \\
        --vllm_mode server --vllm_server_base_url http://192.168.0.26:8000 ...

torchrun does two things before invoking the script:

1. Sets ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` /
   ``MASTER_PORT`` from its agent state.
2. Sets ``TORCHELASTIC_USE_AGENT_STORE=True`` so subsequent
   ``init_process_group`` calls reuse the agent's TCPStore.

We replicate (1) via the inherited :meth:`SPMDActor._setup_env`.  We do
NOT replicate (2) because we do not run a torchelastic agent inside the
actor -- the inherited env wiring is enough for ms-swift's own DDP init,
and the cross-mesh weight-sync group built by
:func:`forge.engines.msswift.glue.install_client_patches` doesn't need
the agent store.  This sidesteps the whole
``_suspend_torchelastic_agent_store`` workaround that the bash PoC needed.

The vLLM server URL (where the trainer pushes weights to) is shipped
from the driver via ``run(swift_args=...)`` -- the driver knows where it
spawned :class:`forge.actors.msswift_rollout.MsSwiftRolloutActor` and
passes ``http://<rollout_host>:8000`` in the ``vllm_server_base_url``
key.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time
from typing import Any

from monarch._src.spmd.actor import SPMDActor
from monarch.actor import endpoint

from forge.utils.process_tree import kill_descendants

logger = logging.getLogger(__name__)


def _swift_args_to_argv(swift_args: dict[str, Any]) -> list[str]:
    """Turn a dict of swift CLI flags into a ``sys.argv``-style list.

    ms-swift's ``rlhf_main`` accepts either a parsed ``RLHFArguments``
    dataclass or a ``List[str]`` argv (which it then runs through its
    own argparse).  We pick argv because (a) the algo-author YAMLs we
    ship are designed to mirror the ``swift rlhf ...`` CLI 1:1, so
    dict -> argv is a trivial translation, and (b) dataclass
    instantiation pulls in heavy ms-swift imports at the driver, which
    we want to defer to the actor.

    List-valued entries become repeated CLI tokens, mirroring how
    bash splits ``--reward_funcs accuracy format``.  Booleans use
    swift's ``--flag true|false`` convention rather than ``--flag``
    presence semantics, again to match the YAML/CLI symmetry.
    """
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


class MsSwiftTrainerActor(SPMDActor):
    """Run ``swift.pipelines.train.rlhf.rlhf_main`` once per actor instance.

    Subclasses :class:`monarch._src.spmd.actor.SPMDActor` to inherit
    rank / device env wiring + the ``setup_env(master_addr, master_port)``
    endpoint.  Adds a single ``run()`` endpoint that does the
    ms-swift-specific work.

    Endpoint contract intentionally mirrors
    :class:`forge.actors.llamafactory_trainer.LlamaFactoryTrainerActor.run`
    so a future ``forge launch --mode grpo-msswift`` driver can reuse the
    LF / titan-pretrain wiring with no shape divergence at the
    orchestrator level.
    """

    @endpoint
    def run(
        self,
        swift_args: dict[str, Any],
        *,
        vllm_server_base_url: str,
        cwd: str = "/root/ms-swift",
        master_addr: str | None = None,
        master_port: int | None = None,
        ws_master_addr: str | None = None,
        ws_master_port: int | None = None,
        use_modelscope: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Drive one full ms-swift RLHF run on this rank.

        Args:
            swift_args: Dict of ms-swift CLI flags.  E.g.
                ``{"rlhf_type": "grpo", "model": "/path", "use_vllm": True,
                "vllm_mode": "server", "max_steps": 1, ...}``.  Translated
                to ``argv`` and handed to ``rlhf_main``.  ``vllm_server_base_url``
                is appended automatically (see below) -- do NOT include it
                here, the driver knows the rollout actor's URL.
            vllm_server_base_url: HTTP URL of the rollout actor's vLLM
                server (e.g. ``http://192.168.0.26:8000``).  Appended into
                ``swift_args["vllm_server_base_url"]`` here so the driver
                doesn't have to know what host the rollout landed on at
                YAML-write time.
            cwd: Working directory for ms-swift.  Defaults to the in-tree
                ms-swift repo root.
            master_addr / master_port: If provided (driver-supplied), set
                the trainer's own DDP rendezvous via ``self._setup_env``.
                When ``None`` the caller is expected to have invoked
                ``setup_env`` beforehand (the standard SPMDActor flow).
            ws_master_addr / ws_master_port: Cross-mesh weight-sync TCPStore
                endpoint.  Trainer rank 0 binds here; rollout vLLM workers
                connect in.  Exported as ``AREAL_WS_MASTER_ADDR`` /
                ``AREAL_WS_MASTER_PORT`` so
                :func:`forge.engines.msswift.glue._resolve_master_addr`
                picks them up.  When ``ws_master_addr`` is ``None`` the
                glue falls back to this host's primary NIC IP.
            use_modelscope: When ``True``, set ``USE_MODELSCOPE_HUB=1`` in
                this proc *before* the ms-swift import.  Required when
                HuggingFace is unreachable.  SSHJob workers don't inherit
                the driver's env so the actor must export it itself.
            extra_env: Arbitrary env-var overrides applied after the FSDP
                env, before the ms-swift import.  Escape hatch for
                MODELSCOPE_CACHE / TASK_QUEUE_ENABLE / etc. without
                churning this signature.

        Returns:
            ``{"rank": int, "host": str, "elapsed_s": float, "ok": bool}``.

        Notes:
            * Heavy imports (``torch``, ``torch_npu``, ``swift``,
              ``transformers``) happen INSIDE ``run()`` so this module
              stays importable on CPU-only orchestration hosts.
            * ``install_client_patches`` MUST run before
              ``rlhf_main`` because it monkey-patches
              ``swift.rlhf_trainers.vllm_client.VLLMClient`` which
              ms-swift's ``RLHFArguments.__post_init__`` instantiates
              when ``use_vllm=True`` and ``vllm_mode='server'``.
        """

        if master_addr is not None and master_port is not None:
            self._setup_env(master_addr, master_port)

        os.chdir(cwd)

        # ms-swift's RLHFArguments.__post_init__ guards against model-parallel
        # under GRPO+vLLM with ``is_mp()``: it raises if
        # ``visible_npus // local_world_size >= 2``.  When the host exposes
        # 8 NPUs but our trainer pool only uses 4 procs (the other 4 are
        # for the rollout pool on a different host or idle on a colocate),
        # is_mp() trips.  Mirror yesterday's bash PoC: pin
        # ``ASCEND_RT_VISIBLE_DEVICES`` to exactly ``LOCAL_WORLD_SIZE`` NPUs
        # so every rank sees only the slice it owns and ``is_mp()`` returns
        # False.  Honor a caller-supplied value (extra_env / env) if present.
        local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0)
        if local_ws > 0 and not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(local_ws)
            )
            os.environ.setdefault(
                "CUDA_VISIBLE_DEVICES", os.environ["ASCEND_RT_VISIBLE_DEVICES"]
            )

        if use_modelscope:
            os.environ.setdefault("USE_MODELSCOPE_HUB", "1")
            os.environ.setdefault("MODELSCOPE_CACHE", "/root/.cache/modelscope/hub")

        # NPU graph capture is incompatible with TASK_QUEUE_ENABLE=2 in
        # vllm-ascend (we hit this in the AReaL self-validation smoke).
        # Force =1 on NPU hosts unless the caller explicitly overrides.
        os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
        os.environ.setdefault("VLLM_USE_V1", "1")
        os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "600")

        # HuggingFace / datasets cache resolution: prefer offline so we
        # never try to reach huggingface.co from a closed cluster.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if ws_master_addr:
            os.environ["AREAL_WS_MASTER_ADDR"] = str(ws_master_addr)
        if ws_master_port:
            os.environ["AREAL_WS_MASTER_PORT"] = str(int(ws_master_port))

        if extra_env:
            os.environ.update({str(k): str(v) for k, v in extra_env.items()})

        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        world_size = int(os.environ.get("WORLD_SIZE", "-1"))
        host = socket.gethostname()

        # Force NPU device pinning per LOCAL_RANK before any torch / swift
        # import touches a device.  Same dance as the LF actor.
        import torch

        try:
            import torch_npu  # noqa: F401  -- registers the npu backend

            if local_rank >= 0 and torch.npu.is_available():
                torch.npu.set_device(local_rank)
        except ImportError:
            pass

        # Apply the ms-swift -> areal.weight_sync glue BEFORE rlhf_main
        # imports trl.VLLMClient.  The glue is idempotent so re-applying
        # in a repeated run() call is safe.
        from forge.engines.msswift.glue import install_client_patches

        install_client_patches()

        # math_verify uses ``signal.alarm`` for parse() timeouts which only
        # works in the main thread.  Monarch SPMDActor endpoints execute on
        # a worker thread, so any reward call into ``MathAccuracy`` raises
        # before computing.  Force parsing_timeout=None (= no timeout) for
        # all parse() invocations so reward funcs work inside actors.  The
        # algo author can still timebox at the workflow level if desired.
        try:
            import math_verify as _mv

            if not getattr(_mv.parse, "_areal_thread_safe", False):
                _orig_parse = _mv.parse

                def _parse_thread_safe(*args, **kwargs):
                    kwargs.setdefault("parsing_timeout", None)
                    return _orig_parse(*args, **kwargs)

                _parse_thread_safe._areal_thread_safe = True
                _mv.parse = _parse_thread_safe
                # ms-swift imports parse via ``from math_verify import parse``
                # which captures the symbol at import time.  Patch the
                # already-imported reference too if the reward module is
                # already loaded (no-op otherwise).
                try:
                    from swift.rewards import orm as _swift_orm

                    if hasattr(_swift_orm, "parse"):
                        _swift_orm.parse = _parse_thread_safe
                except ImportError:
                    pass
                logger.info(
                    "math_verify.parse patched to parsing_timeout=None "
                    "(signal.alarm needs main thread, actor runs on worker)"
                )
        except ImportError:
            pass

        # Compose the final argv: caller-supplied dict + the rollout URL
        # the driver discovered for us.  Caller args win for everything
        # except vllm_server_base_url (driver owns network topology).
        merged: dict[str, Any] = dict(swift_args)
        merged["vllm_server_base_url"] = vllm_server_base_url
        merged.setdefault("use_vllm", True)
        merged.setdefault("vllm_mode", "server")
        merged.setdefault("enable_flattened_weight_sync", False)
        argv = _swift_args_to_argv(merged)

        if rank == 0:
            logger.info(
                "MsSwiftTrainerActor rank=0 host=%s argv=%s",
                host,
                " ".join(argv),
            )
        logger.info(
            "MsSwiftTrainerActor rank=%d local_rank=%d world_size=%d host=%s "
            "ws_master=%s:%s vllm_server=%s",
            rank,
            local_rank,
            world_size,
            host,
            os.environ.get("AREAL_WS_MASTER_ADDR", "<auto>"),
            os.environ.get("AREAL_WS_MASTER_PORT", "<auto>"),
            vllm_server_base_url,
        )

        from swift.pipelines.train.rlhf import rlhf_main

        t0 = time.time()
        err: BaseException | None = None
        try:
            rlhf_main(argv)
        except BaseException as e:  # noqa: BLE001
            err = e
            logger.exception("MsSwiftTrainerActor rank=%d crashed", rank)
            raise
        finally:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning(
                    "MsSwiftTrainerActor rank=%d destroy_process_group raised %r",
                    rank,
                    cleanup_exc,
                )

        elapsed = time.time() - t0
        try:
            torch.npu.synchronize()
        except Exception:  # noqa: BLE001
            pass

        sys.stdout.flush()
        sys.stderr.flush()

        return {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "host": host,
            "elapsed_s": elapsed,
            "ok": err is None,
        }

    @endpoint
    def cross_mesh_group_status(self) -> dict[str, Any]:
        """Per-rank snapshot of active cross-mesh weight-sync groups.

        Surfaces the ``WeightSyncClient`` registry maintained by
        :mod:`forge.engines.msswift.glue` so Monarch (or a human at the
        driver console) can ask "is the cross-mesh group up?" without
        having to ssh into the trainer host and grep ``ps``.

        Per-rank because this is an SPMD endpoint -- typically only
        rank 0 has populated entries (TRL ``GRPOTrainer`` instantiates
        ``VLLMClient`` only on rank 0), other ranks return an empty
        ``groups`` list, which itself is informative.

        Per ``framework-first-principles.mdc`` 准则 3 ("any adapter
        must plug into Monarch"): the cross-mesh PG cannot move out of
        the trainer process (``torch.distributed`` PGs are
        process-local), so we expose it through this endpoint instead
        of pretending to spawn a separate ``CrossMeshGroupActor``.
        """
        rank = int(os.environ.get("RANK", "-1"))
        host = socket.gethostname()
        try:
            from forge.engines.msswift.glue import get_cross_mesh_group_status

            groups = get_cross_mesh_group_status()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "MsSwiftTrainerActor rank=%d cross_mesh_group_status raised: %s",
                rank,
                e,
            )
            groups = []
        return {
            "rank": rank,
            "host": host,
            "num_groups": len(groups),
            "groups": groups,
        }

    @endpoint
    def cross_mesh_group_teardown(self) -> dict[str, Any]:
        """Force-close every cross-mesh weight-sync group on this rank.

        Recovery hook: when the driver detects a stuck weight sync
        (e.g. push timing out across N retries), it can call this to
        release the trainer-side ``ProcessGroup`` + HCCL/XCCL handles
        without killing the trainer process.  After this returns, a
        subsequent TRL step that rebuilds ``VLLMClient.init_communicator``
        will rebuild the group from scratch.

        Idempotent: returns ``{"closed": 0}`` when no groups are
        registered (e.g. on non-rank-0 ranks, or before any sync has
        run).  Errors during ``WeightSyncClient.close`` do NOT raise --
        they're surfaced in ``details`` so the driver can decide
        whether to escalate to a process kill.
        """
        rank = int(os.environ.get("RANK", "-1"))
        host = socket.gethostname()
        try:
            from forge.engines.msswift.glue import teardown_cross_mesh_groups

            result = teardown_cross_mesh_groups()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "MsSwiftTrainerActor rank=%d cross_mesh_group_teardown raised: %s",
                rank,
                e,
            )
            result = {"closed": 0, "errors": 1, "details": [{"error": str(e)}]}
        return {"rank": rank, "host": host, **result}

    @endpoint
    def teardown(
        self,
        *,
        term_grace_s: float = 5.0,
        kill_grace_s: float = 3.0,
    ) -> dict[str, Any]:
        """Best-effort cleanup symmetric to :class:`MsSwiftRolloutActor`.

        Trainer doesn't normally own a multiprocessing tree (DDP runs
        in-proc), but ms-swift's ``DataLoader`` may have ``num_workers
        > 0``, and a crashed ``run()`` can leave those (or any tool
        ms-swift forked off) lingering.  Symmetric ``teardown`` lets
        the driver reuse the same finally-block shape on both sides.

        Steps:

        1. Destroy the torch.distributed process group if ``run()``
           bailed before reaching its own ``finally`` (e.g. SIGTERM
           from the driver).
        2. Recursively SIGTERM/SIGKILL every descendant of this actor
           via :func:`forge.utils.process_tree.kill_descendants`.

        Returns ``{"rank", "host", "terminated", "killed", "survivors",
        "dist_destroyed"}`` so the driver can assert no zombies.
        """
        rank = int(os.environ.get("RANK", "-1"))
        host = socket.gethostname()

        # Tear down cross-mesh weight-sync PGs *before* the main DDP
        # group to avoid leaking HCCL/XCCL handles (each cross-mesh PG
        # holds its own bcast comm).  Best-effort -- errors are
        # logged but do not block the rest of teardown.
        try:
            from forge.engines.msswift.glue import teardown_cross_mesh_groups

            cm_result = teardown_cross_mesh_groups()
            if cm_result["closed"] > 0 or cm_result["errors"] > 0:
                logger.info(
                    "MsSwiftTrainerActor teardown rank=%d cross-mesh: closed=%d errors=%d",
                    rank,
                    cm_result["closed"],
                    cm_result["errors"],
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "MsSwiftTrainerActor rank=%d teardown_cross_mesh_groups raised: %s",
                rank,
                e,
            )

        dist_destroyed = False
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
                dist_destroyed = True
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "MsSwiftTrainerActor rank=%d destroy_process_group raised: %s",
                rank,
                e,
            )

        terminated, killed, survivors = kill_descendants(
            os.getpid(),
            term_grace_s=term_grace_s,
            kill_grace_s=kill_grace_s,
        )

        logger.info(
            "MsSwiftTrainerActor teardown rank=%d host=%s "
            "dist_destroyed=%s terminated=%d killed=%d survivors=%d",
            rank,
            host,
            dist_destroyed,
            terminated,
            killed,
            survivors,
        )
        return {
            "rank": rank,
            "host": host,
            "dist_destroyed": dist_destroyed,
            "terminated": terminated,
            "killed": killed,
            "survivors": survivors,
        }
