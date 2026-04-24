"""``forge launch`` subcommand — one-command multi-node GRPO launch.

Replaces the flow::

    $ bash forge/scripts/worker_manager.sh start --hostfile ...
    $ FORGE_X=... FORGE_Y=... bash forge/scripts/run_multinode.sh \\
          --hostfile ... --model ...
    $ # on ctrl-c, remember to run worker_manager.sh stop manually

with::

    $ python -m forge launch config.yaml --steps 3 --model /path/to/model

The subcommand is deliberately thin: worker spawn / stop still go through
``worker_manager.sh`` (which already handles SSH correctly), and the
actual training runs under ``python -m forge.apps.grpo`` on the driver
host.  The Python wrapper adds:

1. **Pre-flight checks** — verify hostfile exists, workers are SSH-
   reachable, YAML is parseable, required tooling (``ssh``) is present.
   Fails fast with a clear message instead of half-starting and
   hanging inside Monarch.
2. **Cleanup-on-exit guarantee** — a signal handler + ``finally``
   clause invokes ``worker_manager.sh stop`` even if the user hits
   Ctrl+C or the training crashes, preventing orphan Monarch workers
   that block the next run.
3. **Friendly error translation** — known HCCL / HiXL error codes
   (``103901``, ``EI0015``, ``503900``, ``0x0000000005000007``) are
   post-processed into actionable hints pointing at the relevant
   section of ``weight_sync.md``.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# --- constants ---------------------------------------------------------

FORGE_ROOT = Path(__file__).resolve().parents[2]
"""Repository root (``/root/AReaL`` in dev)."""

WORKER_MGR = FORGE_ROOT / "forge" / "scripts" / "worker_manager.sh"
DEFAULT_ALGO_YAML = FORGE_ROOT / "examples" / "math" / "gsm8k_grpo_npu.yaml"
DEFAULT_TRAIN_SCRIPT = FORGE_ROOT / "examples" / "math" / "gsm8k_rl.py"
DEFAULT_CANN_HOME = "/usr/local/Ascend/cann-9.0.0-beta.1"


# --- parsing -----------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge launch",
        description="Single-command multi-node GRPO launch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Extra args after `--` are forwarded verbatim to "
            "`python -m forge.apps.grpo` (OmegaConf overrides, etc.).\n\n"
            "Example:\n"
            "  python -m forge launch \\\n"
            "      forge/configs/launcher_bare_metal_2node.yaml \\\n"
            "      --steps 3 --model /path/to/model -- actor.path=/override\n"
        ),
    )
    p.add_argument(
        "config",
        help=(
            "Launcher YAML (with optional top-level `allocation_mode` and "
            "`launcher:` block). See forge/configs/launcher_*.yaml."
        ),
    )
    p.add_argument(
        "--algo-config",
        default=str(DEFAULT_ALGO_YAML),
        help=(
            "Algorithm-side YAML (gsm8k_grpo_npu.yaml et al).  Defaults to "
            "examples/math/gsm8k_grpo_npu.yaml.  This is the file that holds "
            "`allocation_mode`, rollout / trainer args, etc.  Will be merged "
            "with --config by grpo.py's `--forge-config` glue until the two "
            "YAMLs are unified."
        ),
    )
    p.add_argument(
        "--train-script",
        default=str(DEFAULT_TRAIN_SCRIPT),
        help="Training entry module path (first arg to `forge.apps.grpo`).",
    )
    p.add_argument(
        "--hostfile",
        default=None,
        help=(
            "Path to hostfile.txt (one `IP slots=N` per line).  Defaults to "
            "forge/configs/hostfile.txt next to the launcher YAML if omitted."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override `+total_train_steps` on the training CLI.",
    )
    p.add_argument(
        "--model",
        default=None,
        help=("Override `actor.path` (Huggingface model path or local snapshot)."),
    )
    p.add_argument(
        "--backend",
        default=None,
        help="Training backend (titan / areal / fsdp).",
    )
    p.add_argument(
        "--model-name",
        default=None,
        help="Backend-specific model name (titan trainspec key).",
    )
    p.add_argument(
        "--model-flavor",
        default=None,
        help="Backend-specific model flavor (titan trainspec variant).",
    )
    p.add_argument(
        "--ssh-port",
        type=int,
        default=36000,
        help="SSH port for worker_manager / driver invocation (default 36000).",
    )
    p.add_argument(
        "--worker-port",
        type=int,
        default=22222,
        help="Monarch worker bootstrap port (default 22222).",
    )
    p.add_argument(
        "--cann",
        default=os.environ.get("CANN_HOME", DEFAULT_CANN_HOME),
        help="CANN install root (for `source set_env.sh`).",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight connectivity / YAML-parse checks.",
    )
    p.add_argument(
        "--launcher-impl",
        default=None,
        choices=("bash", "ssh_job"),
        help=(
            "Which worker lifecycle manager to use.  'bash' = legacy "
            "worker_manager.sh path (default).  'ssh_job' = Monarch-"
            "native ForgeSSHJob (forward-compat with SlurmJob / K8sJob).  "
            "If omitted, read from YAML ``launcher.launcher_impl`` "
            "(itself defaulting to 'bash')."
        ),
    )
    p.add_argument(
        "--keep-workers",
        action="store_true",
        help=(
            "Don't stop workers on exit.  Useful when iterating locally and "
            "you want to re-run launch without paying the worker-start cost."
        ),
    )
    return p


# --- pre-flight --------------------------------------------------------


def _preflight_checks(
    *,
    hostfile: Path,
    config: Path,
    algo_config: Path,
    ssh_port: int,
    worker_port: int,
) -> list[str]:
    """Return a list of error strings; empty list means all green.

    Each check is independent so the user sees every broken thing at
    once rather than one-at-a-time.
    """
    errs: list[str] = []

    # Hostfile exists & has at least one host.
    if not hostfile.is_file():
        errs.append(f"hostfile not found: {hostfile}")
    else:
        hosts = _read_hosts(hostfile)
        if not hosts:
            errs.append(f"hostfile has no non-comment hosts: {hostfile}")
        else:
            # SSH reachability probe on each host.  ~1s budget per host;
            # bail out of the loop if we accumulate too many errs.
            for host in hosts:
                if not _ssh_reachable(host, ssh_port, timeout_s=3):
                    errs.append(
                        f"ssh to {host}:{ssh_port} unreachable — "
                        f"check --ssh-port, SSH daemon, and key-based login"
                    )

    # YAML files exist & parse.
    for label, p in (("--config", config), ("--algo-config", algo_config)):
        if not p.is_file():
            errs.append(f"{label} not found: {p}")
            continue
        try:
            import yaml

            yaml.safe_load(p.read_text())
        except Exception as e:  # pragma: no cover — yaml errors are obvious
            errs.append(f"{label} unparseable YAML ({p}): {e}")

    # ssh binary present.
    if not _which("ssh"):
        errs.append("ssh not found on PATH")

    return errs


def _read_hosts(hostfile: Path) -> list[str]:
    hosts = []
    for raw in hostfile.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        hosts.append(line.split()[0])  # first whitespace-delimited field = IP
    return hosts


def _ssh_reachable(host: str, port: int, *, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except (OSError, TimeoutError):
        return False


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


# --- main orchestration ------------------------------------------------


def main(argv: list[str]) -> int:
    # Support `--` to split launch-level args from grpo.py extras.
    if "--" in argv:
        dash_idx = argv.index("--")
        launch_argv = argv[:dash_idx]
        extra_argv = argv[dash_idx + 1 :]
    else:
        launch_argv = argv
        extra_argv = []

    parser = _build_parser()

    # Catch a very common new-user mistake: putting a forge-launch flag
    # (``--keep-workers`` / ``--skip-preflight`` / ``--hostfile`` / ...)
    # AFTER the ``--`` separator.  Everything after ``--`` is forwarded
    # to hydra as OmegaConf overrides, which means a leading ``--`` trips
    # hydra's grammar lexer and the user sees a cryptic
    # ``LexerNoViableAltException``.  Produce a concrete, actionable
    # error instead.
    launch_flag_names = {
        act.option_strings[0] for act in parser._actions if act.option_strings
    }
    misplaced = [a for a in extra_argv if a.startswith("--") and a in launch_flag_names]
    if misplaced:
        print(
            "=" * 64,
            " forge launch: misplaced flag(s)",
            "=" * 64,
            sep="\n",
            file=sys.stderr,
        )
        for a in misplaced:
            print(
                f"  ✗ {a!r} is a `forge launch` option, not a hydra\n"
                f"    override.  Move it BEFORE the `--` separator.\n"
                f"    Example:\n"
                f"      python -m forge launch config.yaml {a} \\\n"
                f"          --hostfile hostfile.txt \\\n"
                f"          -- allocation_mode=vllm:d1p1t1+d4p1t1",
                file=sys.stderr,
            )
        return 2

    args = parser.parse_args(launch_argv)

    config = Path(args.config).resolve()
    algo_config = Path(args.algo_config).resolve()
    train_script = Path(args.train_script).resolve()

    hostfile = (
        Path(args.hostfile).resolve()
        if args.hostfile is not None
        else (FORGE_ROOT / "forge" / "configs" / "hostfile.txt")
    )

    # Resolve launcher_impl: CLI flag > YAML launcher.launcher_impl >
    # default "bash".  Keeping the precedence identical to other
    # fields (CLI > YAML > default) avoids a second rule users have
    # to remember.
    launcher_impl = args.launcher_impl or _read_launcher_impl_from_yaml(config)

    # --- Pre-flight ----------------------------------------------------
    if not args.skip_preflight:
        errs = _preflight_checks(
            hostfile=hostfile,
            config=config,
            algo_config=algo_config,
            ssh_port=args.ssh_port,
            worker_port=args.worker_port,
        )
        if errs:
            print("=" * 64, file=sys.stderr)
            print(" forge launch: pre-flight FAILED", file=sys.stderr)
            print("=" * 64, file=sys.stderr)
            for e in errs:
                print(f"  ✗ {e}", file=sys.stderr)
            print(
                "\n  Re-run with --skip-preflight to bypass these checks\n"
                "  (not recommended for first runs).",
                file=sys.stderr,
            )
            return 2

    # Resolve driver host: second non-empty line of hostfile, fallback
    # first.  Matches run_multinode.sh behavior for backward compat.
    hosts = _read_hosts(hostfile)
    driver = hosts[1] if len(hosts) >= 2 else hosts[0]

    # Pre-build the workers string for grpo.py's --bare-metal-workers arg.
    workers_arg = ",".join(f"tcp://{h}:{args.worker_port}" for h in hosts)

    print("=" * 64)
    print(" forge launch: multi-node GRPO")
    print("=" * 64)
    print(f"  launcher config : {config}")
    print(f"  algo config     : {algo_config}")
    print(f"  hostfile        : {hostfile}")
    print(f"  driver host     : {driver}")
    print(f"  workers         : {workers_arg}")
    print(f"  launcher_impl   : {launcher_impl}")
    print(
        f"  steps           : {args.steps if args.steps is not None else '(YAML default)'}"
    )
    print("=" * 64, flush=True)

    # --- Start workers -------------------------------------------------
    # Two interchangeable paths:
    #
    # - ``bash``   : legacy worker_manager.sh start/stop.  Battle-tested,
    #               kept as default during the migration window.
    # - ``ssh_job``: Monarch-native ForgeSSHJob.apply() / _kill().  Same
    #               SSH commands, same env, same worker process -- just
    #               managed inside the Python CLI instead of a 300-line
    #               shell script.  Forward-compatible with SlurmJob /
    #               KubernetesJob down the line.
    worker_fleet: _WorkerFleet
    if launcher_impl == "ssh_job":
        worker_fleet = _SSHJobFleet(
            hostfile=hostfile,
            worker_port=args.worker_port,
            ssh_port=args.ssh_port,
            cann=args.cann,
        )
    elif launcher_impl == "bash":
        worker_fleet = _BashFleet(
            hostfile=hostfile,
            worker_port=args.worker_port,
            ssh_port=args.ssh_port,
            cann=args.cann,
        )
    else:
        print(
            f"[launch] unknown launcher_impl={launcher_impl!r} "
            f"(expected 'bash' or 'ssh_job')",
            file=sys.stderr,
        )
        return 2

    rc = worker_fleet.start()
    if rc != 0:
        print(
            f"[launch] worker start failed (rc={rc}).  Aborting.",
            file=sys.stderr,
        )
        return rc

    # From here on, workers exist remotely and must be cleaned up on
    # any exit path, including Ctrl+C and crashes.
    try:
        _install_cleanup_handlers(
            worker_fleet=worker_fleet,
            keep_workers=args.keep_workers,
        )

        # --- Run training via SSH to driver ----------------------------
        train_rc = _run_driver_training(
            args=args,
            driver=driver,
            hostfile=hostfile,
            config=config,
            algo_config=algo_config,
            train_script=train_script,
            workers_arg=workers_arg,
            extra_argv=extra_argv,
        )
    finally:
        if not args.keep_workers:
            worker_fleet.stop()

    return train_rc


# --- step handlers -----------------------------------------------------


def _run_worker_mgr(
    action: str,
    *,
    hostfile: Path,
    worker_port: int,
    ssh_port: int,
    cann: str | None = None,
) -> int:
    cmd: list[str] = [
        "bash",
        str(WORKER_MGR),
        action,
        "--hostfile",
        str(hostfile),
        "--port",
        str(worker_port),
        "--ssh-port",
        str(ssh_port),
    ]
    if action == "start":
        assert cann is not None
        cmd += ["--cann", cann, "--areal-root", str(FORGE_ROOT)]
    return subprocess.call(cmd)


def _stop_workers(*, hostfile: Path, worker_port: int, ssh_port: int) -> None:
    print("[launch] stopping workers ...", flush=True)
    rc = _run_worker_mgr(
        "stop",
        hostfile=hostfile,
        worker_port=worker_port,
        ssh_port=ssh_port,
    )
    if rc != 0:
        print(
            f"[launch] worker stop returned rc={rc}; some remote procs "
            f"may still be live.  Run `bash {WORKER_MGR} stop --hostfile "
            f"{hostfile}` manually if needed.",
            file=sys.stderr,
            flush=True,
        )


def _install_cleanup_handlers(
    *,
    worker_fleet: _WorkerFleet,
    keep_workers: bool,
) -> None:
    """Route SIGINT / SIGTERM to ``worker_fleet.stop()``.

    We don't rely on just ``finally`` because a raw Ctrl+C during a
    long-running ssh call can produce a partial teardown.  Explicit
    signal handler translates those signals into a clean exit path.
    """
    if keep_workers:
        return

    def _handler(signum, _frame):
        print(
            f"\n[launch] caught signal {signum} — cleaning up workers ...",
            file=sys.stderr,
            flush=True,
        )
        worker_fleet.stop()
        sys.exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# --- worker lifecycle backends ----------------------------------------


class _WorkerFleet:
    """Minimal protocol: ``start`` and ``stop``.

    Concrete backends (:class:`_BashFleet`, :class:`_SSHJobFleet`) share
    the same constructor signature so the CLI can pick one based on the
    ``launcher_impl`` setting.  No real ABC on purpose -- the surface
    is two methods and the indirection is internal to this file.
    """

    def start(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _BashFleet(_WorkerFleet):
    """Legacy ``worker_manager.sh start/stop`` path.

    Kept as default during the migration window.  Exact behavior as
    the pre-refactor CLI: shell out to bash, rely on its per-host
    TCP-probe loop (up to 60s) to decide readiness.
    """

    def __init__(
        self,
        *,
        hostfile: Path,
        worker_port: int,
        ssh_port: int,
        cann: str,
    ) -> None:
        self._hostfile = hostfile
        self._worker_port = worker_port
        self._ssh_port = ssh_port
        self._cann = cann

    def start(self) -> int:
        return _run_worker_mgr(
            "start",
            hostfile=self._hostfile,
            worker_port=self._worker_port,
            ssh_port=self._ssh_port,
            cann=self._cann,
        )

    def stop(self) -> None:
        _stop_workers(
            hostfile=self._hostfile,
            worker_port=self._worker_port,
            ssh_port=self._ssh_port,
        )


class _SSHJobFleet(_WorkerFleet):
    """Monarch-native :class:`ForgeSSHJob` lifecycle.

    Drop-in replacement for :class:`_BashFleet`: starts the same
    ``run_worker_loop_forever`` workers via the same SSH command,
    just managed by Monarch's ``JobTrait`` protocol so the API
    lines up with ``SlurmJob`` / ``KubernetesJob`` for future
    migrations.

    Also faster: the POC measured ~4-12s per host vs the bash path's
    60s ceiling, because we attach immediately after apply() returns
    instead of polling each host's TCP port in sequence.
    """

    def __init__(
        self,
        *,
        hostfile: Path,
        worker_port: int,
        ssh_port: int,
        cann: str,
        conda_env: str = "monarch_ascend",
        conda_bin: str = "/root/miniconda3/bin/conda",
    ) -> None:
        self._hostfile = hostfile
        self._worker_port = worker_port
        self._ssh_port = ssh_port
        self._cann = cann
        self._conda_env = conda_env
        self._conda_bin = conda_bin
        self._job = None  # type: ignore[assignment]

    def start(self) -> int:
        # Import lazily so ``forge launch --help`` works without a
        # Monarch install (useful in CI unit tests).
        try:
            from forge.provisioner_ssh import ForgeSSHJob
        except ImportError as e:
            print(
                f"[launch] launcher_impl=ssh_job needs Monarch importable: {e!r}",
                file=sys.stderr,
            )
            return 2

        hosts = _read_hosts(self._hostfile)
        if not hosts:
            print("[launch] ssh_job: empty hostfile", file=sys.stderr)
            return 2

        ssh_args = [
            "-p",
            str(self._ssh_port),
            "-o",
            "StrictHostKeyChecking=no",
        ]
        job = ForgeSSHJob(
            cann_home=self._cann,
            conda_env=self._conda_env,
            conda_bin=self._conda_bin,
            areal_root=str(FORGE_ROOT),
            python_exe="python",
            ssh_args=ssh_args,
            monarch_port=self._worker_port,
        )
        # The mesh name here is ONLY used internally by the Job for
        # grouping; it does not surface to BareMetalLauncher, which
        # attaches independently via attach_to_workers.
        job.add_mesh("forge_bare_metal", hosts)
        try:
            job.apply(client_script=None)
        except Exception as e:  # noqa: BLE001
            print(
                f"[launch] ssh_job: apply() failed: {e!r}",
                file=sys.stderr,
            )
            return 10
        self._job = job

        # Wait for each worker to listen on its port (same readiness
        # criterion the bash path uses, but parallel instead of serial).
        # TCP reachable == ``run_worker_loop_forever`` has progressed
        # past address bind.
        deadline = time.time() + 60
        remaining = {h: self._worker_port for h in hosts}
        while remaining and time.time() < deadline:
            ready_now = [
                h
                for h in remaining
                if _ssh_reachable(h, self._worker_port, timeout_s=2)
            ]
            for h in ready_now:
                del remaining[h]
            if remaining:
                time.sleep(1)
        if remaining:
            print(
                f"[launch] ssh_job: workers never listening: {list(remaining)}",
                file=sys.stderr,
            )
            return 11
        return 0

    def stop(self) -> None:
        if self._job is None:
            return
        print("[launch] ssh_job: stopping workers ...", flush=True)
        try:
            self._job._kill()
        except Exception as e:  # noqa: BLE001
            print(
                f"[launch] ssh_job: _kill raised {e!r}; "
                f"some remote procs may still be live",
                file=sys.stderr,
            )


def _read_launcher_impl_from_yaml(config_path: Path) -> str:
    """Parse ``launcher.launcher_impl`` out of the launcher YAML.

    Falls back to ``"bash"`` on any read/parse error: the launcher
    YAML is read a second time here (first time is pre-flight's
    parse check) on purpose, so a malformed YAML doesn't silently
    flip us onto the ssh_job path.  Default-bash matches
    :class:`LauncherConfig.launcher_impl`.
    """
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return "bash"
    launcher_block = data.get("launcher") or {}
    if not isinstance(launcher_block, dict):
        return "bash"
    val = launcher_block.get("launcher_impl", "bash")
    return str(val) if val else "bash"


def _run_driver_training(
    *,
    args: argparse.Namespace,
    driver: str,
    hostfile: Path,
    config: Path,
    algo_config: Path,
    train_script: Path,
    workers_arg: str,
    extra_argv: list[str],
) -> int:
    """SSH into the driver host and run ``python -m forge.apps.grpo``."""
    # Build the forge.apps.grpo command line.
    grpo_argv = [
        "python",
        "-m",
        "forge.apps.grpo",
        str(train_script),
        "--config",
        str(algo_config),
        "--forge-config",
        str(config),
        "--bare-metal-workers",
        workers_arg,
        "--bare-metal-master-addr",
        driver,
    ]
    if args.backend:
        grpo_argv += ["--backend", args.backend]
    if args.model_name:
        grpo_argv += ["--model-name", args.model_name]
    if args.model_flavor:
        grpo_argv += ["--model-flavor", args.model_flavor]
    if args.model:
        grpo_argv += [f"actor.path={args.model}"]
    if args.steps is not None:
        grpo_argv += [f"+total_train_steps={args.steps}"]
    grpo_argv += extra_argv

    # Compose the remote bash one-liner.  These env exports match the
    # production run_multinode.sh block; keeping parity is essential
    # for the smoke-regression guarantee across the two entrypoints.
    env_lines = [
        f"source {args.cann}/set_env.sh 2>/dev/null",
        f"ASCEND_ROOT=$(dirname {args.cann})",
        '[[ -f "$ASCEND_ROOT/nnal/atb/set_env.sh" ]] && '
        'source "$ASCEND_ROOT/nnal/atb/set_env.sh"',
        'eval "$(/root/miniconda3/bin/conda shell.bash hook)"',
        "conda activate monarch_ascend",
        "export VLLM_USE_MODELSCOPE=true",
        "export HF_ENDPOINT=https://hf-mirror.com",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        'export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"',
        'export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-1}"',
        'export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"',
        "export MONARCH_HIXL_TRANSPORT=roce",
        "export HCCL_INTRA_ROCE_ENABLE=1",
        "export HCCL_CONNECT_TIMEOUT=120",
        "export HCCL_NPU_SOCKET_PORT_RANGE=60000-60255",
        "export TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0",
        "export TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:0",
    ]

    # Forward user-set forge/torchstore env vars so
    # forge.engines.weight_sync._config_resolver can pick them up as
    # overrides (and emit the deprecation warning).  Vars unset in the
    # caller's shell are left unset, letting the YAML value win.
    for var in (
        "FORGE_WEIGHT_SYNC",
        "FORGE_WEIGHT_SYNC_BACKEND",
        "FORGE_SHARD_PUBLISH",
        "FORGE_MESH_PLACEMENT",
        "FORGE_STORAGE_HOST_MESH",
        "FORGE_BCAST_SRC_VOL_IDX",
        "FORGE_BCAST_MASTER_PORT",
        "FORGE_SUPPRESS_DEPRECATION",
        "TORCHSTORE_STORAGE_NPU_BASE",
        "TORCHSTORE_MONARCH_RDMA_POOL_MB",
    ):
        v = os.environ.get(var)
        if v is not None:
            # Shell-escape via Python's shlex-style quoting.
            import shlex

            env_lines.append(f"export {var}={shlex.quote(v)}")

    remote_cmd_parts = [
        *env_lines,
        f"cd {FORGE_ROOT}",
        _shellify(grpo_argv) + " 2>&1 | tee /tmp/forge_multinode.log",
    ]
    remote_cmd = "; ".join(remote_cmd_parts)

    ssh_cmd = [
        "ssh",
        "-p",
        str(args.ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        f"root@{driver}",
        remote_cmd,
    ]

    start = time.time()
    print(f"[launch] training on {driver} (started at {time.strftime('%H:%M:%S')})")
    rc = subprocess.call(ssh_cmd)
    elapsed = time.time() - start

    print("=" * 64)
    if rc == 0:
        print(f" forge launch: training OK ({elapsed:.0f}s)")
    else:
        print(f" forge launch: training FAILED rc={rc} ({elapsed:.0f}s)")
        print(_error_hint(rc))
    print("=" * 64)
    return rc


def _shellify(argv: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(a) for a in argv)


def _error_hint(rc: int) -> str:
    """Best-effort error code translation.

    Pulled from ``forge_multinode.log`` post-hoc would be ideal, but
    for now we give a generic pointer to the doc.  Wire in keyword
    grepping of the log if this turns out to be useful.
    """
    return (
        f"  Hint: rc={rc}. Inspect /tmp/forge_multinode.log on the driver host.\n"
        f"  Common failure modes are documented in\n"
        f"    forge/docs/weight_sync.md §7 (known-pain ledger)\n"
        f"  with each CANN/HCCL/HiXL error code mapped to a root cause."
    )
