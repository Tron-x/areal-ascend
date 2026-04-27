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
import shlex
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

_LAUNCHER_LAYER_KEYS: frozenset[str] = frozenset(
    {"launcher_preset", "roles", "mode", "titan", "inference"}
)
"""Top-level algo-YAML keys owned by ``forge launch`` (not the inner app).

These describe infrastructure intent + dispatch intent and are
consumed by the preset composer (``forge/cli/presets.py``) and the
mode dispatcher in this module.  grpo.py's ``load_expr_config``
merges against a strict ``GRPOConfig`` dataclass which rejects
unknown top-level keys, so we strip these before writing the
algo-side temp YAML.  ``cluster`` is *not* stripped -- it maps to
AReaL's existing ``ClusterSpecConfig`` and is legitimately part of
GRPOConfig.

Why each key is here:

* ``launcher_preset``  -- preset-resolution input
* ``roles``            -- per-role devices override merged into
                          the preset
* ``mode``             -- selects which ``forge.apps.<name>`` runs
                          on the driver (``grpo`` vs.
                          ``titan-pretrain`` vs. ``inference-bench``)
* ``titan``            -- B-full algo block (config / cwd /
                          procs_per_host / overrides) consumed by
                          ``forge.apps.titan_pretrain``, never by
                          ``forge.apps.grpo``
* ``inference``        -- vLLM benchmark block (model / tp_size /
                          benchmark) consumed by
                          ``forge.apps.inference_bench``, never by
                          ``forge.apps.grpo``
"""

_VALID_MODES: frozenset[str] = frozenset({"grpo", "titan-pretrain", "inference-bench"})
"""Recognised values for ``mode:`` / ``--mode``.

* ``grpo``            -- default; runs ``forge.apps.grpo`` on the
                         driver.  Worker profile defaults to
                         ``hixl-coexist`` (HCCL + HiXL coexistence).
* ``titan-pretrain``  -- B-full TorchTitan pretrain entry; runs
                         ``forge.apps.titan_pretrain`` on the driver.
                         Worker profile defaults to ``pure-training``.
* ``inference-bench`` -- 2-node vLLM benchmark entry; runs
                         ``forge.apps.inference_bench`` on the
                         driver.  Worker profile defaults to
                         ``pure-training`` (no HiXL/torchstore env
                         needed for standalone inference).
"""

_MODE_DEFAULT_PROFILE: dict[str, str] = {
    "grpo": "hixl-coexist",
    "titan-pretrain": "pure-training",
    "inference-bench": "pure-training",
}
"""Default worker env profile for each mode.

Picked when neither ``--profile`` (currently inferred only via
preset YAML) nor ``launcher.profile`` in the preset overrides it.
The defaults match each mode's transport requirements -- GRPO needs
the HiXL/torchstore env to do weight sync in-process; titan-pretrain
just wants vanilla HCCL with the optimal HCCS+RoCE auto-routing.
"""


# --- parsing -----------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge launch",
        description="Single-command multi-node GRPO launch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Extra args after `--` are forwarded verbatim to "
            "`python -m forge.apps.grpo` (OmegaConf overrides, etc.).\n\n"
            "Preferred (two-layer YAML, algo author view):\n"
            "  python -m forge launch \\\n"
            "      examples/math/gsm8k_grpo_npu.yaml \\\n"
            "      --steps 3 --model /path/to/snapshot\n"
            "  (the algo YAML selects a cluster preset via\n"
            "   `launcher_preset:` under forge/configs/clusters/)\n\n"
            "Legacy one-layer (raw launcher YAML, for infra bring-up):\n"
            "  python -m forge launch \\\n"
            "      forge/configs/clusters/2node_colocated.yaml \\\n"
            "      --algo-config examples/math/gsm8k_grpo_npu.yaml \\\n"
            "      --steps 3 --model /path/to/snapshot\n"
        ),
    )
    p.add_argument(
        "config",
        help=(
            "Algorithm YAML with a top-level ``launcher_preset:`` key "
            "(e.g. examples/math/gsm8k_grpo_npu.yaml) OR a raw launcher "
            "YAML with a top-level ``launcher:`` block (e.g. "
            "forge/configs/clusters/2node_colocated.yaml -- used for "
            "bring-up / debugging a new cluster).  Auto-detected.  The "
            "algo-YAML path is preferred for production runs because it "
            "keeps infra-only fields (pool, colocate, ...) out of the "
            "algorithm author's view."
        ),
    )
    p.add_argument(
        "--algo-config",
        default=None,
        help=(
            "Override the algorithm-side YAML path.  In the new two-layer "
            "mode this is usually unnecessary -- the positional `config` "
            "is itself the algo YAML.  In legacy mode (positional `config` "
            "is a raw launcher YAML), defaults to "
            "examples/math/gsm8k_grpo_npu.yaml.  Always forwarded as "
            "grpo.py's --config so it still carries allocation_mode / "
            "reward / trainer settings."
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
        "--duration",
        type=float,
        default=None,
        help=(
            "Inference-bench only: override "
            "``inference.benchmark.duration_seconds`` from the algo YAML.  "
            "Ignored by grpo / titan-pretrain modes."
        ),
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
        "--mode",
        default=None,
        choices=sorted(_VALID_MODES),
        help=(
            "Driver dispatch mode.  ``grpo`` (default) runs "
            "``forge.apps.grpo``; ``titan-pretrain`` runs "
            "``forge.apps.titan_pretrain`` for pure TorchTitan "
            "pretraining (B-full, no GRPO loop / vLLM / "
            "torchstore); ``inference-bench`` runs "
            "``forge.apps.inference_bench`` for a 2-node vLLM "
            "throughput benchmark (one TP=N replica per host).  "
            "If omitted, read from the algo YAML's top-level "
            "``mode:`` key, falling back to ``grpo``.  Mode "
            "determines worker env profile (grpo -> hixl-coexist; "
            "titan-pretrain / inference-bench -> pure-training) "
            "and driver env exports (the pure-training modes skip "
            "the HiXL/torchstore vars)."
        ),
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
        "--sync",
        action="store_true",
        help=(
            "Before starting workers, push local source trees to every "
            "host listed in --hostfile via `forge sync`.  Safeguards "
            "against stale code on the driver (the cause of the "
            "`ConfigKeyError: Key 'reward' not in 'GRPOConfig'` class of "
            "failures).  Defaults to syncing `forge` + `areal`; override "
            "with --sync-path.  Honors --ssh-port."
        ),
    )
    p.add_argument(
        "--sync-path",
        action="append",
        default=None,
        dest="sync_paths",
        help=(
            "Extra path to include in `--sync` (relative to FORGE_ROOT or "
            "absolute).  Repeatable.  Implies --sync.  If any --sync-path "
            "is given the defaults are replaced, not extended -- pass "
            "`--sync-path forge --sync-path areal --sync-path examples` to "
            "keep the built-ins plus `examples`."
        ),
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
    hosts_override: list[str] | None = None,
) -> list[str]:
    """Return a list of error strings; empty list means all green.

    Each check is independent so the user sees every broken thing at
    once rather than one-at-a-time.

    ``hosts_override`` lets callers bypass the hostfile read and
    supply a pre-derived host list (e.g. from ``launcher.pool`` in
    the YAML).  When set, the hostfile-existence check is skipped.
    """
    errs: list[str] = []

    # Host list: prefer override, else read from hostfile.
    hosts: list[str] = []
    if hosts_override is not None:
        hosts = hosts_override
        if not hosts:
            errs.append("launcher.pool parsed empty (no valid host entries)")
    elif not hostfile.is_file():
        errs.append(f"hostfile not found: {hostfile}")
    else:
        hosts = _read_hosts(hostfile)
        if not hosts:
            errs.append(f"hostfile has no non-comment hosts: {hostfile}")

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

    positional = Path(args.config).resolve()

    # Two-layer YAML auto-detect (see forge/cli/presets.py):
    #
    # - If the positional arg is an *algo* YAML -- it has a top-level
    #   ``launcher_preset:`` key -- resolve the preset, compose the
    #   algo's role-device overrides into it, write the result to a
    #   temp YAML, and use that as the launcher config.  The original
    #   positional arg doubles as --algo-config.
    # - If the positional arg already has a ``launcher:`` block, it's
    #   a raw launcher YAML (legacy / debugging).  Use it verbatim.
    # - Anything else is an author error -- fail early with a pointer.
    #
    # This keeps algo authors off the infra-field surface (pool,
    # colocate, weight_sync, ...) without breaking any existing
    # operator / CI invocation that still passes a launcher YAML
    # directly.
    config, algo_config = _resolve_configs(positional, algo_override=args.algo_config)

    # Resolve mode: CLI flag > algo YAML's ``mode:`` > default ``grpo``.
    # Resolved BEFORE pre-flight so subsequent steps (profile pick,
    # driver dispatch) all see the same value.
    #
    # IMPORTANT: read from ``positional`` (the original algo YAML),
    # NOT from ``algo_config``.  ``_resolve_configs`` returns a
    # sanitized algo YAML with launcher-layer keys stripped --
    # including ``mode`` itself -- so reading ``algo_config`` here
    # would always miss it.
    mode = args.mode or _read_mode_from_algo_yaml(positional) or "grpo"

    # Resolve worker env profile: launcher YAML's ``launcher.profile``
    # (preset-authoritative) > mode-derived default.  No CLI flag
    # for profile yet -- if a user needs to override per-mode they
    # author / pick a different preset, which keeps the
    # "preset = infra decisions" contract clean.
    profile = _read_profile_from_yaml(config) or _MODE_DEFAULT_PROFILE[mode]

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
    # NOTE: ssh_job and bash are both profile-aware as of the
    # ForgeSSHJob refactor (see ``forge/provisioner_ssh.py::
    # _profile_env_parts``); no special-casing needed here.

    # R1.5c: prefer ``launcher.pool`` from the YAML over the hostfile
    # when both are available.  The pool is the authoritative cluster
    # description (ports + driver tag + hardware tier); the hostfile
    # is the legacy flat list.  When both are authored and disagree
    # we warn and trust the pool -- the hostfile's most likely explanation
    # is "operator edited the pool and forgot to re-export hostfile.txt".
    pool_hosts, pool_driver, pool_worker_port = _read_pool_from_yaml(config)
    using_pool = bool(pool_hosts)

    # --- Pre-flight ----------------------------------------------------
    if not args.skip_preflight:
        # With a pool, hostfile is optional -- pre-flight still runs
        # against the pool-derived host list so SSH reachability is
        # verified the same way.
        errs = _preflight_checks(
            hostfile=hostfile,
            config=config,
            algo_config=algo_config,
            ssh_port=args.ssh_port,
            worker_port=args.worker_port,
            hosts_override=pool_hosts if using_pool else None,
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

    # Resolve driver + worker list.  Pool first (explicit driver tag);
    # hostfile second (legacy "second non-empty line is driver"
    # heuristic).  We continue to populate ``hosts`` for the
    # downstream ssh-fleet / sync paths -- they still read one list.
    if using_pool:
        hosts = pool_hosts
        driver = pool_driver or hosts[0]
        if pool_worker_port is not None and pool_worker_port != args.worker_port:
            # CLI --worker-port wins (historical default) but mention
            # the divergence so users notice stale overrides.
            print(
                f"[launch] NOTE: pool declares worker_port="
                f"{pool_worker_port}; CLI --worker-port="
                f"{args.worker_port} takes precedence.",
                flush=True,
            )

        # ``_BashFleet`` / ``_SSHJobFleet`` / ``_run_driver_training`` /
        # ``_stop_workers`` all take a ``hostfile: Path`` and call
        # ``worker_manager.sh --hostfile ...`` on the remote.  In
        # pool-only mode the user might not have a hostfile on disk,
        # so materialize one next to the YAML.  File is deterministic
        # (``<config>.pool.hostfile``) so re-runs reuse it and stale
        # entries are visible to the operator.
        pool_hostfile = config.with_suffix(config.suffix + ".pool.hostfile")
        pool_hostfile.write_text(
            "# Auto-generated by `forge launch` from launcher.pool\n"
            "# Source: "
            + str(config)
            + "\n"
            + "\n".join(f"{h} slots=8" for h in hosts)
            + "\n"
        )
        hostfile = pool_hostfile
    else:
        hosts = _read_hosts(hostfile)
        driver = hosts[1] if len(hosts) >= 2 else hosts[0]

    # Pre-build the workers string for grpo.py's --bare-metal-workers arg.
    workers_arg = ",".join(f"tcp://{h}:{args.worker_port}" for h in hosts)

    print("=" * 64)
    print(f" forge launch: multi-node {mode}")
    print("=" * 64)
    print(f"  launcher config : {config}")
    print(f"  algo config     : {algo_config}")
    print(f"  mode            : {mode}")
    print(f"  worker profile  : {profile}")
    print(f"  host source     : {'pool (yaml)' if using_pool else 'hostfile'}")
    print(f"  hostfile        : {hostfile}")
    print(f"  driver host     : {driver}")
    print(f"  workers         : {workers_arg}")
    print(f"  launcher_impl   : {launcher_impl}")
    print(
        f"  steps           : {args.steps if args.steps is not None else '(YAML default)'}"
    )
    print("=" * 64, flush=True)

    # --- Optional source sync ------------------------------------------
    # Before starting workers, optionally push local source trees to
    # every host so the driver + workers all run identical code.  The
    # canonical failure this prevents is the driver's
    # ``areal/api/cli_args.py`` being one schema-change behind the
    # local checkout and blowing up mid-init with a ``ConfigKeyError``
    # -- we lost two full reruns to this class of drift before
    # introducing ``--sync``.
    #
    # --sync-path implies --sync (users typically only pass the paths
    # list, which would otherwise be silently ignored).
    #
    # ``--sync`` is redundant when ``launcher_impl: ssh_job`` runs
    # with ``mount_code=True`` (the default): ``_SSHJobFleet.start()``
    # exposes the launcher's checkout to workers via FUSE
    # ``remote_mount``, which is drift-free and skips the ``tar | ssh``
    # transfer entirely.  We emit a hint instead of silently no-op'ing
    # so users discover the better path; we still run the sync (it's
    # harmless -- just wasteful) in case the user has bash workers
    # mixed in or is debugging.
    if (args.sync or args.sync_paths) and launcher_impl == "ssh_job":
        print(
            "[launch] note: --sync is redundant under "
            "``launcher_impl: ssh_job`` -- workers already see "
            "the launcher's source via FUSE ``remote_mount``.  "
            "Drop --sync from your invocation unless you also "
            "have bash-managed workers."
        )

    if args.sync or args.sync_paths:
        from forge.cli.sync import sync_paths_to_hosts

        print("[launch] --sync: pushing source trees to all hosts ...")
        sync_results = sync_paths_to_hosts(
            hosts,
            paths=args.sync_paths,
            ssh_port=args.ssh_port,
            parallel=max(2, len(hosts)),
        )
        failed = [r for r in sync_results if not r.ok]
        for r in sync_results:
            tag = "ok  " if r.ok else "FAIL"
            print(f"  [{tag}] {r.host:<20} {r.elapsed_s:5.1f}s")
            if not r.ok and r.error:
                print(f"         -> {r.error}")
        if failed:
            print(
                f"[launch] --sync: {len(failed)} host(s) failed; aborting.",
                file=sys.stderr,
            )
            return 2

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
            profile=profile,
        )
    elif launcher_impl == "bash":
        worker_fleet = _BashFleet(
            hostfile=hostfile,
            worker_port=args.worker_port,
            ssh_port=args.ssh_port,
            cann=args.cann,
            profile=profile,
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
        # Mode dispatch: GRPO, titan-pretrain, and inference-bench go
        # through different apps with different env/CLI surfaces.
        # Keeping them as distinct functions keeps the GRPO path's
        # complex flag forwarding (model / model_name / model_flavor /
        # forge env vars) separate from the much simpler
        # algo-YAML-driven invocations of the standalone modes.
        # When the worker fleet manages its own code-sync (e.g.
        # ``_SSHJobFleet`` with ``mount_code=True``), use the
        # worker-visible mount path for the driver's ``cd`` /
        # ``PYTHONPATH`` -- otherwise the driver reads its local
        # checkout, which can drift from the launcher's checkout
        # and reintroduce the stale-code bug class ``--sync`` /
        # ``forge.cli.sync`` was originally written to defeat.
        # Bash fleet has no opinion -- defaults to FORGE_ROOT.
        driver_code_root = getattr(worker_fleet, "code_root", str(FORGE_ROOT))

        if mode == "titan-pretrain":
            # titan-pretrain reads the ``titan:`` block out of the
            # algo YAML, which is stripped from ``algo_config`` by
            # _resolve_configs (because GRPOConfig rejects it).  Pass
            # the ORIGINAL ``positional`` here -- it lives on disk at
            # the same path on every host (algo YAMLs under
            # examples/ are checked-in code, not /tmp scratch).
            train_rc = _run_driver_titan_pretrain(
                args=args,
                driver=driver,
                algo_config=positional,
                workers_arg=workers_arg,
                code_root=driver_code_root,
            )
        elif mode == "inference-bench":
            # Same reasoning as titan-pretrain: ``inference:`` is
            # stripped from ``algo_config`` by _resolve_configs, so
            # the inference_bench driver needs the ORIGINAL algo
            # YAML path to read its block back.
            train_rc = _run_driver_inference_bench(
                args=args,
                driver=driver,
                algo_config=positional,
                workers_arg=workers_arg,
                code_root=driver_code_root,
            )
        else:
            train_rc = _run_driver_training(
                args=args,
                driver=driver,
                hostfile=hostfile,
                config=config,
                algo_config=algo_config,
                train_script=train_script,
                workers_arg=workers_arg,
                extra_argv=extra_argv,
                code_root=driver_code_root,
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
    profile: str | None = None,
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
        # ``forge launch`` is the GRPO + weight-sync entrypoint, so it
        # always wants the ``hixl-coexist`` worker env (HiXL RoCE +
        # HCCL port range + torchstore RDMA pool).  Skipping the flag
        # would silently fall through to ``worker_manager.sh``'s new
        # default of ``pure-training`` -- correct for B-mini, fatal
        # for HiXL Connect on the GRPO path.
        if profile is not None:
            cmd += ["--profile", profile]
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
    """``worker_manager.sh start/stop`` path with explicit env profile.

    Shells out to bash; relies on the per-host TCP-probe loop (up to
    60s) for readiness.  Caller picks the worker env profile
    (``hixl-coexist`` for GRPO, ``pure-training`` for titan-pretrain).
    """

    def __init__(
        self,
        *,
        hostfile: Path,
        worker_port: int,
        ssh_port: int,
        cann: str,
        profile: str,
    ) -> None:
        self._hostfile = hostfile
        self._worker_port = worker_port
        self._ssh_port = ssh_port
        self._cann = cann
        self._profile = profile

    def start(self) -> int:
        return _run_worker_mgr(
            "start",
            hostfile=self._hostfile,
            worker_port=self._worker_port,
            ssh_port=self._ssh_port,
            cann=self._cann,
            profile=self._profile,
        )

    def stop(self) -> None:
        _stop_workers(
            hostfile=self._hostfile,
            worker_port=self._worker_port,
            ssh_port=self._ssh_port,
        )


DEFAULT_MOUNT_ROOT: str = "/root/AReaL_remote"
"""Worker-visible path where launcher source is FUSE-mounted by
:class:`_SSHJobFleet` when ``mount_code=True``.

We deliberately pick a *different* path from the launcher's local
checkout (``/root/AReaL`` aka :data:`FORGE_ROOT`) so the FUSE mount
does NOT shadow whichever local copy the worker host already has.
Two practical wins:

* The launcher itself runs on one of the worker hosts (the same
  python process registers the SSHJob and is also a worker host).
  Shadowing ``/root/AReaL`` on the launcher would mask the source we
  read with -- a self-foot-shooting hazard on every launch.
* On Ctrl-C the FUSE mount may take a moment to unmount cleanly; in
  the meantime any process whose ``cwd`` is still inside the mount
  prevents teardown.  Keeping the mount under a dedicated path means
  we never accidentally cwd into it from the launcher itself.
"""

DEFAULT_MOUNT_SUBDIRS: tuple[str, ...] = ("forge", "areal")
"""Subdirectories of :data:`FORGE_ROOT` to expose via FUSE on workers.

Mirrors :data:`forge.cli.sync.DEFAULT_PATHS`: these are the only two
editable trees the workers need at import time.  Mounting only the
subset that workers actually import keeps each FUSE setup small
(forge/ + areal/ ~10MB combined vs. 2.5GB for the full repo with
``.venv`` / ``.git``) and makes the per-host mount latency bounded
(~5-10s on our 100GbE testbed).
"""


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

    Profile-aware: the ``profile`` constructor arg is forwarded
    verbatim to :class:`ForgeSSHJob` and selects which env block
    every worker boots with.  See ``forge/provisioner_ssh.py``
    module docstring for the full profile catalog.

    FUSE-backed code sync (``mount_code=True``, default)
    ----------------------------------------------------

    When enabled, ``start()`` registers a Monarch ``remote_mount``
    for every entry of :data:`mount_subdirs` (default ``forge/`` and
    ``areal/``) onto the workers under :data:`mount_root` (default
    ``/root/AReaL_remote``).  After ``apply()`` brings up the workers
    and TCP-readiness passes, ``state()`` triggers the FUSE setup --
    spawning a ``FUSEActor`` mesh on each host and attaching the
    block-transfer pipeline that serves source files on demand.

    Workers and drivers see the launcher's checkout through the FUSE
    mount, not whatever happens to be on the worker host's local
    disk.  This replaces the older :func:`forge.cli.sync.sync_paths_to_hosts`
    (``tar | ssh``) preamble: instead of physically copying the
    source tree to each host before starting workers, we expose it
    declaratively and let the kernel + FUSE driver fetch blocks as
    workers actually open files.

    Trade-off: FUSE mount setup adds 5-10s per launch; the alternative
    ``--sync`` step also takes a comparable amount of time but
    transferred the entire tree (most of which the workers never
    read).  The mount-based path is also drift-free: edits to the
    launcher's checkout become visible on workers immediately,
    without an explicit re-sync.

    Set ``mount_code=False`` to fall back to the legacy non-mount
    behaviour where workers/drivers read from their local
    ``/root/AReaL`` -- caller must then arrange for that path to be
    in sync (typically via ``forge launch --sync`` or out-of-band
    ``rsync``).
    """

    def __init__(
        self,
        *,
        hostfile: Path,
        worker_port: int,
        ssh_port: int,
        cann: str,
        profile: str,
        conda_env: str = "monarch_ascend",
        conda_bin: str = "/root/miniconda3/bin/conda",
        mount_code: bool = True,
        mount_root: str = DEFAULT_MOUNT_ROOT,
        mount_subdirs: tuple[str, ...] = DEFAULT_MOUNT_SUBDIRS,
    ) -> None:
        self._hostfile = hostfile
        self._worker_port = worker_port
        self._ssh_port = ssh_port
        self._cann = cann
        self._profile = profile
        self._conda_env = conda_env
        self._conda_bin = conda_bin
        self._mount_code = bool(mount_code)
        self._mount_root = mount_root.rstrip("/") or "/"
        self._mount_subdirs = tuple(mount_subdirs)
        self._job = None  # type: ignore[assignment]

    @property
    def code_root(self) -> str:
        """Worker-visible path that holds ``forge/`` + ``areal/``.

        Either :attr:`_mount_root` (FUSE) or :data:`FORGE_ROOT`
        (legacy local-checkout) depending on ``mount_code``.  The
        outer launcher reads this to point ``_run_driver_*`` at the
        right ``cd`` / ``PYTHONPATH`` location -- the driver process
        on the chosen driver host needs to see the same source view
        as the workers it orchestrates.
        """
        return self._mount_root if self._mount_code else str(FORGE_ROOT)

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
        # When mount_code is on, point ForgeSSHJob's areal_root at the
        # FUSE mountpoint so the worker preamble's PYTHONPATH and ``cd``
        # both target the soon-to-be-mounted view.  ForgeSSHJob already
        # does ``mkdir -p {areal_root}`` before ``cd``, so the directory
        # not existing yet at apply() time is fine -- the FUSE mount
        # populates it asynchronously inside state().
        worker_areal_root = self._mount_root if self._mount_code else str(FORGE_ROOT)
        job = ForgeSSHJob(
            cann_home=self._cann,
            conda_env=self._conda_env,
            conda_bin=self._conda_bin,
            areal_root=worker_areal_root,
            python_exe="python",
            ssh_args=ssh_args,
            monarch_port=self._worker_port,
            profile=self._profile,
        )
        # The mesh name here is ONLY used internally by the Job for
        # grouping; it does not surface to BareMetalLauncher, which
        # attaches independently via attach_to_workers.
        job.add_mesh("forge_bare_metal", hosts)

        # Register remote_mount entries BEFORE apply().
        # ``remote_mount`` is config-only here: the actual FUSE setup
        # only runs inside ``state()``.  We mount each subdir
        # individually (forge -> mount_root/forge, areal ->
        # mount_root/areal) so the FUSE mount tree mirrors the
        # repo layout from PYTHONPATH=mount_root.
        if self._mount_code:
            for sub in self._mount_subdirs:
                src = FORGE_ROOT / sub
                if not src.is_dir():
                    print(
                        f"[launch] ssh_job: mount source {src} missing; skipping",
                        flush=True,
                    )
                    continue
                target = f"{self._mount_root}/{sub}"
                print(
                    f"[launch] ssh_job: registering FUSE mount "
                    f"{src} -> {target} (transfer_mode=actor)",
                    flush=True,
                )
                # transfer_mode="actor" routes block transfer through
                # Monarch actor messages -- works without Meta-internal
                # TLS certs (the ``rust_tls`` default needs ``fb-tls``
                # cert paths we don't have on bare-metal NPU hosts).
                # python_exe=None tells JobTrait NOT to rewrite the
                # worker's python path to ``{mountpoint}/.venv/bin/python``;
                # we use a conda env at /root/miniconda3 unrelated to
                # the source tree, so the rewrite would be wrong.
                job.remote_mount(
                    source=str(src),
                    mntpoint=target,
                    meshes=["forge_bare_metal"],
                    python_exe=None,
                    transfer_mode="actor",
                )

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

        # Activate FUSE mounts.  state() does both attach + mount; we
        # only care about the mount side here (the driver side does
        # its own attach_to_workers later).  Calling state() now also
        # surfaces FUSE / transport bugs at start time rather than at
        # first import inside the actor process.
        if self._mount_code:
            print(
                "[launch] ssh_job: activating FUSE mounts via state() "
                "(this can take ~5-10s) ...",
                flush=True,
            )
            try:
                job.state(cached_path=None)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[launch] ssh_job: state() failed (FUSE mount): {e!r}",
                    file=sys.stderr,
                )
                return 12
        return 0

    def stop(self) -> None:
        if self._job is None:
            return
        print("[launch] ssh_job: stopping workers ...", flush=True)
        try:
            # ``kill()`` (public) calls ``_mounts.ensure_stopped()``
            # before tearing down workers.  Without the unmount the
            # mount_worker subprocess survives and leaks the FUSE
            # mount across runs.  ``_kill()`` (private) skips this
            # step -- never call it directly when remote_mount is in
            # use.  See JobTrait.kill in monarch/_src/job/job.py.
            self._job.kill()
        except Exception as e:  # noqa: BLE001
            print(
                f"[launch] ssh_job: kill() raised {e!r}; "
                f"some remote procs may still be live",
                file=sys.stderr,
            )


def _resolve_configs(
    positional: Path,
    *,
    algo_override: str | None,
) -> tuple[Path, Path]:
    """Split the positional YAML into ``(launcher_config, algo_config)``.

    Supports three author patterns:

    1. **Two-layer (preferred)**: positional is an algo YAML with a
       top-level ``launcher_preset:`` key.  We resolve the preset
       under ``forge/configs/clusters/``, merge the algo YAML's
       top-level ``roles:`` devices overrides into the preset, write
       the composed launcher block to a temp YAML, and return
       ``(temp, positional)``.  ``--algo-config`` is ignored in this
       mode -- the positional IS the algo config.
    2. **Legacy one-layer**: positional is a raw launcher YAML with a
       top-level ``launcher:`` block.  We return
       ``(positional, --algo-config or DEFAULT_ALGO_YAML)`` so the
       pre-R1.5 invocation continues to work.
    3. **Author error**: neither key present -- raise with a pointer
       to the two-layer doc.

    The composed temp YAML is written to ``/tmp/forge_composed_*.yaml``
    (deterministic per invocation; not reused across runs).  Leaving
    it on disk on purpose -- when a run fails mid-start the operator
    can inspect the exact composed config the launcher saw.
    """
    import tempfile

    import yaml

    try:
        data = yaml.safe_load(positional.read_text()) or {}
    except Exception as e:
        print(
            f"[launch] could not parse positional YAML {positional}: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    # Legacy launcher YAML (debug path) -- positional already has a
    # launcher: block.  Honor --algo-config if given, else fall back
    # to the historical default.
    if isinstance(data, dict) and "launcher" in data:
        algo = (
            Path(algo_override).resolve()
            if algo_override is not None
            else Path(DEFAULT_ALGO_YAML).resolve()
        )
        return positional, algo

    # Two-layer mode -- compose preset + algo overrides into temp file.
    if isinstance(data, dict) and "launcher_preset" in data:
        from forge.cli.presets import (
            PresetError,
            compose_launcher_yaml,
            resolve_cluster_preset,
        )

        preset_name = str(data["launcher_preset"])
        try:
            preset_path = resolve_cluster_preset(
                preset_name, algo_yaml_dir=positional.parent
            )
            preset_data = yaml.safe_load(preset_path.read_text()) or {}

            def _warn_escape_hatch(role: str, field: str, value: object) -> None:
                print(
                    f"[launch] NOTE: algo YAML overrides roles.{role}.{field}"
                    f"={value!r} -- this is a placement field owned by the "
                    f"cluster preset.  Prefer editing the preset directly "
                    f"if this isn't a one-off experiment.",
                    file=sys.stderr,
                )

            composed = compose_launcher_yaml(
                data, preset_data, on_escape_hatch=_warn_escape_hatch
            )
        except PresetError as e:
            print(f"[launch] preset composition failed: {e}", file=sys.stderr)
            raise SystemExit(2) from e

        fd, tmp_path = tempfile.mkstemp(
            prefix="forge_composed_", suffix=".yaml", dir="/tmp"
        )
        with os.fdopen(fd, "w") as fh:
            fh.write(
                "# Auto-generated by `forge launch` from:\n"
                f"#   algo YAML : {positional}\n"
                f"#   preset    : {preset_path}\n"
                "# This file is overwritten on every launch; do not edit.\n"
            )
            yaml.safe_dump(composed, fh, sort_keys=False)
        print(
            f"[launch] composed launcher YAML from preset "
            f"{preset_name!r} -> {tmp_path}",
            flush=True,
        )

        # Strip the launcher-layer keys from the algo YAML before
        # grpo.py consumes it.  ``load_expr_config`` merges the algo
        # YAML against the strict ``GRPOConfig`` dataclass which
        # rejects unknown top-level keys -- including our new
        # ``launcher_preset`` / ``roles`` / ``cluster`` additions.
        # The cleanest split is: algo YAML owns the two-layer
        # declarations for ``forge launch``, but what grpo.py sees
        # should only be the GRPOConfig-shaped subset.
        sanitized = {k: v for k, v in data.items() if k not in _LAUNCHER_LAYER_KEYS}
        algo_fd, algo_tmp = tempfile.mkstemp(
            prefix="forge_algo_", suffix=".yaml", dir="/tmp"
        )
        with os.fdopen(algo_fd, "w") as fh:
            fh.write(
                "# Auto-generated by `forge launch` from:\n"
                f"#   source    : {positional}\n"
                f"#   stripped  : {sorted(_LAUNCHER_LAYER_KEYS & set(data))}\n"
                "# This file is overwritten on every launch; do not edit.\n"
            )
            yaml.safe_dump(sanitized, fh, sort_keys=False)
        return Path(tmp_path), Path(algo_tmp)

    print(
        f"[launch] {positional} has neither a top-level `launcher_preset:` "
        f"(algo YAML) nor `launcher:` (raw launcher YAML).  See "
        f"forge/docs/role_abstraction_design.md for the two-layer authoring "
        f"guide.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _read_mode_from_algo_yaml(algo_path: Path) -> str | None:
    """Return the algo YAML's top-level ``mode:`` value, or ``None``.

    Used as the second-priority source for mode resolution
    (CLI flag > algo YAML > default ``grpo``).  Any read/parse
    error degrades to ``None`` so a malformed YAML can't silently
    flip mode -- caller falls back to the default.
    """
    try:
        import yaml

        data = yaml.safe_load(algo_path.read_text()) or {}
    except Exception:
        return None
    val = data.get("mode") if isinstance(data, dict) else None
    if val is None:
        return None
    val_str = str(val)
    if val_str not in _VALID_MODES:
        # Don't propagate garbage -- caller will default to grpo
        # and the algo author gets one warning instead of a cryptic
        # downstream failure.
        print(
            f"[launch] algo YAML has unrecognised mode={val_str!r}; "
            f"valid choices: {sorted(_VALID_MODES)}.  Falling back "
            f"to default.",
            file=sys.stderr,
        )
        return None
    return val_str


def _read_profile_from_yaml(config_path: Path) -> str | None:
    """Return ``launcher.profile`` from the launcher YAML, or ``None``.

    The cluster preset is the authoritative source for which worker
    env profile to install -- e.g. ``2node_pure_training.yaml`` sets
    ``launcher.profile: pure-training``.  Caller falls back to
    :data:`_MODE_DEFAULT_PROFILE` when this returns ``None``.
    """
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return None
    launcher_block = data.get("launcher") or {}
    if not isinstance(launcher_block, dict):
        return None
    val = launcher_block.get("profile")
    return str(val) if val else None


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


def _read_pool_from_yaml(
    config_path: Path,
) -> tuple[list[str], str | None, int | None]:
    """Parse ``launcher.pool`` out of the launcher YAML.

    Returns ``(hosts, driver, worker_port)`` where:

    * ``hosts`` is the ordered list of worker IPs / hostnames (empty
      if the YAML has no pool -- caller falls back to the hostfile).
    * ``driver`` is the IP of the pool entry tagged ``role: driver``,
      or ``None`` when no entry is tagged (caller applies the legacy
      "second line of hostfile" heuristic).
    * ``worker_port`` is the port of the first pool entry, or
      ``None`` when the pool is empty.  The scheduler assumes every
      entry uses the same port (mirrors what
      ``LauncherConfig.__post_init__`` does) -- if someone mixes
      ports we honor the first and emit a warning from the
      launcher side.

    Any parse / schema error degrades to ``([], None, None)`` so
    malformed YAML can't silently break legacy hostfile flows.
    """
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return [], None, None
    launcher_block = data.get("launcher") or {}
    if not isinstance(launcher_block, dict):
        return [], None, None
    raw_pool = launcher_block.get("pool") or []
    if not isinstance(raw_pool, list) or not raw_pool:
        return [], None, None

    hosts: list[str] = []
    driver: str | None = None
    worker_port: int | None = None
    for entry in raw_pool:
        if not isinstance(entry, dict):
            continue
        host = entry.get("host")
        if not host:
            continue
        hosts.append(str(host))
        if worker_port is None:
            port = entry.get("port")
            if port is not None:
                try:
                    worker_port = int(port)
                except (TypeError, ValueError):
                    worker_port = None
        if entry.get("role") == "driver" and driver is None:
            driver = str(host)
    return hosts, driver, worker_port


def _stage_tmp_yamls_on_driver(
    *,
    config: Path,
    algo_config: Path,
    driver: str,
    ssh_port: int,
) -> tuple[Path, Path]:
    """scp auto-generated ``/tmp/forge_*.yaml`` files to the driver host.

    The two-layer composer in :func:`_resolve_configs` writes
    ``/tmp/forge_composed_*.yaml`` and ``/tmp/forge_algo_*.yaml`` on
    the local launcher host.  When the driver is a *different*
    machine, ``python -m forge.apps.grpo --config /tmp/...`` fails
    with ``Config file does not exist`` because /tmp is not shared.

    We keep the absolute path identical on both sides (``/tmp/...``)
    so ``grpo.py``'s CLI line doesn't need rewriting -- the remote
    copy lands at the same path as the local one.  Non-``/tmp``
    paths (committed YAMLs) are left untouched.
    """
    to_stage = [p for p in (config, algo_config) if str(p).startswith("/tmp/")]
    if not to_stage:
        return config, algo_config

    for p in to_stage:
        scp_cmd = [
            "scp",
            "-P",
            str(ssh_port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            "-q",
            str(p),
            f"root@{driver}:{p}",
        ]
        rc = subprocess.call(scp_cmd)
        if rc != 0:
            print(
                f"[launch] failed to scp {p} to {driver}: rc={rc}",
                file=sys.stderr,
            )
            raise SystemExit(3)
        print(f"[launch] staged {p} -> {driver}:{p}", flush=True)
    return config, algo_config


# ---------------------------------------------------------------------------
# Driver dispatch
# ---------------------------------------------------------------------------
#
# All three driver entrypoints (grpo / titan-pretrain / inference-bench)
# share the same machinery: SSH into the driver host with a bash
# preamble that sources CANN, activates the conda env, exports a few
# dozen runtime knobs, ``cd`` into the source root (FUSE mount or
# local checkout), and pipes ``python -m forge.apps.X ...`` through
# ``tee`` for an on-disk log.  Before the extraction below this was
# 240 lines of nearly-identical bash composition spread across three
# functions, with subtle drift (one missed ``set -o pipefail`` for
# almost a full session).
#
# The split is now:
#
# * :func:`_driver_base_env_lines` -- env block every mode needs
#   (CANN, conda, ASCEND/HCCL knobs, HF offline trio).  Stable
#   across modes; changes here apply uniformly.
# * :func:`_run_driver_via_ssh` -- the bash composition + SSH wrap +
#   timed status print.  Single owner of the ugly shell parts; mode
#   builders just hand it ``argv``, ``env_extras``, and a log path.
# * :func:`_run_driver_training` / :func:`_run_driver_titan_pretrain`
#   / :func:`_run_driver_inference_bench` -- thin builders that
#   compose the per-mode argv + env extras, then call the helper.
#
# Future direction: replace SSH+tee with a Monarch ``BashActor``
# spawned on the driver host's worker proc, using
# ``start()`` / ``poll_output()`` for live log forwarding.  Deferred
# until the streaming-stdout story on top of Monarch is mature
# enough to drop tee without losing live progress visibility.
# ---------------------------------------------------------------------------


# Forge / torchstore env vars forwarded from the operator's shell to
# the driver process so ``forge.engines.weight_sync._config_resolver``
# can pick them up as overrides.  Vars not set in the operator's shell
# are left unset, letting the YAML value win.  Currently only used
# by the GRPO mode (titan / inference don't run weight_sync), but
# kept module-level so a single audit point covers the forwarding
# contract.
_GRPO_FORWARDED_ENV: tuple[str, ...] = (
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
)


def _driver_base_env_lines(cann: str) -> list[str]:
    """Return the env preamble lines every driver mode needs.

    These are deliberately mode-agnostic: CANN sourcing, the
    ``monarch_ascend`` conda activation, the universally-safe
    ASCEND/HCCL knobs, and the HuggingFace offline trio.  Mode-
    specific env (HiXL/torchstore for GRPO, PYTHONPATH for
    titan/inference, etc.) goes in the ``env_extras`` parameter
    of :func:`_run_driver_via_ssh`.

    The HF offline trio (``HF_DATASETS_OFFLINE`` / ``TRANSFORMERS_OFFLINE``
    / ``HF_HUB_OFFLINE``) is here even though TT loads from
    ``c4_test`` fixtures and inference-bench loads from local
    snapshots: any helper module either driver imports may
    indirectly pull in ``transformers``, which will block on
    ``huggingface.co`` if the firewall is in front.  The
    ``${VAR:-1}`` form lets operators override with
    ``HF_DATASETS_OFFLINE=0 forge launch ...`` when warming the
    cache on a one-off basis.
    """
    return [
        f"source {cann}/set_env.sh 2>/dev/null",
        f"ASCEND_ROOT=$(dirname {cann})",
        '[[ -f "$ASCEND_ROOT/nnal/atb/set_env.sh" ]] && '
        'source "$ASCEND_ROOT/nnal/atb/set_env.sh"',
        'eval "$(/root/miniconda3/bin/conda shell.bash hook)"',
        "conda activate monarch_ascend",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        'export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"',
        'export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-1}"',
        'export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"',
        "export HCCL_CONNECT_TIMEOUT=120",
        'export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"',
        'export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"',
        'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"',
    ]


def _run_driver_via_ssh(
    *,
    mode_label: str,
    driver: str,
    ssh_port: int,
    cann: str,
    code_root: str,
    argv: list[str],
    env_extras: list[str],
    log_path: str,
    show_error_hint: bool = False,
) -> int:
    """Single owner of the SSH + bash + tee driver dispatch path.

    ``mode_label`` is purely cosmetic, used in the start/finish
    print lines (``[launch] training on ...`` vs ``titan-pretrain``
    vs ``inference-bench``).

    ``code_root`` is the worker-visible source path -- typically
    ``/root/AReaL_remote`` (FUSE mount, the default when
    ``mount_code=True``) or ``/root/AReaL`` (local checkout).  The
    ``mkdir -p`` before ``cd`` is defensive: when ``code_root`` is
    a FUSE mountpoint and the mount activation hasn't reached this
    host yet, ``cd`` would fail with ENOENT.

    ``env_extras`` are appended after the base env block so a mode
    can override base defaults (e.g.\\ titan re-exports a fuller
    PYTHONPATH for explicit ``/root/monarch/python`` access).

    ``log_path`` is on the driver host's filesystem (we still tee
    to ``/tmp/forge_*.log`` for post-hoc inspection); separate per
    mode so concurrent smokes don't clobber each other.

    ``set -o pipefail`` is universal: without it, ``python ... | tee``
    returns tee's exit status (always 0) and silently masks driver
    failures.  We lost a debugging session to a missing pipefail in
    the titan path before this was extracted; centralising it here
    is the easiest way to ensure it's never forgotten on a future
    mode addition.

    ``show_error_hint`` controls whether :func:`_error_hint` is
    printed on non-zero rc.  Currently only the GRPO mode opts in
    -- titan/inference failures are typically argparse / config
    errors that don't benefit from the weight_sync-flavoured hint
    text.
    """
    base_env = _driver_base_env_lines(cann)
    remote_cmd_parts = [
        "set -o pipefail",
        *base_env,
        *env_extras,
        f"mkdir -p {shlex.quote(code_root)}",
        f"cd {shlex.quote(code_root)}",
        _shellify(argv) + f" 2>&1 | tee {shlex.quote(log_path)}",
    ]
    remote_cmd = "; ".join(remote_cmd_parts)

    ssh_cmd = [
        "ssh",
        "-p",
        str(ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        f"root@{driver}",
        remote_cmd,
    ]

    start = time.time()
    print(f"[launch] {mode_label} on {driver} (started at {time.strftime('%H:%M:%S')})")
    rc = subprocess.call(ssh_cmd)
    elapsed = time.time() - start

    print("=" * 64)
    if rc == 0:
        print(f" forge launch: {mode_label} OK ({elapsed:.0f}s)")
    else:
        print(f" forge launch: {mode_label} FAILED rc={rc} ({elapsed:.0f}s)")
        if show_error_hint:
            print(_error_hint(rc, log_path=log_path))
    print("=" * 64)
    return rc


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
    code_root: str | None = None,
) -> int:
    """SSH into the driver host and run ``python -m forge.apps.grpo``.

    Composes the GRPO-specific argv + env extras (HiXL/torchstore
    transport vars, ModelScope/HF mirror plumbing for in-process
    weight pulls, plus operator-forwarded ``FORGE_*`` overrides),
    then hands off to :func:`_run_driver_via_ssh` for the bash +
    SSH wrapping.

    ``code_root`` overrides the worker-visible ``cd`` target.
    Default :data:`FORGE_ROOT` keeps the legacy behaviour (driver
    reads its own local ``/root/AReaL`` checkout); pass
    :attr:`_SSHJobFleet.code_root` (typically ``/root/AReaL_remote``)
    when ``mount_code=True`` so the driver sees the same FUSE-mounted
    source view as the workers it orchestrates -- no more local-disk
    drift across hosts.
    """
    if code_root is None:
        code_root = str(FORGE_ROOT)
    # When the YAML paths live under /tmp (i.e. were auto-generated by
    # ``_resolve_configs`` for the two-layer composer), they only exist
    # on the local launcher host.  The driver SSH below reads them from
    # its own filesystem, so we need to push a copy over first.  For
    # checked-in paths (legacy one-layer invocation) this is a no-op.
    config, algo_config = _stage_tmp_yamls_on_driver(
        config=config,
        algo_config=algo_config,
        driver=driver,
        ssh_port=args.ssh_port,
    )

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

    # GRPO env extras: HiXL/torchstore transport (weight sync runs in
    # the driver proc itself for the colocated mesh), VLLM/HF mirror
    # for vLLM-side model fetching, and operator-forwarded FORGE_*
    # vars (so ``FORGE_WEIGHT_SYNC=ascend forge launch ...`` reaches
    # _config_resolver and trips the deprecation warning).
    env_extras = [
        "export VLLM_USE_MODELSCOPE=true",
        "export HF_ENDPOINT=https://hf-mirror.com",
        "export MONARCH_HIXL_TRANSPORT=roce",
        "export HCCL_INTRA_ROCE_ENABLE=1",
        "export HCCL_NPU_SOCKET_PORT_RANGE=60000-60255",
        "export TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0",
        "export TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:0",
    ]
    for var in _GRPO_FORWARDED_ENV:
        v = os.environ.get(var)
        if v is not None:
            env_extras.append(f"export {var}={shlex.quote(v)}")

    return _run_driver_via_ssh(
        mode_label="training",
        driver=driver,
        ssh_port=args.ssh_port,
        cann=args.cann,
        code_root=code_root,
        argv=grpo_argv,
        env_extras=env_extras,
        log_path="/tmp/forge_multinode.log",
        show_error_hint=True,
    )


def _run_driver_titan_pretrain(
    *,
    args: argparse.Namespace,
    driver: str,
    algo_config: Path,
    workers_arg: str,
    code_root: str | None = None,
) -> int:
    """SSH into the driver host and run ``python -m forge.apps.titan_pretrain``.

    The titan-pretrain mode runs with the slim ``pure-training``
    profile env: no HiXL, no torchstore, no ModelScope/HF mirror
    plumbing.  TT itself loads bundled ``c4_test`` fixtures, so the
    only mode-specific extra is the explicit PYTHONPATH (so
    ``import monarch`` / ``import torchstore`` resolve through the
    out-of-tree checkouts at ``/root/monarch/python`` and
    ``/root/torchstore``, which aren't installed into the conda env).
    """
    if code_root is None:
        code_root = str(FORGE_ROOT)
    _, algo_config = _stage_tmp_yamls_on_driver(
        config=algo_config,
        algo_config=algo_config,
        driver=driver,
        ssh_port=args.ssh_port,
    )

    pretrain_argv = [
        "python",
        "-m",
        "forge.apps.titan_pretrain",
        "--algo-config",
        str(algo_config),
        "--bare-metal-workers",
        workers_arg,
    ]
    if args.steps is not None:
        pretrain_argv += ["--steps", str(args.steps)]

    env_extras = [
        f'export PYTHONPATH={shlex.quote(code_root)}":/root/torchstore:/root/monarch/python:${{PYTHONPATH:-}}"',
    ]

    return _run_driver_via_ssh(
        mode_label="titan-pretrain",
        driver=driver,
        ssh_port=args.ssh_port,
        cann=args.cann,
        code_root=code_root,
        argv=pretrain_argv,
        env_extras=env_extras,
        log_path="/tmp/forge_titan_pretrain.log",
    )


def _run_driver_inference_bench(
    *,
    args: argparse.Namespace,
    driver: str,
    algo_config: Path,
    workers_arg: str,
    code_root: str | None = None,
) -> int:
    """SSH into the driver host and run ``python -m forge.apps.inference_bench``.

    Same slim pure-training env as titan-pretrain, plus a
    ``--duration`` knob unique to the bench harness.  The driver
    itself doesn't load the model (orchestration only); the vLLM
    replicas it spawns on each worker do, and they have their own
    HF cache via the worker profile env.
    """
    if code_root is None:
        code_root = str(FORGE_ROOT)
    _, algo_config = _stage_tmp_yamls_on_driver(
        config=algo_config,
        algo_config=algo_config,
        driver=driver,
        ssh_port=args.ssh_port,
    )

    bench_argv = [
        "python",
        "-m",
        "forge.apps.inference_bench",
        "--algo-config",
        str(algo_config),
        "--bare-metal-workers",
        workers_arg,
    ]
    # ``args.steps`` is the GRPO/titan-pretrain training-step knob;
    # for inference we surface ``--duration`` instead.  Forwarded
    # only when the user actually set it so the YAML value wins
    # otherwise.
    duration = getattr(args, "duration", None)
    if duration is not None:
        bench_argv += ["--duration", str(duration)]

    env_extras = [
        f'export PYTHONPATH={shlex.quote(code_root)}":/root/torchstore:/root/monarch/python:${{PYTHONPATH:-}}"',
    ]

    return _run_driver_via_ssh(
        mode_label="inference-bench",
        driver=driver,
        ssh_port=args.ssh_port,
        cann=args.cann,
        code_root=code_root,
        argv=bench_argv,
        env_extras=env_extras,
        log_path="/tmp/forge_inference_bench.log",
    )


def _shellify(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def _error_hint(rc: int, *, log_path: str = "/tmp/forge_multinode.log") -> str:
    """Best-effort error code translation.

    Generic pointer to the post-mortem log + the documented
    failure-mode ledger.  Wire in keyword grepping of ``log_path``
    if the static text turns out to be insufficient.
    """
    return (
        f"  Hint: rc={rc}. Inspect {log_path} on the driver host.\n"
        f"  Common failure modes are documented in\n"
        f"    forge/docs/weight_sync.md §7 (known-pain ledger)\n"
        f"  with each CANN/HCCL/HiXL error code mapped to a root cause."
    )
