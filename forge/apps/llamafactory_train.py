"""``forge.apps.llamafactory_train`` -- B-full LlamaFactory SFT entry.

Runs on the driver host after ``forge launch`` brings up Monarch
worker daemons.  Reads a two-layer YAML config (algo YAML composed
with cluster preset by ``forge/cli/presets.py``), spawns one
:class:`LlamaFactoryTrainerActor` per (host, NPU), and drives
``llamafactory.train.tuner.run_exp`` end-to-end across the cluster.

Relationship to the legacy standalone driver
--------------------------------------------
:mod:`forge.scripts.llamafactory_train_singlehost` (B-mini) is the
original ``this_host()``-based single-host driver.  This module is
its B-full descendant: same orchestration body, but plugged into
the ``forge launch`` lifecycle so:

* Worker spawn / cleanup go through ``_BashFleet`` /
  ``_SSHJobFleet`` -- the same code paths GRPO and titan-pretrain
  use.  ``len(pool) == 1`` collapses to a 1-host run with no
  branching here.
* Pre-flight (SSH reachability, YAML parse, tooling check) is
  shared with the rest of forge launch.
* Per-host code drift is impossible -- the driver host pulls this
  module from a single source-of-truth (the launcher host's
  ``/root/AReaL`` via SSH cwd), and remote workers only need
  ``LlamaFactoryTrainerActor`` (which lives under
  :mod:`forge.actors.llamafactory_trainer` and is part of the
  daemon's PYTHONPATH).
* Same SIGINT / SIGTERM cleanup contract as GRPO / titan-pretrain.

Why an LF-specific app instead of folding into titan-pretrain
-------------------------------------------------------------
TT and LF actors share the SPMDActor lifecycle (setup_env / get_host_port
/ run) but have *different* run() signatures: TT takes
``(toml_path, cwd, overrides)`` where overrides is a list of CLI
flags; LF takes ``(lf_yaml_path, accelerate_yaml_path, cwd, overrides)``
where overrides is a dict merged into the parsed YAML.  The
algo-YAML schema mirrors that asymmetry (``titan:`` block vs.
``llamafactory:`` block), so each app gets its own thin orchestrator
and the actor-level abstraction stays clean.

Usage
-----
This module is normally invoked indirectly via::

    python -m forge launch examples/sft/qwen3vl_4b_lf_npu.yaml --steps 3

Direct invocation (after ``forge launch`` already started workers,
e.g. for re-runs against the same fleet)::

    python -m forge.apps.llamafactory_train \\
        --algo-config examples/sft/qwen3vl_4b_lf_npu.yaml \\
        --bare-metal-workers tcp://192.168.0.26:22222 \\
        --steps 3

Exit codes
----------
* ``0``   every rank returned ``ok=True``
* ``2``   bad CLI args / YAML
* ``20``  Monarch attach / spawn failed
* ``30``  at least one rank reported ``ok=False`` or raised
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge.apps.llamafactory_train",
        description=(
            "Multi-node LlamaFactory SFT/PT driver.  Normally invoked "
            "via `forge launch`; can be run directly when iterating "
            "against an already-started worker fleet."
        ),
    )
    p.add_argument(
        "--algo-config",
        type=Path,
        required=True,
        help=(
            "Algorithm YAML with a top-level ``llamafactory:`` block "
            "(lf_config / accelerate_config / cwd / procs_per_host / "
            "master_port / overrides / use_modelscope) and an "
            "optional ``mode: llamafactory-train`` tag."
        ),
    )
    p.add_argument(
        "--bare-metal-workers",
        type=str,
        required=True,
        help=(
            "Comma-separated list of Monarch worker URIs, e.g. "
            "``tcp://192.168.0.26:22222,tcp://192.168.0.23:22222``.  "
            "Same flag GRPO / titan-pretrain use, kept identical so "
            "launch.py's wiring is unchanged."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Convenience override for ``llamafactory.overrides.max_steps`` "
            "-- merged into the LF YAML at runtime.  CLI > YAML so a "
            "``forge launch ... --steps N`` always wins."
        ),
    )
    return p


def _load_lf_section(algo_yaml: Path) -> dict:
    """Read and validate the ``llamafactory:`` block of the algo YAML.

    Returns the dict with required keys (``lf_config``,
    ``accelerate_config``, ``cwd``) populated and optional keys
    defaulted.  Raises ``SystemExit(2)`` with a clear message on any
    structural problem -- the algo author shouldn't have to dig
    through stack traces for typos.
    """
    import yaml

    if not algo_yaml.is_file():
        print(
            f"[llamafactory_train] algo YAML not found: {algo_yaml}",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        data = yaml.safe_load(algo_yaml.read_text()) or {}
    except yaml.YAMLError as e:
        print(
            f"[llamafactory_train] algo YAML unparseable ({algo_yaml}): {e}",
            file=sys.stderr,
        )
        sys.exit(2)

    lf = data.get("llamafactory")
    if not isinstance(lf, dict):
        print(
            "[llamafactory_train] algo YAML missing required "
            "``llamafactory:`` block (see "
            "examples/sft/qwen3vl_4b_lf_npu.yaml)",
            file=sys.stderr,
        )
        sys.exit(2)

    for key in ("lf_config", "accelerate_config", "cwd"):
        if not lf.get(key):
            print(
                f"[llamafactory_train] algo YAML ``llamafactory.{key}`` is required",
                file=sys.stderr,
            )
            sys.exit(2)

    lf.setdefault("procs_per_host", 4)
    lf.setdefault("master_port", 0)
    lf.setdefault("overrides", {})
    lf.setdefault("use_modelscope", True)

    # Resolve config paths to absolute -- the actor on a remote host
    # won't share cwd with the launcher.
    lf["lf_config"] = Path(lf["lf_config"]).resolve()
    lf["accelerate_config"] = Path(lf["accelerate_config"]).resolve()
    lf["cwd"] = Path(lf["cwd"]).resolve()

    if not lf["lf_config"].is_file():
        print(
            f"[llamafactory_train] llamafactory.lf_config does not exist: "
            f"{lf['lf_config']}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not lf["accelerate_config"].is_file():
        print(
            f"[llamafactory_train] llamafactory.accelerate_config does not "
            f"exist: {lf['accelerate_config']}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not lf["cwd"].is_dir():
        print(
            f"[llamafactory_train] llamafactory.cwd does not exist: {lf['cwd']}",
            file=sys.stderr,
        )
        sys.exit(2)

    if not isinstance(lf["overrides"], dict):
        print(
            "[llamafactory_train] llamafactory.overrides must be a mapping "
            "(LF args dict merged into lf_config YAML), got "
            f"{type(lf['overrides']).__name__}",
            file=sys.stderr,
        )
        sys.exit(2)

    return lf


async def _drive_training(
    *,
    workers: list[str],
    lf_config: Path,
    accelerate_config: Path,
    lf_cwd: Path,
    procs_per_host: int,
    master_port: int,
    overrides: dict[str, Any],
    use_modelscope: bool,
) -> int:
    """One full multi-host LlamaFactory run.  0 on success, 30 on any rank failure.

    Body is a near-twin of
    :func:`forge.apps.titan_pretrain._drive_training` -- same
    attach_to_workers / spawn_procs / setup_env / actor.run.call
    sequence, with the LF actor's run() signature wired in.  Diff
    is ~10 lines.
    """
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers

    from forge.actors.llamafactory_trainer import LlamaFactoryTrainerActor

    # Same transport choice as the rest of forge's bare-metal stack
    # (see forge.apps.titan_pretrain for the rationale).
    configure(default_transport=ChannelTransport.TcpWithHostname)

    print(f"[llamafactory_train] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized
    n_hosts = hosts.size()
    world_size = n_hosts * procs_per_host
    print(
        f"[llamafactory_train] attached to {n_hosts} host(s); "
        f"world_size = {world_size}",
        flush=True,
    )

    proc_mesh = hosts.spawn_procs(per_host={"gpus": procs_per_host})
    actor = proc_mesh.spawn("lf_trainer", LlamaFactoryTrainerActor)

    # Pull MASTER_ADDR/PORT from rank 0; honour pinned port if requested.
    first_values = dict.fromkeys(proc_mesh._labels, 0)
    rank0 = actor.slice(**first_values)
    if master_port > 0:
        master_addr = (await rank0.get_host_port.call_one(None))[0]
        chosen_port = master_port
    else:
        master_addr, chosen_port = await rank0.get_host_port.call_one(None)
    print(
        f"[llamafactory_train] rendezvous = {master_addr}:{chosen_port}",
        flush=True,
    )

    # Publish env (RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR /
    # MASTER_PORT) to every rank -- LF + accelerate + transformers
    # then init_pg via env:// just like a torchrun launch.
    await actor.setup_env.call(master_addr, chosen_port)

    print("=" * 72, flush=True)
    print(
        f"[llamafactory_train] launching LlamaFactory: lf_yaml={lf_config} "
        f"accelerate_yaml={accelerate_config} cwd={lf_cwd}",
        flush=True,
    )
    if overrides:
        print(f"[llamafactory_train] overrides = {overrides}", flush=True)
    print("=" * 72, flush=True)

    t0 = time.time()
    results = await actor.run.call(
        lf_yaml_path=str(lf_config),
        accelerate_yaml_path=str(accelerate_config),
        cwd=str(lf_cwd),
        overrides=overrides or None,
        # Critical for SSHJob workers: env doesn't inherit from the
        # driver, so the actor must export USE_MODELSCOPE_HUB itself
        # before importing LF.  See the actor docstring for the full
        # rationale.
        use_modelscope=use_modelscope,
    )
    elapsed = time.time() - t0

    failures: list[tuple[int, str]] = []
    try:
        items = list(results)
    except TypeError:
        items = [(0, results)]

    for entry in items:
        if isinstance(entry, tuple):
            _idx, payload = entry
        else:
            payload = entry
        if isinstance(payload, dict):
            ok = bool(payload.get("ok"))
            rank = payload.get("rank", "?")
            host = payload.get("host", "?")
            secs = payload.get("elapsed_s", -1)
            print(
                f"[llamafactory_train] rank={rank} host={host} "
                f"elapsed={secs:.1f}s ok={ok}",
                flush=True,
            )
            if not ok:
                failures.append((rank, str(payload)))
        else:
            print(
                f"[llamafactory_train] rank=? payload={payload!r}",
                flush=True,
            )

    print("=" * 72, flush=True)
    print(
        f"[llamafactory_train] all ranks returned in {elapsed:.1f}s",
        flush=True,
    )
    if failures:
        for rank, msg in failures:
            print(
                f"[llamafactory_train] FAIL rank={rank}: {msg}",
                file=sys.stderr,
            )
        return 30

    return 0


def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    lf = _load_lf_section(args.algo_config)

    workers = [w.strip() for w in args.bare_metal_workers.split(",") if w.strip()]
    if not workers:
        print(
            "[llamafactory_train] --bare-metal-workers parsed empty",
            file=sys.stderr,
        )
        return 2

    # Compose final LF override dict.  ``--steps N`` from the CLI
    # wins over llamafactory.overrides because it's applied last.
    overrides: dict[str, Any] = dict(lf["overrides"])
    if args.steps is not None:
        overrides["max_steps"] = int(args.steps)

    # Set USE_MODELSCOPE_HUB on the driver too -- harmless, and lets
    # any driver-side LF code (none today, but possible future
    # pre-flight) pick it up.  The CRITICAL setter is the actor itself
    # (passed via use_modelscope below) because SSHJob workers do not
    # inherit the driver's env.
    use_modelscope = bool(lf["use_modelscope"])
    if use_modelscope:
        os.environ.setdefault("USE_MODELSCOPE_HUB", "1")

    try:
        return asyncio.run(
            _drive_training(
                workers=workers,
                lf_config=lf["lf_config"],
                accelerate_config=lf["accelerate_config"],
                lf_cwd=lf["cwd"],
                procs_per_host=int(lf["procs_per_host"]),
                master_port=int(lf["master_port"]),
                overrides=overrides,
                use_modelscope=use_modelscope,
            )
        )
    except KeyboardInterrupt:
        print(
            "[llamafactory_train] interrupted -- tearing down",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(
            f"[llamafactory_train] unexpected error: {exc!r}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 20


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
