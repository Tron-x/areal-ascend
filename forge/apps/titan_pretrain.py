"""``forge.apps.titan_pretrain`` -- B-full TorchTitan pretrain entry.

Runs on the driver host after ``forge launch`` brings up Monarch
worker daemons.  Reads a two-layer YAML config (algo YAML composed
with cluster preset by ``forge/cli/presets.py``), spawns one
:class:`TitanTrainerActor` per (host, NPU), and drives
``Trainer.train()`` end-to-end across the cluster.

Relationship to the legacy standalone driver
--------------------------------------------
:mod:`forge.scripts.titan_train_multinode` (B-mini) is the original
hand-rolled multi-node TT driver.  This module is its B-full
descendant: same orchestration body, but plugged into the
``forge launch`` lifecycle so:

* Worker spawn / cleanup go through ``_BashFleet`` /
  ``_SSHJobFleet`` -- the same code paths GRPO uses.
* Pre-flight (SSH reachability, YAML parse, tooling check) is
  shared.
* Per-host code drift is impossible -- the driver host pulls this
  module from a single source-of-truth (the launcher host's
  /root/AReaL via SSH cwd), and remote workers only need
  ``TitanTrainerActor`` (which already lives under ``forge.actors``
  and is a hard dependency of the daemon's PYTHONPATH).
* Same SIGINT / SIGTERM cleanup contract as GRPO.

Usage
-----
This module is normally invoked indirectly via::

    python -m forge launch examples/pretrain/llama3_titan_debug.yaml \\
        --steps 5

Direct invocation (after ``forge launch`` already started workers,
e.g. for re-runs against the same fleet)::

    python -m forge.apps.titan_pretrain \\
        --algo-config examples/pretrain/llama3_titan_debug.yaml \\
        --bare-metal-workers tcp://192.168.0.26:22222,tcp://192.168.0.23:22222 \\
        --steps 5

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

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge.apps.titan_pretrain",
        description=(
            "Multi-node TorchTitan pretraining driver.  Normally "
            "invoked via `forge launch`; can be run directly when "
            "iterating against an already-started worker fleet."
        ),
    )
    p.add_argument(
        "--algo-config",
        type=Path,
        required=True,
        help=(
            "Algorithm YAML with a top-level ``titan:`` block "
            "(config/cwd/procs_per_host/master_port/overrides) and "
            "an optional ``mode: titan-pretrain`` tag."
        ),
    )
    p.add_argument(
        "--bare-metal-workers",
        type=str,
        required=True,
        help=(
            "Comma-separated list of Monarch worker URIs, e.g. "
            "``tcp://192.168.0.26:22222,tcp://192.168.0.23:22222``.  "
            "Same flag GRPO uses, kept identical so launch.py's "
            "wiring is unchanged."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Convenience override for ``titan.overrides`` -- "
            "appended as ``--training.steps N`` after the YAML "
            "overrides so it always wins.  Equivalent to passing "
            "``-- --training.steps N`` to forge launch."
        ),
    )
    return p


def _load_titan_section(algo_yaml: Path) -> dict:
    """Read and validate the ``titan:`` block of the algo YAML.

    Returns the dict with required keys (``config``, ``cwd``)
    populated and optional keys defaulted.  Raises ``SystemExit(2)``
    with a clear message on any structural problem -- the algo
    author shouldn't have to dig through stack traces for typos.
    """
    import yaml

    if not algo_yaml.is_file():
        print(f"[titan_pretrain] algo YAML not found: {algo_yaml}", file=sys.stderr)
        sys.exit(2)

    try:
        data = yaml.safe_load(algo_yaml.read_text()) or {}
    except yaml.YAMLError as e:
        print(
            f"[titan_pretrain] algo YAML unparseable ({algo_yaml}): {e}",
            file=sys.stderr,
        )
        sys.exit(2)

    titan = data.get("titan")
    if not isinstance(titan, dict):
        print(
            "[titan_pretrain] algo YAML missing required ``titan:`` block "
            "(see examples/pretrain/llama3_titan_debug.yaml)",
            file=sys.stderr,
        )
        sys.exit(2)

    for key in ("config", "cwd"):
        if not titan.get(key):
            print(
                f"[titan_pretrain] algo YAML ``titan.{key}`` is required",
                file=sys.stderr,
            )
            sys.exit(2)

    titan.setdefault("procs_per_host", 4)
    titan.setdefault("master_port", 0)
    titan.setdefault("overrides", [])

    # Resolve config/cwd to absolute Path objects -- the actor on a
    # remote host won't have the same cwd as the launcher.
    titan["config"] = Path(titan["config"]).resolve()
    titan["cwd"] = Path(titan["cwd"]).resolve()

    if not titan["config"].is_file():
        print(
            f"[titan_pretrain] titan.config does not exist: {titan['config']}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not titan["cwd"].is_dir():
        print(
            f"[titan_pretrain] titan.cwd does not exist: {titan['cwd']}",
            file=sys.stderr,
        )
        sys.exit(2)

    return titan


async def _drive_training(
    *,
    workers: list[str],
    tt_config: Path,
    tt_cwd: Path,
    procs_per_host: int,
    master_port: int,
    overrides: list[str],
) -> int:
    """One full multi-host TorchTitan run.  0 on success, 30 on any rank failure.

    Body lifted verbatim from
    :func:`forge.scripts.titan_train_multinode._run` so the two
    entry points stay behaviourally equivalent.  When B-mini is
    eventually deleted, this becomes the canonical implementation.
    """
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers

    from forge.actors.titan_trainer import TitanTrainerActor

    # Same transport choice as the rest of forge's bare-metal stack.
    # Without this the default in-proc transport refuses cross-host
    # attach.  See forge/scripts/test_weight_sync_2node.py.
    configure(default_transport=ChannelTransport.TcpWithHostname)

    print(f"[titan_pretrain] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized
    n_hosts = hosts.size()
    world_size = n_hosts * procs_per_host
    print(
        f"[titan_pretrain] attached to {n_hosts} host(s); world_size = {world_size}",
        flush=True,
    )

    # Single proc mesh spanning every host.  Dim name ``gpus`` is
    # what SPMDActor's __init__ keys off when computing LOCAL_RANK
    # (`point.extent.labels[-1]`).  Renaming silently breaks
    # local_rank wiring for any subclass.
    proc_mesh = hosts.spawn_procs(per_host={"gpus": procs_per_host})
    actor = proc_mesh.spawn("titan_trainer", TitanTrainerActor)

    # Pull MASTER_ADDR/PORT from rank 0.  ``master_port=0`` lets
    # SPMDActor pick a free port (recommended); a positive value
    # pins it (useful in firewall-restricted envs).
    first_values = dict.fromkeys(proc_mesh._labels, 0)
    rank0 = actor.slice(**first_values)
    if master_port > 0:
        master_addr = (await rank0.get_host_port.call_one(None))[0]
        chosen_port = master_port
    else:
        master_addr, chosen_port = await rank0.get_host_port.call_one(None)
    print(
        f"[titan_pretrain] rendezvous = {master_addr}:{chosen_port}",
        flush=True,
    )

    # Publish the rendezvous + RANK/LOCAL_RANK/WORLD_SIZE to every
    # rank.  After this returns every proc has an env that looks
    # exactly like a torchrun launch -- TT's `env://` init_pg path
    # works with no extra wiring.
    await actor.setup_env.call(master_addr, chosen_port)

    print("=" * 72, flush=True)
    print(
        f"[titan_pretrain] launching TorchTitan: config={tt_config} "
        f"cwd={tt_cwd} overrides={overrides}",
        flush=True,
    )
    print("=" * 72, flush=True)

    t0 = time.time()
    results = await actor.run.call(
        toml_path=str(tt_config),
        cwd=str(tt_cwd),
        overrides=list(overrides),
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
                f"[titan_pretrain] rank={rank} host={host} elapsed={secs:.1f}s ok={ok}",
                flush=True,
            )
            if not ok:
                failures.append((rank, str(payload)))
        else:
            print(f"[titan_pretrain] rank=? payload={payload!r}", flush=True)

    print("=" * 72, flush=True)
    print(
        f"[titan_pretrain] all ranks returned in {elapsed:.1f}s",
        flush=True,
    )
    if failures:
        for rank, msg in failures:
            print(f"[titan_pretrain] FAIL rank={rank}: {msg}", file=sys.stderr)
        return 30

    return 0


def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    titan = _load_titan_section(args.algo_config)

    workers = [w.strip() for w in args.bare_metal_workers.split(",") if w.strip()]
    if not workers:
        print(
            "[titan_pretrain] --bare-metal-workers parsed empty",
            file=sys.stderr,
        )
        return 2

    # Compose final TT override list.  ``--steps N`` from the CLI
    # wins over titan.overrides because it's appended last
    # (argparse / tyro lets later flags override earlier ones).
    overrides = [str(x) for x in titan["overrides"]]
    if args.steps is not None:
        overrides.extend(["--training.steps", str(args.steps)])

    try:
        return asyncio.run(
            _drive_training(
                workers=workers,
                tt_config=titan["config"],
                tt_cwd=titan["cwd"],
                procs_per_host=int(titan["procs_per_host"]),
                master_port=int(titan["master_port"]),
                overrides=overrides,
            )
        )
    except KeyboardInterrupt:
        print("[titan_pretrain] interrupted -- tearing down", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(
            f"[titan_pretrain] unexpected error: {exc!r}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 20


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
