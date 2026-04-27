#!/usr/bin/env python3
"""Standalone multi-node TorchTitan trainer driver -- B-mini integration.

.. deprecated:: 2026-04
    Superseded by :mod:`forge.apps.titan_pretrain` driven via
    ``python -m forge launch examples/pretrain/llama3_titan_debug.yaml``
    (B-full).  The new path goes through the same launcher
    abstraction GRPO uses, so worker spawn / pre-flight / cleanup /
    profile selection are shared instead of forked.  This script is
    kept around for one release window because it's still the
    fastest way to iterate on :class:`TitanTrainerActor` itself
    without re-running the full ``forge launch`` lifecycle on
    every code change.

    For new work, prefer::

        python -m forge launch examples/pretrain/llama3_titan_debug.yaml \\
            --steps 5

Spawns one :class:`TitanTrainerActor` per (host, NPU) cell and asks it to
run TorchTitan's native ``Trainer.train()`` end-to-end.  This is the
smallest viable Monarch-orchestrated TorchTitan pretrain entry point on
NPU; no Forge GRPO loop, no vLLM, no torchstore weight-sync.

Why standalone (vs. ``forge launch``)?
    Historical -- this was the validation harness for the
    TorchTitan-on-NPU integration.  Once the smoke test here was
    green the same actor was lifted into ``forge.apps.titan_pretrain``
    so deployment patterns line up.  See ``forge launch --mode
    titan-pretrain`` (or just author an algo YAML with
    ``mode: titan-pretrain``) for the canonical entry point.

Topology assumed
----------------
* N hosts (default 2) listed in ``--hostfile``, each with P NPUs
  (default 4).  Total world size = ``N * P``.
* All hosts already running
  ``forge/scripts/worker_manager.sh start --profile pure-training``
  -- this driver only orchestrates **inside** the existing worker
  daemons, it does not start them.  Use ``--launch-workers`` if you
  want the driver to spin them up via :class:`_SSHJobFleet`.

  The ``pure-training`` profile is critical here: the alternative
  ``hixl-coexist`` profile sets ``HCCL_INTRA_ROCE_ENABLE=1`` (needed
  for GRPO + torchstore weight sync) which forces HCCL's intra-host
  pairs onto RoCE and costs ~10x intra-host bandwidth.  B-mini has
  no HiXL/torchstore in-process so it should run on the minimal
  ``pure-training`` env -- HCCL then auto-selects HCCS for
  intra-host pairs (``transport_p2p.cc:161 ... transporttype[HCCS]``)
  and RoCE only for inter-host pairs (``transport_ibverbs.cc:248
  ... transporttype[ROCE]``), which is the right answer.

* Hosts mutually reachable on ``--worker-port`` (Monarch TCP) and
  on ``--master-port`` (HCCL rendezvous).

Lifecycle
---------
1. (optional) Bring up worker daemons via ``ForgeSSHJob``.
2. ``attach_to_workers`` to get the unified :class:`HostMesh`.
3. ``host_mesh.spawn_procs(per_host={"gpus": P})`` -- single proc
   mesh of size ``N * P``.  The dim is named ``gpus`` so the
   inherited :class:`SPMDActor` picks the right dim for ``LOCAL_RANK``.
4. Spawn :class:`TitanTrainerActor` on that mesh.
5. Get ``MASTER_ADDR`` / ``MASTER_PORT`` from rank 0 via the
   ``get_host_port`` endpoint that ``SPMDActor`` already provides,
   then call ``setup_env`` to publish them to every rank.
6. ``await actor.run.call(toml_path, cwd, overrides)`` -- one TT
   training run finishes, the call returns one ``ok=True`` dict per
   rank.
7. Cleanup: stop proc mesh; if we launched workers, kill the SSHJob.

Usage
-----
::

    # 1) Bring up worker daemons in pure-training mode (one-time per
    #    cluster; survives across many B-mini runs):
    bash forge/scripts/worker_manager.sh start \\
        --hostfile forge/configs/hostfile.txt \\
        --profile pure-training

    # 2) Run B-mini smoke against the live workers:
    python forge/scripts/titan_train_multinode.py \\
        --hostfile forge/configs/hostfile.txt \\
        --tt-config /root/torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml \\
        --tt-cwd /root/torchtitan \\
        --procs-per-host 4 \\
        --steps 3

    # Driver also starts workers via SSHJob (parity with forge launch):
    python forge/scripts/titan_train_multinode.py \\
        --hostfile forge/configs/hostfile.txt \\
        --tt-config /root/torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml \\
        --tt-cwd /root/torchtitan \\
        --launch-workers --steps 3

    # Pass arbitrary TT overrides after ``--``:
    python forge/scripts/titan_train_multinode.py [...] -- \\
        --training.local_batch_size 2 --metrics.log_freq 1

Stale-file foot-gun (bare-metal dev only)
-----------------------------------------
``/root/AReaL`` is a per-host **local** filesystem on the current
bare-metal cluster (each container has its own copy), NOT a shared
mount.  Monarch unpickles :class:`TitanTrainerActor` on every remote
host, so any edit to either of these two files only takes effect on
hosts where the file has been updated:

* ``forge/actors/titan_trainer.py``
* ``forge/scripts/titan_train_multinode.py`` (this file)

Symptoms of forgetting to sync:

* ``ModuleNotFoundError: No module named 'forge.actors.titan_trainer'``
  on a remote rank (file missing on that host)
* ``TypeError: run() got an unexpected keyword argument ...`` on a
  remote rank (signature drift between hosts)
* Silent stale behavior (loss curve doesn't match local edits)

One-liner to push both files to every host in the hostfile::

    for H in $(awk 'NF && $1 !~ /^#/ {print $1}' forge/configs/hostfile.txt); do
        scp -P 36000 -o StrictHostKeyChecking=no \\
            forge/actors/titan_trainer.py \\
            "root@${H}:/root/AReaL/forge/actors/titan_trainer.py"
        scp -P 36000 -o StrictHostKeyChecking=no \\
            forge/scripts/titan_train_multinode.py \\
            "root@${H}:/root/AReaL/forge/scripts/titan_train_multinode.py"
    done

This whole section becomes obsolete once one of:

* B-full lands (``forge launch --backend titan-pretrain``) -- driver
  no longer ssh-execs a hand-rolled script, so there's nothing for
  the user to keep in sync.
* Cluster moves to image-based deployment (k8s/slurm with a frozen
  container image) -- code drift becomes structurally impossible.

Until then: edit, then run the loop above, then run the smoke.

Exit codes
----------
* ``0``  every rank returned ``ok=True``
* ``2``  bad CLI args (hostfile missing, etc.)
* ``10`` worker fleet failed to start
* ``20`` Monarch attach / spawn failed
* ``30`` at least one rank reported ``ok=False`` or raised
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _read_hosts(hostfile: Path) -> list[str]:
    """Parse a DeepSpeed/MPI-style hostfile.

    First whitespace-delimited token per non-comment line is the IP.
    Mirrors ``forge.cli.launch._read_hosts`` so the hostfile format
    stays identical between ``forge launch`` and this script.
    """
    hosts: list[str] = []
    for raw in hostfile.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        hosts.append(line.split()[0])
    return hosts


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run TorchTitan's native Trainer.train() across multiple "
            "Monarch worker hosts (B-mini integration)."
        ),
        epilog=(
            "Anything after a literal `--` is forwarded verbatim as "
            "extra TorchTitan CLI overrides "
            "(e.g. `--training.local_batch_size 2`)."
        ),
    )
    p.add_argument(
        "--hostfile",
        type=Path,
        required=True,
        help="DeepSpeed-style hostfile (`forge/configs/hostfile.txt`).",
    )
    p.add_argument(
        "--tt-config",
        type=Path,
        required=True,
        help="TorchTitan .toml config (e.g. debug_model.toml).",
    )
    p.add_argument(
        "--tt-cwd",
        type=Path,
        default=Path("/root/torchtitan"),
        help=(
            "Working directory for every actor before TorchTitan's "
            "Trainer is constructed.  Required because TT debug "
            "configs use relative paths to bundled tokenizer + "
            "c4_test fixtures."
        ),
    )
    p.add_argument(
        "--procs-per-host",
        type=int,
        default=4,
        help="NPUs (== procs) per host.  Default 4 to match the 4+4 layout.",
    )
    p.add_argument(
        "--worker-port",
        type=int,
        default=22222,
        help="Monarch TCP worker port (must match worker_manager.sh).",
    )
    p.add_argument(
        "--ssh-port",
        type=int,
        default=36000,
        help="SSH port used by --launch-workers and stop.",
    )
    p.add_argument(
        "--cann",
        type=str,
        default=os.environ.get("CANN_HOME", "/usr/local/Ascend/cann-9.0.0-beta.1"),
        help="CANN install root, only consumed by --launch-workers.",
    )
    p.add_argument(
        "--master-port",
        type=int,
        default=0,
        help=(
            "HCCL rendezvous port for torch.distributed.  0 lets "
            "Monarch's SPMDActor pick a free port on rank 0 (recommended)."
        ),
    )
    p.add_argument(
        "--launch-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, the driver spawns Monarch workers itself via "
            "ForgeSSHJob and tears them down on exit.  Default off "
            "(assumes worker_manager.sh start was already run)."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Convenience override for `--training.steps N`.  Equivalent "
            "to passing `-- --training.steps N`; both work, but this "
            "form is friendlier for smoke tests."
        ),
    )
    return p


async def _run(
    *,
    workers: list[str],
    tt_config: Path,
    tt_cwd: Path,
    procs_per_host: int,
    master_port: int,
    extra_overrides: list[str],
) -> int:
    """Drive one full multi-host TorchTitan training run.

    Returns 0 if every rank returned ``ok=True``; ``30`` otherwise.
    Wraps the spawn/run/cleanup so the outer ``main`` can layer
    SSHJob lifecycle on top.
    """
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers

    from forge.actors.titan_trainer import TitanTrainerActor

    # Same transport choice as the rest of forge's bare-metal stack
    # (see forge/scripts/test_weight_sync_2node.py); without this the
    # default in-proc transport refuses cross-host attach.
    configure(default_transport=ChannelTransport.TcpWithHostname)

    print(f"[driver] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized
    n_hosts = hosts.size()
    print(
        f"[driver] attached to {n_hosts} host(s); "
        f"world_size = {n_hosts * procs_per_host}",
        flush=True,
    )

    # Single proc mesh spanning every host.  The dim name ``gpus`` is
    # what SPMDActor's __init__ keys off when computing LOCAL_RANK
    # (`point.extent.labels[-1]`).  Renaming to ``procs`` would
    # silently break local_rank wiring for any subclass.
    proc_mesh = hosts.spawn_procs(per_host={"gpus": procs_per_host})

    actor = proc_mesh.spawn("titan_trainer", TitanTrainerActor)

    # Pull MASTER_ADDR/PORT from rank 0 -- the SPMDActor.get_host_port
    # endpoint already exists for exactly this case.  We let it pick a
    # free port on rank 0's host unless --master-port pinned one.
    first_values = dict.fromkeys(proc_mesh._labels, 0)
    rank0 = actor.slice(**first_values)
    if master_port > 0:
        master_addr = (await rank0.get_host_port.call_one(None))[0]
        chosen_port = master_port
    else:
        master_addr, chosen_port = await rank0.get_host_port.call_one(None)
    print(
        f"[driver] rendezvous = {master_addr}:{chosen_port}",
        flush=True,
    )

    # Publish the rendezvous + RANK/LOCAL_RANK/WORLD_SIZE to every
    # rank.  After this returns every proc has an env that looks
    # exactly like a torchrun launch -- TT's `env://` init_pg path
    # works with no extra wiring.
    await actor.setup_env.call(master_addr, chosen_port)

    print("=" * 72, flush=True)
    print(
        f"[driver] launching TorchTitan: config={tt_config} cwd={tt_cwd} "
        f"overrides={extra_overrides}",
        flush=True,
    )
    print("=" * 72, flush=True)

    t0 = time.time()
    # ``call`` (not ``call_one``) so every rank's return is collected;
    # SPMDActor pattern fans out across the whole mesh.  No master_addr
    # arg needed -- already published via setup_env above.
    results = await actor.run.call(
        toml_path=str(tt_config),
        cwd=str(tt_cwd),
        overrides=list(extra_overrides),
    )
    elapsed = time.time() - t0

    # ``results`` is an ActorMeshResult; iterate to surface per-rank
    # ok/err.  We tolerate both list-like and dict-like shapes here
    # because Monarch's return shape varies by version.
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
                f"[driver] rank={rank} host={host} elapsed={secs:.1f}s ok={ok}",
                flush=True,
            )
            if not ok:
                failures.append((rank, str(payload)))
        else:
            print(f"[driver] rank=? payload={payload!r}", flush=True)

    print("=" * 72, flush=True)
    print(f"[driver] all ranks returned in {elapsed:.1f}s", flush=True)
    if failures:
        for rank, msg in failures:
            print(f"[driver] FAIL rank={rank}: {msg}", file=sys.stderr)
        return 30

    return 0


def _maybe_launch_workers(args, hosts: list[str]):
    """Bring up Monarch worker daemons via :class:`ForgeSSHJob`.

    Returns the live :class:`_SSHJobFleet` (so caller can stop it on
    exit) or ``None`` when ``--no-launch-workers``.  Reuses the exact
    fleet implementation that ``forge launch`` uses, so behavior /
    timeout / readiness checks stay in one place.
    """
    if not args.launch_workers:
        return None

    from forge.cli.launch import _SSHJobFleet  # type: ignore[attr-defined]

    fleet = _SSHJobFleet(
        hostfile=args.hostfile,
        worker_port=args.worker_port,
        ssh_port=args.ssh_port,
        cann=args.cann,
    )
    rc = fleet.start()
    if rc != 0:
        raise SystemExit(rc if rc >= 10 else 10)
    return fleet


def main(argv: list[str]) -> int:
    if "--" in argv:
        dash_idx = argv.index("--")
        own_argv = argv[:dash_idx]
        passthrough = argv[dash_idx + 1 :]
    else:
        own_argv = argv
        passthrough = []

    parser = _build_argparser()
    args = parser.parse_args(own_argv)

    if not args.hostfile.exists():
        print(f"[driver] hostfile not found: {args.hostfile}", file=sys.stderr)
        return 2
    if not args.tt_config.exists():
        print(f"[driver] tt-config not found: {args.tt_config}", file=sys.stderr)
        return 2
    if not args.tt_cwd.exists():
        print(f"[driver] tt-cwd not found: {args.tt_cwd}", file=sys.stderr)
        return 2

    hosts = _read_hosts(args.hostfile)
    if not hosts:
        print(f"[driver] empty hostfile: {args.hostfile}", file=sys.stderr)
        return 2

    workers = [f"tcp://{h}:{args.worker_port}" for h in hosts]

    # Compose the final TT override list.  ``--steps N`` is sugar for
    # ``-- --training.steps N``; both stack with --, with --steps
    # taking precedence (it's appended last, and tyro / argparse let
    # later flags override earlier ones).
    extra = list(passthrough)
    if args.steps is not None:
        extra.extend(["--training.steps", str(args.steps)])

    fleet = None
    try:
        fleet = _maybe_launch_workers(args, hosts)
        return asyncio.run(
            _run(
                workers=workers,
                tt_config=args.tt_config,
                tt_cwd=args.tt_cwd,
                procs_per_host=args.procs_per_host,
                master_port=args.master_port,
                extra_overrides=extra,
            )
        )
    except KeyboardInterrupt:
        print("[driver] interrupted -- tearing down", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[driver] unexpected error: {exc!r}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 20
    finally:
        if fleet is not None:
            try:
                fleet.stop()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[driver] fleet.stop() raised {exc!r} "
                    f"(some remote procs may still be live)",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
