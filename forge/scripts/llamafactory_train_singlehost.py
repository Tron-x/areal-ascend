#!/usr/bin/env python3
"""Single-host LlamaFactory trainer driver -- B-mini integration.

Mirror of :mod:`forge.scripts.titan_train_multinode` but targeting a
**single** host with N NPUs, since LlamaFactory + ``Qwen3-VL-4B`` fits
comfortably in one box.  Once this is green, the same actor can be
driven from a multi-node SSHJob driver with no changes.

Why single-host first?
    The user explicitly asked to validate LF-on-NPU on one machine
    before paying the multi-node operational tax (SSHJob + worker
    daemons + cross-host HCCL bring-up).  Single-host smoke is a
    superset of "actor + FSDP env mapping + LF imports work end-to-end"
    -- the actual training math is identical to the multi-node case,
    only the rendezvous is local.

Topology
--------
* 1 host, ``--procs-per-host`` NPUs (default 4).  World size = procs.
* No SSHJob, no worker daemons.  We use Monarch's ``this_host()`` to
  spawn procs locally.  This works without ``worker_manager.sh``
  because the procs run inside the same process tree as the driver --
  they pick up CANN env from whatever shell launched the driver.

Lifecycle
---------
1. ``this_host().spawn_procs({"gpus": P})`` -- single proc mesh of size P.
2. Spawn :class:`LlamaFactoryTrainerActor` on that mesh.
3. Pull ``MASTER_ADDR`` / ``MASTER_PORT`` from rank 0; publish via
   ``setup_env`` to every rank.
4. ``await actor.run.call(lf_yaml, accelerate_yaml, cwd, overrides)``.
5. Per-rank ok/err report.  Cleanup is automatic when proc mesh drops.

Usage
-----
::

    # one-host smoke, 4 NPUs, 3 SFT steps, qwen3vl 4B + FSDP2
    /root/miniconda3/envs/monarch_ascend/bin/python \\
        forge/scripts/llamafactory_train_singlehost.py \\
        --lf-config examples/sft/qwen3vl_4b_lf_fsdp2_debug.yaml \\
        --accelerate-config examples/accelerate/lf_fsdp2_npu.yaml \\
        --lf-cwd /root/LlamaFactory \\
        --procs-per-host 4 \\
        --max-steps 3

The driver MUST run from inside the ``monarch_ascend`` conda env --
this is the only environment with ``torch_npu`` + LF + Monarch
co-installed.  See ``CLAUDE.md`` for env setup.

Exit codes
----------
* ``0``  every rank returned ``ok=True``
* ``2``  bad CLI args (yaml missing, etc.)
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


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run LlamaFactory's run_exp() across one host with N NPUs "
            "(B-mini single-host integration)."
        ),
    )
    p.add_argument(
        "--lf-config",
        type=Path,
        required=True,
        help="LlamaFactory training YAML.",
    )
    p.add_argument(
        "--accelerate-config",
        type=Path,
        required=True,
        help="Accelerate FSDP2 config YAML (FSDP env vars source).",
    )
    p.add_argument(
        "--lf-cwd",
        type=Path,
        default=Path("/root/LlamaFactory"),
        help=(
            "Working directory for every actor before run_exp().  Must "
            "be the LlamaFactory repo root so ``data/dataset_info.json`` "
            "and the bundled demo datasets resolve."
        ),
    )
    p.add_argument(
        "--procs-per-host",
        type=int,
        default=4,
        help="NPUs (== procs) on this host.",
    )
    p.add_argument(
        "--master-port",
        type=int,
        default=0,
        help=(
            "HCCL rendezvous port.  0 lets SPMDActor pick a free port "
            "on rank 0 (recommended for single-host)."
        ),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Convenience override for LF's ``max_steps``.  Equivalent "
            "to overriding the YAML key inline.  Lower this to 3 for "
            "smoke runs."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override LF YAML's ``output_dir``.  Useful for smoke runs.",
    )
    p.add_argument(
        "--use-modelscope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Set ``USE_MODELSCOPE_HUB=1`` so HuggingFace model lookups "
            "fall through to ModelScope (HF is not reachable from this "
            "cluster).  Default on."
        ),
    )
    return p


async def _run(
    *,
    lf_config: Path,
    accelerate_config: Path,
    lf_cwd: Path,
    procs_per_host: int,
    master_port: int,
    overrides: dict,
) -> int:
    """Drive one full single-host LlamaFactory training run."""
    from monarch.actor import this_host

    from forge.actors.llamafactory_trainer import LlamaFactoryTrainerActor

    print(f"[driver] spawning {procs_per_host} procs on this_host()", flush=True)
    proc_mesh = this_host().spawn_procs(per_host={"gpus": procs_per_host})

    actor = proc_mesh.spawn("lf_trainer", LlamaFactoryTrainerActor)

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

    await actor.setup_env.call(master_addr, chosen_port)

    print("=" * 72, flush=True)
    print(
        f"[driver] launching LlamaFactory: lf_yaml={lf_config} "
        f"accelerate_yaml={accelerate_config} cwd={lf_cwd}",
        flush=True,
    )
    if overrides:
        print(f"[driver] overrides = {overrides}", flush=True)
    print("=" * 72, flush=True)

    t0 = time.time()
    results = await actor.run.call(
        lf_yaml_path=str(lf_config),
        accelerate_yaml_path=str(accelerate_config),
        cwd=str(lf_cwd),
        overrides=overrides or None,
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


def main(argv: list[str]) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if not args.lf_config.exists():
        print(f"[driver] lf-config not found: {args.lf_config}", file=sys.stderr)
        return 2
    if not args.accelerate_config.exists():
        print(
            f"[driver] accelerate-config not found: {args.accelerate_config}",
            file=sys.stderr,
        )
        return 2
    if not args.lf_cwd.exists():
        print(f"[driver] lf-cwd not found: {args.lf_cwd}", file=sys.stderr)
        return 2

    if args.use_modelscope:
        # LF's loader checks USE_MODELSCOPE_HUB env to swap HF -> MS
        # for snapshot lookups.  Set it BEFORE ``import llamafactory``
        # in the actor procs by exporting it at the driver level so
        # the env is inherited by Monarch's spawned procs.
        os.environ["USE_MODELSCOPE_HUB"] = "1"

    overrides: dict = {}
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)

    try:
        return asyncio.run(
            _run(
                lf_config=args.lf_config.resolve(),
                accelerate_config=args.accelerate_config.resolve(),
                lf_cwd=args.lf_cwd.resolve(),
                procs_per_host=args.procs_per_host,
                master_port=args.master_port,
                overrides=overrides,
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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
