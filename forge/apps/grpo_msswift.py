"""``forge.apps.grpo_msswift`` -- ms-swift GRPO via Monarch + Forge.

Phase A of the framework first principles
(``.cursor/rules/framework-first-principles.mdc``): make the existing
ms-swift GRPO PoC swappable / observable / restartable through the
unified Monarch + Forge orchestration layer instead of bare bash +
torchrun.

Topology (matches yesterday's PoC)::

    Rollout pool              Trainer pool
    ┌─────────────────┐      ┌─────────────────────────────┐
    │ Host A (.26)    │      │ Host B (.23)                │
    │ MsSwiftRollout  │◄─HTTP┤ MsSwiftTrainerActor x4      │
    │ Actor (1 proc,  │ /infer│ (procs_per_host=4)         │
    │ TP=4 vLLM)      │      │ + areal.weight_sync         │
    │ /areal_* shim   │◄─HCCL┤ WeightSyncClient broadcasts │
    └─────────────────┘      └─────────────────────────────┘

The two halves run on disjoint Monarch worker pools (separate
``attach_to_workers`` calls), so the algo author lays them out via
``--rollout-workers`` and ``--trainer-workers``.  In the colocate
single-host smoke this is the same URI in both flags -- the launcher
preset will eventually compose this for you.

YAML schema (new ``msswift:`` block)::

    msswift:
      rollout:
        port: 8000                    # rollout HTTP port
        swift_args:                   # passed to `swift rollout`
          model: /root/.cache/.../Qwen3-0___6B
          vllm_tensor_parallel_size: 4
          vllm_data_parallel_size: 1
          vllm_use_async_engine: false
          torch_dtype: bfloat16
          max_model_len: 1024
      trainer:
        procs_per_host: 4
        master_port: 29501            # trainer's own DDP rendezvous
        ws_master_port: 49500         # cross-mesh weight-sync TCPStore
        swift_args:                   # passed to `swift rlhf`
          rlhf_type: grpo
          model: /root/.cache/.../Qwen3-0___6B
          tuner_type: full
          dataset: /tmp/.../gsm8k_smoke.jsonl
          torch_dtype: bfloat16
          max_steps: 1
          per_device_train_batch_size: 1
          gradient_accumulation_steps: 1
          learning_rate: 1.0e-6
          ...

The ``vllm_server_base_url`` field is **deliberately absent** from
``trainer.swift_args`` -- the driver discovers the rollout actor's host
via ``MsSwiftRolloutActor.host_info()`` and injects it before invoking
the trainer.

Usage
-----
Direct invocation (after ``forge launch`` already started workers, e.g.
for re-runs against the same fleet)::

    python -m forge.apps.grpo_msswift \\
        --algo-config examples/grpo/qwen3_06b_msswift_npu.yaml \\
        --rollout-workers tcp://192.168.0.26:22222 \\
        --trainer-workers tcp://192.168.0.23:22222

Single-host (colocate) smoke -- same URI both sides::

    python -m forge.apps.grpo_msswift \\
        --algo-config examples/grpo/qwen3_06b_msswift_npu.yaml \\
        --rollout-workers tcp://192.168.0.26:22222 \\
        --trainer-workers tcp://192.168.0.26:22222

Exit codes
----------
* ``0``   trainer ranks all returned ``ok=True``
* ``2``   bad CLI args / YAML
* ``20``  Monarch attach / spawn failed
* ``30``  at least one trainer rank reported ``ok=False`` or raised
* ``40``  rollout actor never became ready
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
        prog="forge.apps.grpo_msswift",
        description=(
            "ms-swift GRPO driver wired through Monarch + Forge "
            "(rollout actor + trainer actors + areal.weight_sync). "
            "Normally invoked via `forge launch --mode grpo-msswift`."
        ),
    )
    p.add_argument(
        "--algo-config",
        type=Path,
        required=True,
        help=(
            "Algorithm YAML with a top-level ``msswift:`` block "
            "(rollout/trainer subsections; see module docstring)."
        ),
    )
    p.add_argument(
        "--rollout-workers",
        type=str,
        required=True,
        help=(
            "Comma-separated list of Monarch worker URIs hosting the "
            "rollout pool, e.g. ``tcp://192.168.0.26:22222``.  Usually "
            "one URI; multi-replica rollout is a Phase C concern."
        ),
    )
    p.add_argument(
        "--trainer-workers",
        type=str,
        required=True,
        help=(
            "Comma-separated list of Monarch worker URIs hosting the "
            "trainer pool, e.g. ``tcp://192.168.0.23:22222``.  May "
            "equal --rollout-workers for colocate single-host smokes."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Convenience override for ``msswift.trainer.swift_args.max_steps`` "
            "-- merged into the trainer args at runtime.  CLI > YAML."
        ),
    )
    p.add_argument(
        "--rollout-port",
        type=int,
        default=None,
        help="Override ``msswift.rollout.port`` from CLI.",
    )
    return p


def _load_msswift_section(algo_yaml: Path) -> dict:
    """Read and validate the ``msswift:`` block of the algo YAML."""
    import yaml

    if not algo_yaml.is_file():
        print(
            f"[grpo_msswift] algo YAML not found: {algo_yaml}",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        data = yaml.safe_load(algo_yaml.read_text()) or {}
    except yaml.YAMLError as e:
        print(
            f"[grpo_msswift] algo YAML unparseable ({algo_yaml}): {e}",
            file=sys.stderr,
        )
        sys.exit(2)

    block = data.get("msswift")
    if not isinstance(block, dict):
        print(
            "[grpo_msswift] algo YAML missing required ``msswift:`` block "
            "(see examples/grpo/qwen3_06b_msswift_npu.yaml).",
            file=sys.stderr,
        )
        sys.exit(2)

    rollout = block.get("rollout")
    trainer = block.get("trainer")
    if not isinstance(rollout, dict) or not isinstance(trainer, dict):
        print(
            "[grpo_msswift] msswift block must have both ``rollout:`` "
            "and ``trainer:`` mappings.",
            file=sys.stderr,
        )
        sys.exit(2)

    rollout.setdefault("port", 8000)
    rollout.setdefault("swift_args", {})
    trainer.setdefault("procs_per_host", 4)
    trainer.setdefault("master_port", 0)  # 0 = pick automatically from rank 0
    trainer.setdefault("ws_master_port", 49500)
    trainer.setdefault("swift_args", {})
    trainer.setdefault("cwd", "/root/ms-swift")
    block.setdefault("use_modelscope", True)
    block.setdefault("rollout_ready_timeout_s", 600.0)

    if not isinstance(rollout["swift_args"], dict):
        print(
            "[grpo_msswift] msswift.rollout.swift_args must be a mapping",
            file=sys.stderr,
        )
        sys.exit(2)
    if not isinstance(trainer["swift_args"], dict):
        print(
            "[grpo_msswift] msswift.trainer.swift_args must be a mapping",
            file=sys.stderr,
        )
        sys.exit(2)

    return block


def _normalise_results(results) -> list:
    """``actor.run.call`` returns a ValueMesh-ish; coerce to a flat list of dicts."""
    try:
        items = list(results)
    except TypeError:
        return [results]
    out = []
    for entry in items:
        if isinstance(entry, tuple):
            out.append(entry[1])
        else:
            out.append(entry)
    return out


async def _drive(
    *,
    rollout_workers: list[str],
    trainer_workers: list[str],
    msswift_block: dict[str, Any],
) -> int:
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers

    from forge.actors.msswift_rollout import MsSwiftRolloutActor
    from forge.actors.msswift_trainer import MsSwiftTrainerActor

    # Same transport choice as the rest of forge's bare-metal stack.
    configure(default_transport=ChannelTransport.TcpWithHostname)

    rollout_cfg = msswift_block["rollout"]
    trainer_cfg = msswift_block["trainer"]
    use_modelscope = bool(msswift_block["use_modelscope"])
    rollout_ready_timeout = float(msswift_block["rollout_ready_timeout_s"])

    # ------------------------------------------------------------------
    # Phase 1: bring up rollout
    # ------------------------------------------------------------------
    print(
        f"[grpo_msswift] attaching to rollout workers: {rollout_workers}",
        flush=True,
    )
    rollout_hosts = attach_to_workers(
        ca="trust_all_connections", workers=rollout_workers
    )
    await rollout_hosts.initialized
    print(
        f"[grpo_msswift] rollout pool: {rollout_hosts.size()} host(s)",
        flush=True,
    )

    # ONE actor per rollout host (vLLM TP fans out internally).
    rollout_mesh = rollout_hosts.spawn_procs(per_host={"gpus": 1})
    rollout_actor = rollout_mesh.spawn("rollout", MsSwiftRolloutActor)

    # Discover the rollout host's NIC so the trainer knows where to POST.
    info = await rollout_actor.host_info.call_one()
    rollout_ip = info["ip"]
    rollout_port = int(rollout_cfg["port"])
    rollout_url = f"http://{rollout_ip}:{rollout_port}"
    print(
        f"[grpo_msswift] rollout host: ip={rollout_ip} hostname={info['hostname']} "
        f"-> {rollout_url}",
        flush=True,
    )

    print(
        f"[grpo_msswift] starting rollout server (port={rollout_port}, "
        f"args={list(rollout_cfg['swift_args'].keys())}) ...",
        flush=True,
    )
    started = await rollout_actor.start.call_one(
        swift_args=dict(rollout_cfg["swift_args"]),
        host="0.0.0.0",
        port=rollout_port,
        use_modelscope=use_modelscope,
    )
    print(
        f"[grpo_msswift] rollout started: pid={started['pid']} "
        f"world_size={started['world_size']}",
        flush=True,
    )

    # Wait for the rollout to actually serve /health/.  This drives the
    # FastAPI lifespan handshake -- every spawned vLLM child must report
    # "ready" before we proceed, otherwise the trainer's first
    # init_communicator call would hit a half-built worker.
    try:
        ready = await rollout_actor.wait_ready.call_one(
            timeout_s=rollout_ready_timeout,
        )
    except (TimeoutError, RuntimeError) as e:
        print(f"[grpo_msswift] FATAL rollout never became ready: {e}", file=sys.stderr)
        await _safe_stop(rollout_actor)
        return 40
    print(
        f"[grpo_msswift] rollout ready in {ready['elapsed_s']:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase 2: bring up trainer & drive one full GRPO run
    # ------------------------------------------------------------------
    procs_per_host = int(trainer_cfg["procs_per_host"])
    master_port = int(trainer_cfg["master_port"])
    ws_master_port = int(trainer_cfg["ws_master_port"])

    print(
        f"[grpo_msswift] attaching to trainer workers: {trainer_workers}",
        flush=True,
    )

    # Colocate optimisation: if rollout and trainer pools point at the
    # same worker URI, reuse the same hosts handle.  Otherwise re-attach.
    if trainer_workers == rollout_workers:
        trainer_hosts = rollout_hosts
        print("[grpo_msswift] colocate mode: reusing rollout pool", flush=True)
    else:
        trainer_hosts = attach_to_workers(
            ca="trust_all_connections", workers=trainer_workers
        )
        await trainer_hosts.initialized
    n_train_hosts = trainer_hosts.size()
    world_size = n_train_hosts * procs_per_host
    print(
        f"[grpo_msswift] trainer pool: {n_train_hosts} host(s); "
        f"world_size = {world_size}",
        flush=True,
    )

    trainer_mesh = trainer_hosts.spawn_procs(per_host={"gpus": procs_per_host})
    trainer_actor = trainer_mesh.spawn("msswift_trainer", MsSwiftTrainerActor)

    # Pull rendezvous (MASTER_ADDR/PORT) for the trainer's own DDP group
    # from rank 0.  This is the per-trainer torchrun-equivalent rendezvous,
    # NOT the cross-mesh weight-sync TCPStore (that one is at
    # ws_master_port and is bound by the cross-mesh code in the glue).
    first_values = dict.fromkeys(trainer_mesh._labels, 0)
    rank0 = trainer_actor.slice(**first_values)
    if master_port > 0:
        master_addr = (await rank0.get_host_port.call_one(None))[0]
        chosen_port = master_port
    else:
        master_addr, chosen_port = await rank0.get_host_port.call_one(None)
    print(
        f"[grpo_msswift] trainer DDP rendezvous = {master_addr}:{chosen_port}",
        flush=True,
    )
    await trainer_actor.setup_env.call(master_addr, chosen_port)

    # We deliberately do NOT set ws_master_addr from the driver: rollout
    # vLLM workers connect in via the IP form, and ``master_addr`` from
    # ``get_host_port`` is ``socket.gethostname()`` (a hostname which may
    # not resolve from the rollout host's DNS).  The glue's
    # ``_resolve_master_addr`` runs INSIDE each trainer rank and prefers
    # ``socket.gethostbyname(socket.gethostname())`` which gives the
    # primary NIC IP -- safe across hosts.  We just pin the port.
    ws_master_addr_log = "<auto-resolve>"

    print("=" * 72, flush=True)
    print(
        f"[grpo_msswift] launching ms-swift trainer: "
        f"vllm_server={rollout_url} ws_master={ws_master_addr_log}:{ws_master_port}",
        flush=True,
    )
    print("=" * 72, flush=True)

    t0 = time.time()
    failures: list[tuple[Any, str]] = []
    try:
        results = await trainer_actor.run.call(
            swift_args=dict(trainer_cfg["swift_args"]),
            vllm_server_base_url=rollout_url,
            cwd=str(trainer_cfg["cwd"]),
            ws_master_port=ws_master_port,
            use_modelscope=use_modelscope,
        )
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - t0
        print(
            f"[grpo_msswift] trainer raised after {elapsed:.1f}s: {e!r}",
            file=sys.stderr,
        )
        await _safe_trainer_teardown(trainer_actor)
        await _safe_stop(rollout_actor)
        return 30
    elapsed = time.time() - t0

    payloads = _normalise_results(results)
    for payload in payloads:
        if isinstance(payload, dict):
            ok = bool(payload.get("ok"))
            rank = payload.get("rank", "?")
            host = payload.get("host", "?")
            secs = payload.get("elapsed_s", -1)
            print(
                f"[grpo_msswift] rank={rank} host={host} elapsed={secs:.1f}s ok={ok}",
                flush=True,
            )
            if not ok:
                failures.append((rank, str(payload)))
        else:
            print(f"[grpo_msswift] rank=? payload={payload!r}", flush=True)

    print("=" * 72, flush=True)
    print(
        f"[grpo_msswift] trainer ranks returned in {elapsed:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase 3: tear down trainer (waitpid dataloader workers, destroy PG)
    # then rollout (kills vLLM workers, frees HCCL ports).
    # ------------------------------------------------------------------
    await _safe_trainer_teardown(trainer_actor)
    await _safe_stop(rollout_actor)

    if failures:
        for rank, msg in failures:
            print(f"[grpo_msswift] FAIL rank={rank}: {msg}", file=sys.stderr)
        return 30
    return 0


async def _safe_stop(rollout_actor) -> None:
    """Best-effort rollout shutdown.  Never raises.

    Logs the teardown report (terminated/killed/survivors) so a
    survivors>0 case shows up in the driver log even though we swallow
    the exception path.
    """
    try:
        report = await rollout_actor.teardown.call_one()
        print(f"[grpo_msswift] rollout teardown: {report}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(
            f"[grpo_msswift] rollout teardown raised (ignored): {e!r}", file=sys.stderr
        )


async def _safe_trainer_teardown(trainer_actor) -> None:
    """Best-effort trainer shutdown across all SPMD ranks.  Never raises.

    Each rank reaps its own dataloader workers + destroys its DDP group,
    so we ``call`` (broadcast) instead of ``call_one``.  Reports per-rank
    so survivors>0 on any rank shows up.
    """
    try:
        reports = await trainer_actor.teardown.call()
        for payload in _normalise_results(reports):
            print(f"[grpo_msswift] trainer teardown: {payload}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(
            f"[grpo_msswift] trainer teardown raised (ignored): {e!r}", file=sys.stderr
        )


def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    block = _load_msswift_section(args.algo_config)

    rollout_workers = [w.strip() for w in args.rollout_workers.split(",") if w.strip()]
    trainer_workers = [w.strip() for w in args.trainer_workers.split(",") if w.strip()]
    if not rollout_workers or not trainer_workers:
        print(
            "[grpo_msswift] --rollout-workers and --trainer-workers must be non-empty",
            file=sys.stderr,
        )
        return 2

    # CLI overrides win over YAML.
    if args.steps is not None:
        block["trainer"]["swift_args"]["max_steps"] = int(args.steps)
    if args.rollout_port is not None:
        block["rollout"]["port"] = int(args.rollout_port)

    if block["use_modelscope"]:
        os.environ.setdefault("USE_MODELSCOPE_HUB", "1")

    try:
        return asyncio.run(
            _drive(
                rollout_workers=rollout_workers,
                trainer_workers=trainer_workers,
                msswift_block=block,
            )
        )
    except KeyboardInterrupt:
        print("[grpo_msswift] interrupted -- tearing down", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"[grpo_msswift] unexpected error: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        return 20


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
