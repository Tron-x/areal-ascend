"""``forge.apps.inference_bench`` -- 2-node vLLM benchmark entry.

Runs on the driver host after ``forge launch`` brings up Monarch
worker daemons.  Reads a two-layer YAML (algo YAML composed with
cluster preset by ``forge/cli/presets.py``), spawns one
:class:`VLLMInferenceActor` per pool host, and drives a parallel
synthetic-load benchmark across all replicas, aggregating per-host
metrics into a final summary.

Mirror of :mod:`forge.apps.titan_pretrain` -- same orchestration
shape, just inference instead of training.  Where titan_pretrain
spawns N (host, NPU) actors and runs ``Trainer.train()``, this
spawns one actor per host (each owning all NPUs for vLLM TP) and
runs ``run_benchmark()``.

Usage
-----
Normally invoked indirectly via::

    python -m forge launch examples/inference/qwen2_tp8_bench.yaml

Direct invocation (after ``forge launch`` already started workers,
e.g. for re-runs against the same fleet)::

    python -m forge.apps.inference_bench \\
        --algo-config examples/inference/qwen2_tp8_bench.yaml \\
        --bare-metal-workers tcp://192.168.0.26:22222,tcp://192.168.0.23:22222 \\
        --duration 30

Exit codes
----------
* ``0``   every replica returned a metrics dict
* ``2``   bad CLI args / YAML
* ``20``  Monarch attach / spawn failed
* ``30``  at least one replica raised
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
        prog="forge.apps.inference_bench",
        description=(
            "Multi-replica vLLM inference benchmark driver.  Normally "
            "invoked via `forge launch`; can be run directly when "
            "iterating against an already-started worker fleet."
        ),
    )
    p.add_argument(
        "--algo-config",
        type=Path,
        required=True,
        help=(
            "Algorithm YAML with a top-level ``inference:`` block "
            "(model/tensor_parallel_size/dtype/.../benchmark)."
        ),
    )
    p.add_argument(
        "--bare-metal-workers",
        type=str,
        required=True,
        help=(
            "Comma-separated list of Monarch worker URIs, e.g. "
            "``tcp://192.168.0.26:22222,tcp://192.168.0.23:22222``.  "
            "Same flag GRPO and titan-pretrain use."
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Convenience override for ``inference.benchmark.duration_seconds``.  "
            "Always wins over the YAML value when set.  Useful for ad-hoc "
            "longer runs without editing the YAML."
        ),
    )
    return p


def _load_inference_section(algo_yaml: Path) -> dict[str, Any]:
    """Read and validate the ``inference:`` block of the algo YAML.

    Returns the dict with required keys populated and optional
    keys defaulted.  Raises ``SystemExit(2)`` with a clear message
    on any structural problem.
    """
    import yaml

    if not algo_yaml.is_file():
        print(
            f"[inference_bench] algo YAML not found: {algo_yaml}",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        data = yaml.safe_load(algo_yaml.read_text()) or {}
    except yaml.YAMLError as e:
        print(
            f"[inference_bench] algo YAML unparseable ({algo_yaml}): {e}",
            file=sys.stderr,
        )
        sys.exit(2)

    inference = data.get("inference")
    if not isinstance(inference, dict):
        print(
            "[inference_bench] algo YAML missing required ``inference:`` block "
            "(see examples/inference/qwen2_tp8_bench.yaml)",
            file=sys.stderr,
        )
        sys.exit(2)

    if not inference.get("model"):
        print(
            "[inference_bench] algo YAML ``inference.model`` is required",
            file=sys.stderr,
        )
        sys.exit(2)

    inference.setdefault("tensor_parallel_size", 8)
    inference.setdefault("dtype", "bfloat16")
    inference.setdefault("max_model_len", 4096)
    inference.setdefault("gpu_memory_utilization", 0.85)
    inference.setdefault("enforce_eager", True)
    inference.setdefault("seed", 1)

    bench = inference.setdefault("benchmark", {})
    if not isinstance(bench, dict):
        print(
            "[inference_bench] algo YAML ``inference.benchmark`` must be a dict",
            file=sys.stderr,
        )
        sys.exit(2)
    bench.setdefault("duration_seconds", 30.0)
    bench.setdefault("concurrency", 16)
    bench.setdefault("input_tokens", 256)
    bench.setdefault("output_tokens", 128)
    bench.setdefault("seed", 0)

    return inference


def _format_per_host_line(r: dict[str, Any]) -> str:
    """One-liner per-host metrics row, matching the plan's output spec."""
    return (
        f"[inference_bench] host={r.get('host', '?')} "
        f"throughput={r.get('throughput_tps', 0.0):.1f} tok/s "
        f"req/s={r.get('throughput_rps', 0.0):.2f} "
        f"batch_p50={r.get('batch_latency_ms', {}).get('p50', 0):.0f}ms "
        f"batch_p99={r.get('batch_latency_ms', {}).get('p99', 0):.0f}ms "
        f"batches={r.get('batches', 0)} "
        f"requests={r.get('requests', 0)} "
        f"mean_out_tok={r.get('mean_output_tokens', 0):.1f}"
    )


async def _drive_benchmark(
    *,
    workers: list[str],
    inference: dict[str, Any],
) -> int:
    """Spawn one actor per worker host, run benchmarks in parallel, aggregate.

    Body intentionally parallels :func:`forge.apps.titan_pretrain._drive_training`
    so the two flows stay structurally similar -- only the actor
    class and the call payload differ.  When a future mode-registry
    refactor lands, both functions become trivial wrappers around
    the same fan-out-then-aggregate primitive.
    """
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers

    from forge.actors.vllm_inference import VLLMInferenceActor

    # Same transport as the rest of forge's bare-metal stack.  The
    # default in-proc transport refuses cross-host attach.
    configure(default_transport=ChannelTransport.TcpWithHostname)

    print(f"[inference_bench] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized
    n_hosts = hosts.size()
    print(
        f"[inference_bench] attached to {n_hosts} host(s); "
        f"will spawn 1 vLLM TP={inference['tensor_parallel_size']} replica per host",
        flush=True,
    )

    # ONE actor per host.  ``per_host={"gpus": 1}`` gives us a
    # single proc per host even though each host has 8 NPUs --
    # vLLM's MP executor inside that single proc fans out across
    # all 8 NPUs internally.
    #
    # The dim name "gpus" is the same convention SPMDActor reads
    # for LOCAL_RANK; we don't subclass SPMDActor here so the name
    # is purely cosmetic for VLLMInferenceActor, but keeping it
    # consistent across actor types makes proc_mesh inspection
    # outputs uniform.
    proc_mesh = hosts.spawn_procs(per_host={"gpus": 1})
    actor = proc_mesh.spawn("generator", VLLMInferenceActor)

    bench = inference["benchmark"]

    # ------------------------------------------------------------------
    # Phase 1: init.  Heavy (model load + KV-cache alloc + optional
    # graph capture) -- ~15-30s for Qwen2.5-1.5B at TP=8.  Driven on
    # all replicas in parallel via .call() -- both replicas overlap
    # their HF cache reads and NPU memory allocation.
    # ------------------------------------------------------------------
    print("=" * 72, flush=True)
    print(
        f"[inference_bench] initialising vLLM on {n_hosts} replica(s): "
        f"model={inference['model']} tp={inference['tensor_parallel_size']}",
        flush=True,
    )
    print("=" * 72, flush=True)

    t_init = time.time()
    init_results = await actor.init.call(
        model=str(inference["model"]),
        tp_size=int(inference["tensor_parallel_size"]),
        dtype=str(inference["dtype"]),
        max_model_len=int(inference["max_model_len"]),
        gpu_memory_utilization=float(inference["gpu_memory_utilization"]),
        enforce_eager=bool(inference["enforce_eager"]),
        seed=int(inference["seed"]),
    )
    init_elapsed = time.time() - t_init

    init_payloads = _normalise_results(init_results)
    for payload in init_payloads:
        if isinstance(payload, dict):
            print(
                f"[inference_bench] init host={payload.get('host', '?')} "
                f"in {payload.get('init_s', -1):.1f}s",
                flush=True,
            )

    print(
        f"[inference_bench] all replicas ready in {init_elapsed:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase 2: benchmark.  Parallel across replicas via .call()
    # again; each replica drives its own loop independently.  We
    # measure end-to-end wall time across the full fan-out for the
    # aggregate-throughput sanity check.
    # ------------------------------------------------------------------
    print("=" * 72, flush=True)
    print(
        f"[inference_bench] starting benchmark: duration={bench['duration_seconds']}s "
        f"concurrency={bench['concurrency']} "
        f"in_tokens={bench['input_tokens']} out_tokens={bench['output_tokens']}",
        flush=True,
    )
    print("=" * 72, flush=True)

    t_bench = time.time()
    bench_results = await actor.run_benchmark.call(
        duration_s=float(bench["duration_seconds"]),
        concurrency=int(bench["concurrency"]),
        in_tokens=int(bench["input_tokens"]),
        out_tokens=int(bench["output_tokens"]),
        seed=int(bench["seed"]),
    )
    bench_elapsed = time.time() - t_bench

    payloads = _normalise_results(bench_results)

    # ------------------------------------------------------------------
    # Phase 3: aggregate + report.  Per-host line + grand totals.
    # ------------------------------------------------------------------
    print("=" * 72, flush=True)
    failures: list[str] = []
    total_tps = 0.0
    total_rps = 0.0
    total_requests = 0
    total_output_tokens = 0
    n_replicas = 0
    for payload in payloads:
        if not isinstance(payload, dict):
            failures.append(f"non-dict payload: {payload!r}")
            continue
        print(_format_per_host_line(payload), flush=True)
        total_tps += float(payload.get("throughput_tps", 0.0))
        total_rps += float(payload.get("throughput_rps", 0.0))
        total_requests += int(payload.get("requests", 0))
        total_output_tokens += int(payload.get("total_output_tokens", 0))
        n_replicas += 1

    print("=" * 72, flush=True)
    print(
        f"[inference_bench] aggregate throughput: {total_tps:.1f} tok/s "
        f"({total_rps:.2f} req/s, {total_requests} requests, "
        f"{total_output_tokens} output tokens) "
        f"across {n_replicas} replica(s) in {bench_elapsed:.1f}s wall",
        flush=True,
    )
    print("=" * 72, flush=True)

    # ------------------------------------------------------------------
    # Phase 4: shutdown.  Best-effort; we report failures but do not
    # fail the run on cleanup errors -- the metrics are already
    # collected and printed.
    # ------------------------------------------------------------------
    try:
        shutdown_results = await actor.shutdown.call()
        shutdown_payloads = _normalise_results(shutdown_results)
        for payload in shutdown_payloads:
            if isinstance(payload, dict) and not payload.get("ok", True):
                print(
                    f"[inference_bench] shutdown host={payload.get('host', '?')} "
                    f"reported ok=False (cleanup partial)",
                    file=sys.stderr,
                )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[inference_bench] shutdown raised {exc!r} (non-fatal)",
            file=sys.stderr,
        )

    if failures:
        for msg in failures:
            print(f"[inference_bench] FAIL: {msg}", file=sys.stderr)
        return 30
    if n_replicas == 0:
        print(
            "[inference_bench] FAIL: no replicas returned a metrics dict",
            file=sys.stderr,
        )
        return 30
    return 0


def _normalise_results(results: Any) -> list[Any]:
    """Flatten Monarch ``.call()`` return into a plain list of payloads.

    ``ActorMesh.<endpoint>.call()`` may return either
    an iterable of ``(point, payload)`` tuples (multi-rank) or a
    bare payload (single rank).  The titan_pretrain driver does the
    same dance -- factor out for clarity.
    """
    try:
        items = list(results)
    except TypeError:
        return [results]

    out: list[Any] = []
    for entry in items:
        if isinstance(entry, tuple) and len(entry) == 2:
            out.append(entry[1])
        else:
            out.append(entry)
    return out


def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    inference = _load_inference_section(args.algo_config)

    workers = [w.strip() for w in args.bare_metal_workers.split(",") if w.strip()]
    if not workers:
        print(
            "[inference_bench] --bare-metal-workers parsed empty",
            file=sys.stderr,
        )
        return 2

    # CLI ``--duration`` always wins over the YAML benchmark value.
    if args.duration is not None:
        inference["benchmark"]["duration_seconds"] = float(args.duration)

    try:
        return asyncio.run(_drive_benchmark(workers=workers, inference=inference))
    except KeyboardInterrupt:
        print("[inference_bench] interrupted -- tearing down", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(
            f"[inference_bench] unexpected error: {exc!r}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 20


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
