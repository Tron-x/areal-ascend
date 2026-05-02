"""VLLMInferenceActor -- standalone TP=8 vLLM replica inside a Monarch actor.

**Does NOT conform to** :class:`forge.core.protocols.InferenceServerProtocol`.
This actor is a **benchmark actor** (``init`` / ``run_benchmark`` /
``shutdown``), not a rollout server.  It exists for synthetic-load
inference performance testing and has no FastAPI shim, no weight-sync
hook, no ``/generate`` endpoint hit by trainers over the network.

When the Pure Path MVP needs an inference server (vLLM-on-NPU rollout
for ``grpo_titan.py``), add a **sibling actor** that conforms to
:class:`InferenceServerProtocol` -- it will share weight-sync glue with
:class:`forge.actors.msswift_rollout.MsSwiftRolloutActor` via
:mod:`forge.engines.msswift.glue` (or a generalized version of it).

This is the inference counterpart of :class:`TitanTrainerActor`: a
single Monarch actor whose sole job is to wrap one TP=8 vLLM
``LLM`` instance and serve a self-contained synthetic-load benchmark.
No GRPO, no weight sync, no HTTP -- the smallest possible bridge
between Monarch's process mesh and a vLLM engine, mirroring the
B-full pretrain shape exactly.

Why one actor per host (not per NPU)
------------------------------------
``vllm.LLM(tensor_parallel_size=8, ...)`` internally manages an MP
executor that owns 8 NPU worker processes within a single driver
process.  Wrapping it in a per-NPU SPMDActor would mean 8 outer
processes each spawning their own 8 vLLM workers (64 total) all
fighting for the same 8 NPUs -- a non-starter.

So we deliberately use ``procs_per_host=1`` in the driver: each
host gets exactly one actor process which `vllm.LLM` then expands
into 8 internal workers.  The Monarch "world" size equals the host
count (2), not the NPU count (16).  This is the same shape vLLM
runs under any other launcher (Ray, plain Python script).

Why plain ``Actor`` (not ``SPMDActor``)
---------------------------------------
``SPMDActor`` populates ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` /
``MASTER_ADDR`` / ``MASTER_PORT`` on each process so torchelastic-
style ``env://`` ``init_process_group`` works.  vLLM does its own
internal distributed init (it picks its own master port, spawns its
own workers, sets its own ranks) so populating those env vars on
the outer process would be misleading at best and confusing at
worst (vLLM might pick them up via ``os.environ`` and break in
subtle ways).  Plain ``Actor`` keeps the contract minimal: one
endpoint call -> one Python method invocation, no env magic.

What the public surface is
--------------------------
Three endpoints, each idempotent / nullipotent so the driver can
safely retry on transient RPC errors:

* ``init(...)``       -- build the LLM (heavy: model load + KV-cache
                         allocation, takes ~10-30s).  Calling twice
                         on the same actor raises (we don't try to
                         tear down + rebuild, that's a different
                         lifecycle).
* ``run_benchmark(...)``  -- one full timed run.  Returns a metrics
                         dict.  Safe to call multiple times against
                         the same actor for back-to-back A/B runs.
* ``shutdown()``      -- best-effort engine teardown.  Always
                         returns; never raises (cleanup must not
                         mask earlier errors).

All endpoint args are kwargs-only so callers can't accidentally
swap argument order.
"""

from __future__ import annotations

import logging
import os
import random
import socket
import statistics
import time
from typing import Any

from monarch.actor import Actor, endpoint

logger = logging.getLogger(__name__)


class VLLMInferenceActor(Actor):
    """One TP=8 vLLM replica per actor instance, owning all NPUs of its host.

    Lifecycle::

        actor = proc_mesh.spawn("generator", VLLMInferenceActor)
        await actor.init.call_one(model="...", tp_size=8, ...)
        result = await actor.run_benchmark.call_one(duration_s=30, ...)
        await actor.shutdown.call_one()

    The actor is stateful between endpoint calls: ``init`` builds
    ``self._llm`` once, ``run_benchmark`` reads it, ``shutdown``
    tears it down.  This matches Monarch's contract that procs stay
    alive between endpoint invocations.
    """

    def __init__(self) -> None:
        super().__init__()
        # ``_llm`` is the heavyweight ``vllm.LLM`` instance once
        # ``init`` has run.  Kept None until then so calls in the
        # wrong order surface as a clear AttributeError-equivalent
        # rather than a deeply nested vLLM crash.
        self._llm: Any | None = None
        self._sampling_params: Any | None = None
        # Cached config so ``run_benchmark`` can include them in the
        # returned metrics dict (useful for downstream comparisons).
        self._model: str = ""
        self._tp_size: int = 0
        self._dtype: str = ""

    @endpoint
    def init(
        self,
        *,
        model: str,
        tp_size: int,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = True,
        seed: int = 1,
    ) -> dict[str, Any]:
        # Force vLLM's multiproc executor to use ``spawn`` rather than
        # the default ``fork``.  Forking from inside a Monarch actor
        # process inherits Monarch's signal handlers (SIGCHLD,
        # SIGINT) into the vLLM worker subprocesses, which then
        # interrupt c10d's TCPStore connect with ``EINTR`` mid-init
        # and the workers crash on ``WorkerProc.wait_for_ready``.
        # ``spawn`` re-execs Python so the workers start with a clean
        # signal mask.  Must be set BEFORE ``vllm.LLM`` is imported
        # because vLLM caches the start method at module load time.
        # See forge/actors/generator/generator.py for the long-term
        # alternative (``AReaLMonarchExecutor`` -- a custom executor
        # backend that integrates with Monarch instead of forking).
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        """Build the underlying ``vllm.LLM`` and warm up the executor.

        Heavy: loads the model weights, allocates KV-cache, runs a
        CUDA/NPU graph capture (skipped if ``enforce_eager``).  On a
        2-node cluster this takes ~15-30s for Qwen2.5-1.5B at TP=8.

        Args:
            model: HuggingFace model id or local path.  Resolved by
                vLLM via the ``HF_HOME`` cache or ``trust_remote_code``
                fetch -- in our pure-training profile we set
                ``HF_HUB_OFFLINE=1`` so the model MUST already be
                cached locally.  See ``forge/runtime/profiles.py``.
            tp_size: Tensor-parallel degree.  MUST equal the host's
                NPU count -- this actor owns every NPU exposed by
                the worker daemon, and vLLM will refuse to start if
                ``tp_size > visible_devices``.
            dtype: ``bfloat16`` (default) / ``float16`` / ``float32``.
                bf16 is the right choice for Qwen2/Llama3 on Ascend.
            max_model_len: KV-cache max token budget.  Smaller =
                more concurrent reqs fit in HBM.  Errors (silently
                truncates) if a request exceeds this.
            gpu_memory_utilization: Fraction of NPU HBM vLLM may
                use.  0.85 leaves ~15% for runtime/tokenizer; bump
                with care.
            enforce_eager: Skip CUDA-graph / NPU-graph capture.
                True is the safer default on NPU where graph capture
                support varies by CANN version.  Set False only
                after verifying it's supported on your CANN build.
            seed: vLLM RNG seed (affects sampling, not weight init).

        Returns:
            ``{"host": str, "model": str, "tp_size": int, "init_s":
            float}`` -- minimal acknowledgement for the driver to
            log per-host startup time.

        Raises:
            RuntimeError: if called twice without an intervening
                ``shutdown``.  We don't auto-tear-down because that
                masks lifecycle bugs in the driver.
        """
        if self._llm is not None:
            raise RuntimeError(
                "VLLMInferenceActor.init called twice without shutdown; "
                "this is almost certainly a driver bug.  Call shutdown() "
                "first to rebuild the engine."
            )

        # Lazy import so this module imports cleanly on a CPU-only
        # driver host (the smoke driver itself can run wherever, it
        # only orchestrates remote workers).
        from vllm import LLM, SamplingParams

        host = socket.gethostname()
        logger.info(
            "VLLMInferenceActor[%s] building LLM: model=%s tp=%d dtype=%s "
            "max_model_len=%d gpu_mem=%.2f enforce_eager=%s",
            host,
            model,
            tp_size,
            dtype,
            max_model_len,
            gpu_memory_utilization,
            enforce_eager,
        )

        t0 = time.time()
        self._llm = LLM(
            model=model,
            dtype=dtype,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            seed=seed,
            trust_remote_code=True,
        )
        init_s = time.time() - t0

        # Default sampling params -- ``run_benchmark`` overrides
        # ``max_tokens`` per call but reuses temperature/top_p/etc.
        # We pin temperature=1.0 for the synthetic-load benchmark so
        # decode work is deterministic-ish (no greedy short-circuit).
        self._sampling_params = SamplingParams(
            max_tokens=128,
            temperature=1.0,
            top_p=1.0,
        )

        self._model = model
        self._tp_size = tp_size
        self._dtype = dtype

        logger.info("VLLMInferenceActor[%s] LLM ready in %.1fs", host, init_s)
        return {
            "host": host,
            "model": model,
            "tp_size": tp_size,
            "init_s": init_s,
        }

    @endpoint
    def run_benchmark(
        self,
        *,
        duration_s: float,
        concurrency: int,
        in_tokens: int,
        out_tokens: int,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Drive synthetic load for ``duration_s`` seconds, return metrics.

        Algorithm (intentionally simple, matches what most public
        vLLM benchmarks do):

        1. Build a pool of ``concurrency`` distinct synthetic
           prompts of length ``in_tokens``.  Random token IDs in
           ``[1000, 30000)`` -- safely inside any modern tokenizer's
           vocab, avoids special tokens.
        2. Submit all ``concurrency`` prompts as one ``llm.generate``
           call.  vLLM's continuous-batching scheduler will keep
           the executor saturated.
        3. Repeat step 2 in a tight loop until wall-clock elapses
           ``duration_s``.  Track per-batch wall time + total
           generated tokens.
        4. Compute aggregate metrics from the per-batch samples.

        We measure WALL-CLOCK throughput (total generated tokens /
        total wall time, including queueing) rather than steady-state
        peak -- the former is what users actually feel.

        Args:
            duration_s: Lower bound on benchmark wall time.  The
                LAST batch is always run to completion even if it
                pushes total time over.  Use longer (>= 30s) to
                amortise the first-batch warmup.
            concurrency: Batch size per ``llm.generate`` call =
                in-flight request count.  vLLM's scheduler will
                handle preemption / KV-cache eviction internally
                if this exceeds available cache.
            in_tokens: Synthetic prompt length.  Pre-tokenised IDs
                are passed directly, bypassing the tokenizer for
                deterministic input length.
            out_tokens: Per-request ``max_tokens`` cap.  Generation
                may stop earlier on EOS; we report the actual mean
                output length as part of the metrics.
            seed: PRNG seed for the synthetic token IDs.

        Returns:
            Dict with::

                {"host": str,
                 "duration_s": float,        # actual elapsed
                 "batches": int,
                 "requests": int,
                 "total_output_tokens": int,
                 "throughput_tps": float,    # tokens / second
                 "throughput_rps": float,    # requests / second
                 "mean_output_tokens": float,
                 "batch_latency_ms": {"p50": ..., "p99": ...},
                 "model": str,
                 "tp_size": int,
                 "dtype": str}

        Raises:
            RuntimeError: if ``init`` has not been called yet.
        """
        if self._llm is None:
            raise RuntimeError("VLLMInferenceActor.run_benchmark called before init")

        # Lazy import of vLLM types -- same reasoning as in init.
        from vllm import SamplingParams, TokensPrompt

        host = socket.gethostname()

        # Build the synthetic prompt pool.  ``TokensPrompt`` lets us
        # bypass the tokenizer entirely so ``in_tokens`` is exact;
        # passing strings would drift in length depending on the
        # model's tokenizer behavior (the benchmark would no longer
        # be apples-to-apples across models).
        rng = random.Random(seed)
        prompt_pool: list[Any] = []
        for _ in range(concurrency):
            token_ids = [rng.randint(1000, 30000) for _ in range(in_tokens)]
            prompt_pool.append(TokensPrompt(prompt_token_ids=token_ids))

        sp = SamplingParams(
            max_tokens=out_tokens,
            temperature=1.0,
            top_p=1.0,
        )

        logger.info(
            "VLLMInferenceActor[%s] starting benchmark: duration=%.1fs "
            "concurrency=%d in_tokens=%d out_tokens=%d",
            host,
            duration_s,
            concurrency,
            in_tokens,
            out_tokens,
        )

        batches = 0
        requests = 0
        total_output_tokens = 0
        batch_latencies_ms: list[float] = []

        t_start = time.time()
        while True:
            elapsed_so_far = time.time() - t_start
            if elapsed_so_far >= duration_s:
                break

            t_batch = time.time()
            outputs = self._llm.generate(prompt_pool, sp, use_tqdm=False)
            batch_ms = (time.time() - t_batch) * 1000.0

            batches += 1
            batch_latencies_ms.append(batch_ms)
            for out in outputs:
                requests += 1
                # ``out.outputs`` is the list of completions for one
                # prompt; n=1 by default so we just take [0].
                if out.outputs:
                    total_output_tokens += len(out.outputs[0].token_ids)

        actual_duration = time.time() - t_start

        # Guard against zero-batch runs (duration_s smaller than
        # one batch's wall time) -- report what we have rather than
        # divide-by-zero.
        if batches == 0:
            logger.warning(
                "VLLMInferenceActor[%s] benchmark finished 0 batches in %.1fs",
                host,
                actual_duration,
            )
            return {
                "host": host,
                "duration_s": actual_duration,
                "batches": 0,
                "requests": 0,
                "total_output_tokens": 0,
                "throughput_tps": 0.0,
                "throughput_rps": 0.0,
                "mean_output_tokens": 0.0,
                "batch_latency_ms": {"p50": 0.0, "p99": 0.0},
                "model": self._model,
                "tp_size": self._tp_size,
                "dtype": self._dtype,
            }

        throughput_tps = total_output_tokens / actual_duration
        throughput_rps = requests / actual_duration
        mean_output_tokens = total_output_tokens / requests

        # Quantile via stdlib ``statistics.quantiles`` -- avoids
        # a numpy dep on the actor process.  n=100 -> 99 cut points;
        # we want p50 (idx 49) and p99 (idx 98).
        if len(batch_latencies_ms) >= 2:
            qs = statistics.quantiles(batch_latencies_ms, n=100, method="inclusive")
            p50 = qs[49]
            p99 = qs[98]
        else:
            p50 = batch_latencies_ms[0]
            p99 = batch_latencies_ms[0]

        result = {
            "host": host,
            "duration_s": actual_duration,
            "batches": batches,
            "requests": requests,
            "total_output_tokens": total_output_tokens,
            "throughput_tps": throughput_tps,
            "throughput_rps": throughput_rps,
            "mean_output_tokens": mean_output_tokens,
            "batch_latency_ms": {"p50": p50, "p99": p99},
            "model": self._model,
            "tp_size": self._tp_size,
            "dtype": self._dtype,
        }

        logger.info(
            "VLLMInferenceActor[%s] benchmark done: %d batches, %d req, "
            "%.1f tok/s, %.2f req/s, batch_p50=%.0fms",
            host,
            batches,
            requests,
            throughput_tps,
            throughput_rps,
            p50,
        )

        return result

    @endpoint
    def shutdown(self) -> dict[str, Any]:
        """Best-effort engine teardown.  Never raises.

        vLLM doesn't expose a public ``LLM.close()``, but dropping
        the reference + a GC pass is enough on Ascend (the worker
        subprocesses watch the parent and exit).  We also try to
        synchronise the NPU stream so any in-flight kernels drain
        before the proc is reused for another endpoint call.

        Returns:
            ``{"host": str, "ok": bool}``.  ``ok=False`` only on
            unexpected exceptions during cleanup -- the actor is
            still safe to discard either way.
        """
        host = socket.gethostname()
        ok = True
        try:
            if self._llm is not None:
                # vLLM's LLMEngine has executor.shutdown() in v1
                # which terminates the MP workers cleanly.  Reach
                # for it via the documented attribute path; if the
                # path drifts in a future vLLM, fall back to GC.
                try:
                    engine = getattr(self._llm, "llm_engine", None)
                    executor = getattr(engine, "model_executor", None)
                    if executor is not None and hasattr(executor, "shutdown"):
                        executor.shutdown()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "VLLMInferenceActor[%s] executor.shutdown raised %r "
                        "(will fall back to GC)",
                        host,
                        exc,
                    )

                self._llm = None

            try:
                import torch  # noqa: PLC0415

                if hasattr(torch, "npu"):
                    torch.npu.synchronize()
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("VLLMInferenceActor[%s] shutdown raised %r", host, exc)
            ok = False

        return {"host": host, "ok": ok, "pid": os.getpid()}
