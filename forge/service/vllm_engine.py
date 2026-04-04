"""In-process vLLM engine wrapper for Monarch actors.

Replaces AsyncLLM + AsyncMPClient (which spawns EngineCore as a subprocess
communicating via ZMQ) with an in-process LLMEngine (InprocClient).  This
eliminates the subprocess and ensures that collective_rpc Monarch RPCs run
from the actor's own process where the transport is properly initialised.

A background thread drives the step loop so the Monarch event loop stays
responsive for incoming endpoint calls.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_SENTINEL = object()


class MonarchVLLMEngine:
    """Thin async wrapper around vLLM's synchronous LLMEngine.

    Lifecycle:
        engine = MonarchVLLMEngine(vllm_config, executor_class)
        engine.start(asyncio.get_running_loop())
        async for output in engine.generate(prompt, params, request_id):
            ...
        engine.shutdown()
    """

    def __init__(self, vllm_config, executor_class):
        from vllm.v1.engine.llm_engine import LLMEngine

        logger.info("[MonarchVLLMEngine] Creating in-process LLMEngine ...")
        self._engine = LLMEngine(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,
            multiprocess_mode=False,
        )
        logger.info("[MonarchVLLMEngine] LLMEngine created")

        self._request_queue: queue.Queue[tuple[str, Any, Any]] = queue.Queue()
        self._output_queues: dict[str, asyncio.Queue] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._step_thread: threading.Thread | None = None
        self._shutdown_flag = False
        self._paused = False

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._step_thread = threading.Thread(
            target=self._step_loop, daemon=True, name="MonarchVLLMEngine-step"
        )
        self._step_thread.start()
        logger.info("[MonarchVLLMEngine] Step thread started")

    def _step_loop(self) -> None:
        """Background thread: drain new requests, step engine, dispatch outputs."""
        while not self._shutdown_flag:
            try:
                self._drain_request_queue()

                if self._paused:
                    time.sleep(0.01)
                    continue

                if not self._engine.has_unfinished_requests():
                    time.sleep(0.001)
                    continue

                request_outputs = self._engine.step()

                for output in request_outputs:
                    with self._lock:
                        q = self._output_queues.get(output.request_id)
                    if q is not None and self._loop is not None:
                        asyncio.run_coroutine_threadsafe(q.put(output), self._loop)
            except Exception:
                logger.exception("[MonarchVLLMEngine] Error in step loop")
                time.sleep(0.1)

    def _drain_request_queue(self) -> None:
        while True:
            try:
                action, request_id, payload = self._request_queue.get_nowait()
            except queue.Empty:
                break
            if action == "add":
                self._engine.add_request(
                    request_id=request_id,
                    prompt=payload[0],
                    params=payload[1],
                )
            elif action == "abort":
                self._engine.abort_request(request_id)

    async def generate(self, prompt, sampling_params, request_id: str):
        """Async generator yielding RequestOutput objects."""
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._output_queues[request_id] = q

        try:
            self._request_queue.put(("add", request_id, (prompt, sampling_params)))

            while True:
                output = await q.get()
                yield output
                if output.finished:
                    break
        finally:
            with self._lock:
                self._output_queues.pop(request_id, None)

    def pause_generation(self) -> None:
        self._paused = True

    def resume_generation(self) -> None:
        self._paused = False

    @property
    def model_executor(self):
        return self._engine.model_executor

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return self._engine.collective_rpc(method, timeout, args, kwargs)

    def shutdown(self) -> None:
        logger.info("[MonarchVLLMEngine] Shutting down ...")
        self._shutdown_flag = True
        if self._step_thread is not None:
            self._step_thread.join(timeout=10)
        if hasattr(self._engine, "shutdown"):
            try:
                self._engine.shutdown()
            except Exception:
                logger.exception("[MonarchVLLMEngine] Error during engine shutdown")
        del self._engine
        logger.info("[MonarchVLLMEngine] Shutdown complete")
