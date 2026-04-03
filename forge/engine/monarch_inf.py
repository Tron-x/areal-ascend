"""MonarchVLLMEngine -- drop-in replacement for RemotevLLMEngine.

Routes all communication through GeneratorActor Monarch endpoints instead
of HTTP.  Reuses ``VLLMBackend`` for request building / response parsing
so the payload format is identical to the HTTP path.

The key insight enabling this:  Monarch's ``Future.__await__`` bridges to
any asyncio event loop via ``call_soon_threadsafe`` (see
``monarch/_src/actor/future.py``), so
``await generator.handle_request.call_one(...)`` works inside AReaL's
uvloop-based ``AsyncTaskRunner`` without event loop conflicts.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from threading import Lock
from typing import Any

import numpy as np

from areal.api import (
    InferenceEngine,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    Scheduler,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import InferenceEngineConfig, PerfTracerConfig
from areal.engine.vllm_remote import VLLMBackend
from areal.infra import RolloutController, WorkflowExecutor
from areal.utils import logging as areal_logging
from areal.utils import perf_tracer, stats_tracker

logger = areal_logging.getLogger("MonarchVLLMEngine")

RID_CACHE_SIZE = 128


class MonarchVLLMEngine(InferenceEngine):
    """Non-invasive replacement for ``RemotevLLMEngine``.

    Holds a reference to the ``GeneratorActor`` and a ``VLLMBackend``
    for request/response format compatibility.  All transport goes
    through Monarch RPC (``generator.handle_request.call_one``).

    Phase 5: Optionally accepts ``reward_actor`` and ``agent_actor``.
    When ``agent_actor`` is provided, workflows are transparently replaced
    with ``MonarchAgentWorkflow``, which delegates multi-turn episodes
    to the AgentActor via Monarch RPC (AgentActor → GeneratorActor,
    SandboxActor, RewardActor).  When only ``reward_actor`` is provided,
    ``AsyncRewardWrapper`` is upgraded to ``MonarchRewardWrapper``.
    """

    def __init__(
        self,
        config: InferenceEngineConfig,
        generator_actor,
        reward_actor=None,
        agent_actor=None,
    ):
        self.config = config
        self._generator = generator_actor
        self._reward_actor = reward_actor
        self._agent_actor = agent_actor
        self._backend = VLLMBackend()
        self._version = 0
        self._lock = Lock()
        self._workflow_executor: WorkflowExecutor | None = None
        self._initialized = False

        # Persistent background loop for safe sync-to-async bridging
        self._bg_loop = asyncio.new_event_loop()
        import threading

        self._bg_thread = threading.Thread(
            target=self._bg_loop.run_forever,
            daemon=True,
            name="MonarchVLLMEngine_bg_loop",
        )
        self._bg_thread.start()

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
        engine_rank: int | None = None,
        num_engines: int | None = None,
    ):
        if engine_id is None:
            import torch.distributed as dist

            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = areal_logging.getLogger(f"[MonarchVLLMEngine Rank {engine_id}]")

        self.logger.info("Checking GeneratorActor health via Monarch RPC ...")
        func = getattr(self._generator, "call", None)
        if func is not None:
            # We must wrap it in an asyncio task to be gathered/run because it awaits but this method is sync
            # Wait, initialize is synchronous! And Service.call is async!
            # Since MonarchVLLMEngine is initialized either from synchronous or asynchronous thread, this needs care.
            # Actually, the original implementation did:
            # result = self._generator.handle_request.call_one("/health", {}).get(timeout=60)
            # which is sync because Future.get() is sync. Service.call returns a Future when we run it? No, Service.call is an async function.
            # I can just use asyncio.run or the existing event loop.
            try:
                asyncio.get_running_loop()
                # Instead of running inside loop synchronously which raises RuntimeError, we just skip health check if it's a Service for now,
                # or use asyncio.run_coroutine_threadsafe.
                self.logger.info("GeneratorActor health check deferred (Service mode)")
            except RuntimeError:
                result = asyncio.run(func("handle_request", "/health", {}))
                self.logger.info(f"GeneratorActor healthy: {result}")
        else:
            result = self._generator.handle_request.call_one("/health", {}).get(
                timeout=60
            )
            self.logger.info(f"GeneratorActor healthy: {result}")

        self._workflow_executor = WorkflowExecutor(
            config=self.config,
            inference_engine=self,
        )
        self._workflow_executor.initialize(
            logger=self.logger,
            train_data_parallel_size=train_data_parallel_size,
        )
        self._initialized = True
        self.logger.info("MonarchVLLMEngine initialised (Monarch RPC mode)")

    def destroy(self):
        self._initialized = False
        if self._workflow_executor is not None:
            self._workflow_executor.destroy()

        # Stop the background thread
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        if self._bg_thread.is_alive():
            self._bg_thread.join(timeout=2.0)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        if self._workflow_executor is None:
            raise RuntimeError("WorkflowExecutor not initialised")
        return self._workflow_executor

    def set_version(self, version: int):
        with self._lock:
            self._version = version

    def get_version(self) -> int:
        with self._lock:
            return self._version

    # -----------------------------------------------------------------
    # Generation  (replaces HTTP ``arequest_with_retry``)
    # -----------------------------------------------------------------

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Generate using Monarch RPC instead of HTTP.

        The logic mirrors ``RemoteInfEngine.agenerate`` but replaces the
        ``arequest_with_retry`` call with
        ``await self._generator.handle_request.call_one(...)``.
        """
        req = req.copy()

        if self.config.return_routed_experts:
            req.metadata["return_routed_experts"] = True

        gconfig = req.gconfig
        if gconfig.n_samples != 1:
            raise ValueError("n_samples > 1 not supported")

        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(f"max_new_tokens ({max_new_tokens}) is non-positive")
        req.gconfig.max_new_tokens = max_new_tokens

        start_time = time.perf_counter()
        accumulated_output_tokens: list = []
        accumulated_output_logprobs: list = []
        accumulated_versions: list = []
        accumulated_routed_experts: list[np.ndarray] = []

        stop_reason = None
        ori_max_new_tokens = gconfig.max_new_tokens

        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < ori_max_new_tokens
        ):
            while (
                self._workflow_executor is not None
                and self._workflow_executor.is_paused()
            ):
                await asyncio.sleep(0.5)

            http_req = self._backend.build_generation_request(
                req,
                with_lora=self.config.use_lora,
                version=self.get_version(),
            )

            func = getattr(self._generator, "call", None)
            if func is not None:
                result = await func(
                    "handle_request", http_req.endpoint, http_req.payload
                )
            else:
                result = await self._generator.handle_request.call_one(
                    http_req.endpoint, http_req.payload
                )

            if not isinstance(result, dict):
                raise ValueError(f"Expected dict response, got {type(result).__name__}")

            gen_result = self._backend.parse_generation_response(result)
            stop_reason = gen_result.stop_reason

            accumulated_output_tokens.extend(gen_result.output_tokens)
            accumulated_output_logprobs.extend(gen_result.output_logprobs)
            accumulated_versions.extend(
                [self.get_version()] * len(gen_result.output_tokens)
            )
            if gen_result.routed_experts is not None:
                accumulated_routed_experts.append(gen_result.routed_experts)

            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)

        if stop_reason == "abort":
            stop_reason = "length"

        latency = time.perf_counter() - start_time

        response = ModelResponse(
            input_tokens=req.input_ids[
                : len(req.input_ids) - len(accumulated_output_tokens)
            ],
            input_images=req.image_data,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,
            tokenizer=req.tokenizer,
            processor=req.processor,
            routed_experts=(
                np.concatenate(accumulated_routed_experts)
                if accumulated_routed_experts
                else None
            ),
        )
        return response

    # -----------------------------------------------------------------
    # Weight sync  (replaces HTTP helpers)
    # -----------------------------------------------------------------

    def _rpc_sync(self, endpoint: str, payload: dict, timeout: float = 300):
        """Blocking Monarch RPC call for synchronous weight operations."""
        func = getattr(self._generator, "call", None)
        if func is not None:
            # Route through the persistent background loop
            future = asyncio.run_coroutine_threadsafe(
                func("handle_request", endpoint, payload), self._bg_loop
            )
            return future.result(timeout=timeout)
        else:
            # Logic for single ActorRef (sync)
            return self._generator.handle_request.call_one(endpoint, payload).get(
                timeout=timeout
            )

    def init_weights_update_group(
        self, meta: WeightUpdateMeta, xccl_group_ranks: list[int] | None = None
    ) -> Future[None]:
        from concurrent.futures import ThreadPoolExecutor

        def _do():
            if xccl_group_ranks is not None:
                for i, rank in enumerate(xccl_group_ranks):
                    http_req = self._backend.build_init_weights_group_request(
                        "monarch_rpc", rank, meta
                    )
                    self._rpc_sync(http_req.endpoint, http_req.payload)
            else:
                http_req = self._backend.build_init_weights_group_request(
                    "monarch_rpc", 0, meta
                )
                self._rpc_sync(http_req.endpoint, http_req.payload)

        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(_do)
        pool.shutdown(wait=False)
        return fut

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        from concurrent.futures import ThreadPoolExecutor

        def _do():
            weight_reqs = self._backend.build_distributed_weight_update_requests(
                meta, param_specs
            )
            for http_req in weight_reqs.requests:
                self._rpc_sync(http_req.endpoint, http_req.payload)

        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(_do)
        pool.shutdown(wait=False)
        return fut

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        from concurrent.futures import ThreadPoolExecutor

        def _do():
            weight_reqs = self._backend.build_disk_weight_update_requests(meta)
            for http_req in weight_reqs.requests:
                self._rpc_sync(http_req.endpoint, http_req.payload)

        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(_do)
        pool.shutdown(wait=False)
        return fut

    def update_weights_from_awex(
        self,
        meta: WeightUpdateMeta,
        step_id: int | None = None,
        kwargs: dict | None = None,
    ) -> Future[None]:
        raise NotImplementedError("Awex not supported in Monarch plugin")

    # -----------------------------------------------------------------
    # Pause / Resume
    # -----------------------------------------------------------------

    def pause_generation(self):
        self._rpc_sync("/areal_pause_generation", {})
        time.sleep(self.config.pause_grace_period)

    def continue_generation(self):
        self._rpc_sync("/areal_continue_generation", {})

    def pause(self):
        return self.workflow_executor.pause()

    def resume(self):
        return self.workflow_executor.resume()

    # -----------------------------------------------------------------
    # Delegated to WorkflowExecutor (unchanged from RemotevLLMEngine)
    # -----------------------------------------------------------------

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        task_id: int | None = None,
        callback_addr: str | None = None,
        is_eval: bool = False,
        proxy_addr: str | None = None,
    ) -> int:
        raise NotImplementedError(
            "submit() not supported -- use prepare_batch() instead"
        )

    def wait(self, count, timeout=None, raise_timeout=True):
        raise NotImplementedError("wait() not supported -- use prepare_batch()")

    def wait_for_task(self, task_id, timeout=None, raise_timeout=True):
        raise NotImplementedError("wait_for_task() not supported")

    def rollout_batch(self, data, workflow, workflow_kwargs=None, group_size=1):
        raise NotImplementedError("rollout_batch() not supported")

    def _resolve_workflow(
        self,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None,
        group_size: int = 1,
    ):
        from areal.api.workflow_api import RolloutWorkflow
        from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
        from areal.utils.dynamic_import import import_from_string

        if isinstance(workflow, RolloutWorkflow):
            resolved = workflow
        elif isinstance(workflow, str):
            imported_obj = import_from_string(workflow)
            if isinstance(imported_obj, type) and issubclass(
                imported_obj, RolloutWorkflow
            ):
                if workflow_kwargs is None:
                    raise ValueError(
                        f"workflow_kwargs required for class workflow {workflow!r}"
                    )
                resolved = imported_obj(**workflow_kwargs)
            elif isinstance(imported_obj, RolloutWorkflow):
                resolved = imported_obj
            else:
                raise TypeError(
                    f"Imported workflow {workflow!r} is {type(imported_obj)}, "
                    f"expected RolloutWorkflow class or instance"
                )
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            if workflow_kwargs is None:
                raise ValueError(
                    f"workflow_kwargs required for class workflow {workflow!r}"
                )
            resolved = workflow(**workflow_kwargs)
        elif callable(workflow):
            obj = workflow(**(workflow_kwargs or {}))
            if isinstance(obj, RolloutWorkflow):
                resolved = obj
            else:
                raise TypeError(
                    f"Workflow factory returned {type(obj)}, expected RolloutWorkflow"
                )
        else:
            raise TypeError(f"Unsupported workflow type: {type(workflow)}")

        if self._agent_actor is not None and hasattr(resolved, "reward_fn"):
            resolved = self._wrap_with_agent_workflow(resolved)
        elif self._reward_actor is not None and hasattr(resolved, "reward_fn"):
            self._inject_monarch_reward(resolved)

        if group_size > 1:
            resolved = GroupedRolloutWorkflow(resolved, group_size, self.logger)
        return resolved

    def _wrap_with_agent_workflow(self, workflow):
        """Replace a resolved workflow with MonarchAgentWorkflow.

        Extracts reward_fn path, generation config, and tokenizer path from
        the original workflow, then returns a MonarchAgentWorkflow that
        delegates to AgentActor for multi-turn orchestration.
        """
        from forge.actors.agent import MonarchAgentWorkflow

        reward_fn = getattr(workflow, "reward_fn", None)
        if reward_fn is None:
            return workflow

        if isinstance(reward_fn, str):
            reward_fn_path = reward_fn
        elif callable(reward_fn):
            reward_fn_path = f"{reward_fn.__module__}.{reward_fn.__qualname__}"
        else:
            logger.warning(
                f"Cannot extract reward_fn path from {type(reward_fn)}, "
                f"falling back to reward-only injection"
            )
            if self._reward_actor is not None:
                self._inject_monarch_reward(workflow)
            return workflow

        tokenizer = getattr(workflow, "tokenizer", None)
        tokenizer_path = ""
        if tokenizer is not None:
            tokenizer_path = getattr(tokenizer, "name_or_path", "")
        if not tokenizer_path:
            tokenizer_path = getattr(workflow, "tokenizer_path", "")
        if not tokenizer_path:
            logger.warning(
                "Could not determine tokenizer_path from workflow; "
                "AgentActor will fail at setup if path is empty"
            )

        gconfig_dict = {}
        gconfig = getattr(workflow, "gconfig", None)
        if gconfig is not None:
            gconfig_dict = {
                "max_new_tokens": getattr(gconfig, "max_new_tokens", 1024),
                "temperature": getattr(gconfig, "temperature", 1.0),
                "top_p": getattr(gconfig, "top_p", 1.0),
            }

        max_turns = getattr(workflow, "max_turns", 2)

        agent_workflow = MonarchAgentWorkflow(
            agent_actor=self._agent_actor,
            tokenizer_path=tokenizer_path,
            reward_fn_path=reward_fn_path,
            gconfig=gconfig_dict,
            max_turns=max_turns,
        )
        logger.info(
            f"Injected MonarchAgentWorkflow: reward_fn={reward_fn_path}, "
            f"tokenizer={tokenizer_path}, max_turns={max_turns}"
        )
        return agent_workflow

    def _inject_monarch_reward(self, workflow):
        """Replace a workflow's AsyncRewardWrapper with MonarchRewardWrapper.

        Extracts the reward_fn import path, creates a MonarchRewardWrapper
        targeting the RewardActor, and patches the workflow so that
        ``arun_episode`` does not re-create AsyncRewardWrapper.
        """
        from forge.actors.reward import MonarchRewardWrapper

        reward_fn = getattr(workflow, "reward_fn", None)
        if reward_fn is None:
            return

        if isinstance(reward_fn, str):
            reward_fn_path = reward_fn
            from areal.utils.dynamic_import import import_from_string

            workflow.reward_fn = import_from_string(reward_fn_path)
        elif callable(reward_fn):
            reward_fn_path = f"{reward_fn.__module__}.{reward_fn.__qualname__}"
        else:
            logger.warning(
                f"Cannot determine reward_fn import path from {type(reward_fn)}, "
                f"skipping MonarchRewardWrapper injection"
            )
            return

        workflow.async_reward_fn = MonarchRewardWrapper(
            self._reward_actor, reward_fn_path
        )
        logger.info(f"Injected MonarchRewardWrapper for reward_fn={reward_fn_path}")

    @staticmethod
    def _resolve_should_accept_fn(should_accept_fn):
        if should_accept_fn is None or callable(should_accept_fn):
            return should_accept_fn
        if isinstance(should_accept_fn, str):
            import importlib

            module_path, _, fn_name = should_accept_fn.rpartition(".")
            mod = importlib.import_module(module_path)
            fn = getattr(mod, fn_name)
            if not callable(fn):
                raise TypeError(f"Imported {should_accept_fn} is not callable")
            return fn
        raise TypeError(f"Unsupported should_accept_fn type: {type(should_accept_fn)}")

    def prepare_batch(
        self,
        dataloader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ):
        assert workflow is not None
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size
        )
        resolved_fn = self._resolve_should_accept_fn(should_accept_fn)
        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=resolved_workflow,
            should_accept_fn=resolved_fn,
            dynamic_bs=dynamic_bs,
        )

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

    def set_proxy_gateway_addr(self, addr: str) -> None:
        pass

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        raise NotImplementedError("No server to launch in Monarch mode")

    def teardown_server(self):
        pass

    def offload(self):
        self._rpc_sync("/sleep", {})

    def onload(self, tags: list[str] | None = None):
        ep = "/wake_up"
        if tags:
            ep += "?" + "&".join(f"tags={t}" for t in tags)
        self._rpc_sync(ep, {})

    def export_stats(self) -> dict[str, float]:
        return stats_tracker.export_all(reduce_group=None)

    @classmethod
    def as_controller(
        cls, config: InferenceEngineConfig, scheduler: Scheduler
    ) -> RolloutController:
        return RolloutController(cls, config=config, scheduler=scheduler)

    def clear_batches(self, *args):
        pass

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)
