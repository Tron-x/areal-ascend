"""Generator actor: vLLM AsyncLLM with Monarch distributed execution.

Follows TorchForge's Generator pattern adapted for AReaL + Ascend NPU:
- Custom ``launch()`` for host mesh provisioning and GPU allocation
- ``setup()`` serializes host_mesh/registry to env vars for EngineCore subprocess
- ``generate()`` endpoint for text generation
- ``update_weights()`` endpoint for RL weight sync
"""

from __future__ import annotations

import base64
import logging
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
import torch
from monarch.actor import endpoint, this_host

from areal.monarch_plugin.actors.generator.worker import WorkerRegistry
from areal.monarch_plugin.controller.actor import AReaLForgeActor
from areal.monarch_plugin.controller.provisioner import _get_provisioner

logger = logging.getLogger(__name__)


@dataclass
class Generator(AReaLForgeActor):
    """vLLM-based text generator using AsyncLLM with Monarch execution.

    Uses AReaLMonarchExecutor for multi-GPU/NPU inference. The executor
    runs inside vLLM's EngineCore subprocess and communicates with
    WorkerWrapper actors via Monarch RPC.

    Args:
        engine_args: vLLM EngineArgs (dict or EngineArgs instance).
        sampling_params: Default SamplingParams (dict or SamplingParams).

    Example::

        gen = await Generator.options(
            procs=1, with_gpus=True, num_replicas=2
        ).as_service(
            engine_args={"model": "Qwen/Qwen2.5-1.5B", "tensor_parallel_size": 4},
            sampling_params={"max_tokens": 512, "temperature": 0.7},
        )
        completions = await gen.generate.route("What is 2+2?")
    """

    engine_args: Any = field(default=None)
    sampling_params: Any = field(default=None)

    def __post_init__(self):
        super().__init__()
        from vllm.engine.arg_utils import EngineArgs
        from vllm.entrypoints.llm import UsageContext
        from vllm.sampling_params import RequestOutputKind, SamplingParams

        self.llm = None
        self.generator_version: int = 0
        self.workers = None

        if self.engine_args is None:
            self.engine_args = EngineArgs()
        elif isinstance(self.engine_args, Mapping):
            self.engine_args = EngineArgs(**self.engine_args)
        self.vllm_config = self.engine_args.create_engine_config(UsageContext.LLM_CLASS)

        if self.sampling_params is None:
            self.sampling_params = SamplingParams()
        elif isinstance(self.sampling_params, Mapping):
            self.sampling_params = SamplingParams.from_optional(**self.sampling_params)
            self.sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

    @classmethod
    async def launch(cls: type[Generator], *args, **kwargs) -> Generator:
        """Custom launch: provision host mesh, allocate GPUs, spawn generator.

        Flow:
        1. Get host mesh (local or remote via Provisioner)
        2. Allocate exclusive GPU IDs from Provisioner
        3. Spawn CPU proc for Generator + WorkerRegistry
        4. Call setup(host_mesh, worker_registry, gpu_ids)
        """
        from vllm.engine.arg_utils import EngineArgs
        from vllm.entrypoints.llm import UsageContext

        engine_args = kwargs.get("engine_args", {})
        if isinstance(engine_args, Mapping):
            engine_args = EngineArgs(**engine_args)
        vllm_config = engine_args.create_engine_config(UsageContext.LLM_CLASS)

        num_gpus = vllm_config.parallel_config.world_size

        provisioner = await _get_provisioner()

        host_mesh = this_host()

        gpu_ids = await provisioner.allocate_gpu_ids(host_mesh, num_gpus)
        logger.info(f"[Generator.launch] Allocated GPUs: {gpu_ids}")

        if cls.hosts:
            singleton_slice = {k: slice(0, 1) for k in host_mesh.extent.keys()}
            head_host = host_mesh.slice(**singleton_slice)
        else:
            head_host = host_mesh

        generator_proc = head_host.spawn_procs(
            per_host={"procs": 1}, name="generator_proc"
        )

        worker_registry = generator_proc.spawn("worker_registry", WorkerRegistry)

        actor_name = kwargs.pop("name", cls.__name__)
        generator = generator_proc.spawn(actor_name, cls, *args, **kwargs)
        generator._generator_proc = generator_proc
        generator._worker_registry = worker_registry

        await generator.setup.call(host_mesh, worker_registry, gpu_ids)
        return generator

    @endpoint
    async def setup(self, host_mesh, worker_registry, gpu_ids: list[str]):
        """Initialize AsyncLLM with AReaLMonarchExecutor.

        Serializes host_mesh and worker_registry for the EngineCore subprocess,
        then creates AsyncLLM which spawns MonarchExecutor internally.
        """
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.executor.abstract import Executor

        num_gpus = self.vllm_config.parallel_config.tensor_parallel_size
        logger.info(f"Setting up AsyncLLM with {num_gpus} GPUs, allocated: {gpu_ids}")

        os.environ["VLLM_MONARCH_GPU_IDS"] = ",".join(gpu_ids)

        serialized_host = base64.b64encode(cloudpickle.dumps(host_mesh)).decode("utf-8")
        os.environ["VLLM_MONARCH_HOST_MESH"] = serialized_host

        serialized_registry = base64.b64encode(
            cloudpickle.dumps(worker_registry)
        ).decode("utf-8")
        os.environ["VLLM_MONARCH_WORKER_REGISTRY"] = serialized_registry

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        self.vllm_config.parallel_config.distributed_executor_backend = (
            "areal.monarch_plugin.actors.generator.executor.AReaLMonarchExecutor"
        )

        try:
            self.llm = AsyncLLM(
                vllm_config=self.vllm_config,
                executor_class=Executor.get_class(self.vllm_config),
                log_stats=True,
            )
            logger.info(f"AsyncLLM initialized with {num_gpus} workers")
        except Exception as e:
            logger.error(f"AsyncLLM initialization failed: {e}")
            raise

        self.workers = await worker_registry.get_workers.call_one()
        if self.workers is None:
            raise RuntimeError(
                "Workers not registered. MonarchExecutor may have failed."
            )
        logger.info(f"Retrieved workers from registry: {self.workers}")

    @endpoint
    async def generate(
        self,
        prompt: str,
        *,
        sampling_params=None,
    ) -> list[dict]:
        """Generate completions for a prompt.

        Args:
            prompt: Input text.
            sampling_params: Override default sampling params.

        Returns:
            List of completion dicts with text, token_ids, logprobs, etc.
        """
        from vllm.sampling_params import RequestOutputKind

        if self.llm is None:
            raise RuntimeError("Generator not initialized. Call setup() first.")

        params = sampling_params or self.sampling_params
        if params.output_kind is None:
            params.output_kind = RequestOutputKind.FINAL_ONLY

        request_output = None
        async for output in self.llm.generate(
            prompt=prompt,
            sampling_params=params,
            request_id=str(uuid.uuid4()),
        ):
            request_output = output

        return self._to_completion_dicts(request_output, prompt)

    @endpoint
    async def update_weights(self, version: int) -> None:
        """Update model weights on all workers.

        Pauses generation, loads weights, resumes generation.

        Args:
            version: Policy version to load.
        """
        if self.llm is None:
            raise RuntimeError("Generator not initialized.")

        logger.info(f"Starting weight update to v{version}")

        await self.llm.pause_generation(
            wait_for_inflight_requests=True, clear_cache=True
        )

        try:
            await self.workers.update_weights.call(version=version)
            self.generator_version = version
        finally:
            await self.llm.resume_generation()

        logger.info(f"Weight update complete, now v{version}")

    @endpoint
    async def stop(self):
        """Stop the generator and clean up local resources."""
        if self.llm is not None:
            self.llm.shutdown()
            self.llm = None

    @classmethod
    async def shutdown(cls, actor):
        """Shutdown generator and all resources."""
        try:
            await actor.stop.call()
        except Exception as e:
            logger.warning(f"Error during actor.stop: {e}")

        try:
            if getattr(actor, "_generator_proc", None):
                await actor._generator_proc.stop()
        except Exception as e:
            logger.warning(f"Error during generator_proc stop: {e}")

    def _extract_logprobs(self, output) -> torch.Tensor | None:
        if output.logprobs is not None:
            return torch.tensor(
                [
                    top_k_dict[token].logprob
                    for token, top_k_dict in zip(output.token_ids, output.logprobs)
                ]
            )
        return None

    def _to_completion_dicts(self, request_output, prompt: str) -> list[dict]:
        """Convert vLLM RequestOutput to serializable completion dicts."""
        completions = []
        for output in request_output.outputs:
            completions.append(
                {
                    "text": output.text,
                    "prompt": prompt,
                    "prompt_ids": list(request_output.prompt_token_ids or []),
                    "token_ids": list(output.token_ids)
                    if hasattr(output, "token_ids")
                    else [],
                    "logprobs": self._extract_logprobs(output),
                    "stop_reason": output.finish_reason,
                    "generator_version": self.generator_version,
                }
            )
        return completions
