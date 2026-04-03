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

from forge.actors.base import ForgeActor
from forge.actors.generator.worker import WorkerRegistry
from forge.provisioner import _get_provisioner

logger = logging.getLogger(__name__)


@dataclass
class Generator(ForgeActor):
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

        self._inproc_engine = None
        self._paused = False
        self.generator_version: int = 0
        self.workers = None

        if self.engine_args is None:
            self.engine_args = EngineArgs()
        elif isinstance(self.engine_args, Mapping):
            import inspect

            valid_params = set(inspect.signature(EngineArgs.__init__).parameters.keys())
            filtered = {k: v for k, v in self.engine_args.items() if k in valid_params}
            self.engine_args = EngineArgs(**filtered)
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
            import inspect

            valid_params = set(inspect.signature(EngineArgs.__init__).parameters.keys())
            filtered = {k: v for k, v in engine_args.items() if k in valid_params}
            engine_args = EngineArgs(**filtered)
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
        """Initialize in-process LLMEngine with AReaLMonarchExecutor.

        Uses in-process LLMEngine (not subprocess-based AsyncLLM) for
        better Ascend NPU compatibility.
        """
        import asyncio

        from vllm.v1.executor.abstract import Executor

        from forge.service.vllm_engine import MonarchVLLMEngine as InprocEngine

        num_gpus = self.vllm_config.parallel_config.tensor_parallel_size
        logger.info(
            f"Setting up in-process LLMEngine with {num_gpus} GPUs, "
            f"allocated: {gpu_ids}"
        )

        os.environ["VLLM_MONARCH_GPU_IDS"] = ",".join(gpu_ids)

        serialized_host = base64.b64encode(cloudpickle.dumps(host_mesh)).decode("utf-8")
        os.environ["VLLM_MONARCH_HOST_MESH"] = serialized_host

        serialized_registry = base64.b64encode(
            cloudpickle.dumps(worker_registry)
        ).decode("utf-8")
        os.environ["VLLM_MONARCH_WORKER_REGISTRY"] = serialized_registry

        self.vllm_config.parallel_config.distributed_executor_backend = (
            "forge.actors.generator.executor.AReaLMonarchExecutor"
        )

        try:
            executor_class = Executor.get_class(self.vllm_config)
            self._inproc_engine = InprocEngine(self.vllm_config, executor_class)
            loop = asyncio.get_running_loop()
            self._inproc_engine.start(loop)
            logger.info(f"In-process LLMEngine initialized with {num_gpus} workers")
        except Exception as e:
            logger.error(f"LLMEngine initialization failed: {e}")
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

        if self._inproc_engine is None:
            raise RuntimeError("Generator not initialized. Call setup() first.")

        params = sampling_params or self.sampling_params
        if params.output_kind is None:
            params.output_kind = RequestOutputKind.FINAL_ONLY

        request_output = None
        async for output in self._inproc_engine.generate(
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
        if self._inproc_engine is None:
            raise RuntimeError("Generator not initialized.")

        logger.info(f"Starting weight update to v{version}")

        self._inproc_engine.pause_generation()

        try:
            await self.workers.update_weights.call(version=version)
            self.generator_version = version
        finally:
            self._inproc_engine.resume_generation()

        logger.info(f"Weight update complete, now v{version}")

    @endpoint
    def get_chat_template(self) -> dict:
        """Return the model's chat template metadata.

        Useful for other actors (e.g. AgentActor) that need to discover
        the correct prompt format at runtime without loading the tokenizer
        themselves.

        Returns:
            Dict with ``model`` (model path) and ``chat_template``
            (Jinja2 template string, or None if the tokenizer has none).
        """
        model_path = getattr(self.engine_args, "model", "")
        template_str = None
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            template_str = tokenizer.chat_template
        except Exception as e:
            logger.warning(f"Could not load chat template for {model_path}: {e}")
        return {"model": model_path, "chat_template": template_str}

    @endpoint
    async def handle_request(self, ep: str, payload: dict) -> dict:
        """Dispatch a request based on the endpoint path.

        Compatible with MonarchVLLMEngine's RPC interface.
        """
        if ep in ("/v1/completions", "/v1/chat/completions"):
            return await self._generate_completion(payload)
        if ep == "/areal_pause_generation":
            return await self._pause_generation()
        if ep == "/areal_continue_generation":
            return await self._continue_generation()
        if ep == "/areal_init_weights_update_group":
            return await self._init_weights_update_group(payload)
        if ep == "/areal_set_update_weight_meta":
            return await self._set_weight_meta(payload)
        if ep == "/areal_set_update_weight_meta_lora":
            return await self._set_weight_meta_lora(payload)
        if ep == "/areal_update_weights_xccl":
            return await self._update_weights_xccl()
        if ep == "/areal_update_weights_lora_xccl":
            return await self._update_weights_lora_xccl()
        if ep == "/areal_update_weights":
            return await self._update_weights_disk(payload)
        if ep == "/health":
            return {"status": "ok"}
        raise ValueError(f"Unknown endpoint: {ep}")

    async def _generate_completion(self, payload: dict) -> dict:
        import asyncio

        from vllm.sampling_params import SamplingParams

        while getattr(self, "_paused", False):
            await asyncio.sleep(0.1)

        prompt = payload.get("prompt")
        messages = payload.get("messages")

        params = SamplingParams(
            top_p=payload.get("top_p", 1.0),
            top_k=payload.get("top_k", -1),
            max_tokens=payload.get("max_tokens", 16),
            temperature=payload.get("temperature", 1.0),
            stop_token_ids=payload.get("stop_token_ids"),
            ignore_eos=payload.get("ignore_eos", False),
            skip_special_tokens=payload.get("skip_special_tokens", True),
            logprobs=1,
        )

        request_id = str(uuid.uuid4())

        if prompt is not None:
            gen_prompt = (
                {"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt
            )
        elif messages is not None:
            gen_prompt = {"messages": messages}
        else:
            raise ValueError("payload must contain 'prompt' or 'messages'")

        request_output = None
        async for output in self._inproc_engine.generate(
            prompt=gen_prompt, sampling_params=params, request_id=request_id
        ):
            request_output = output

        if request_output is None:
            return {
                "choices": [
                    {
                        "finish_reason": "abort",
                        "logprobs": {"tokens": [], "token_logprobs": []},
                        "text": "",
                    }
                ]
            }

        comp_output = request_output.outputs[0]
        tokens = [f"token:{tid}" for tid in comp_output.token_ids]
        logprobs_list = []
        if comp_output.logprobs is not None:
            for tid, lp_dict in zip(comp_output.token_ids, comp_output.logprobs):
                logprobs_list.append(lp_dict[tid].logprob if tid in lp_dict else 0.0)
        else:
            logprobs_list = [0.0] * len(comp_output.token_ids)

        return {
            "choices": [
                {
                    "finish_reason": comp_output.finish_reason or "stop",
                    "text": comp_output.text,
                    "logprobs": {
                        "tokens": tokens,
                        "token_logprobs": logprobs_list,
                    },
                }
            ]
        }

    async def _pause_generation(self) -> dict:
        self._paused = True
        self._inproc_engine.pause_generation()
        return {"success": True, "message": "Generation paused"}

    async def _continue_generation(self) -> dict:
        self._inproc_engine.resume_generation()
        self._paused = False
        return {"success": True, "message": "Generation resumed"}

    async def _collective_rpc(self, method: str, *args):
        import asyncio

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.workers.execute_method.call(method, *args).get(timeout=300),
        )
        ret_list = [v for _, v in result.items()]
        success = True
        message = ""
        for rank, ret_value in enumerate(ret_list):
            ok, msg = ret_value
            success = success and ok
            message += f"TP rank: {rank} {'success' if ok else 'failed: ' + msg}\n"
        return {"success": success, "message": message}

    async def _init_weights_update_group(self, payload: dict) -> dict:
        return await self._collective_rpc(
            "init_update_weight_group",
            payload["master_address"],
            payload["master_port"],
            payload["rank_offset"],
            payload["world_size"],
            payload["backend"],
            payload["group_name"],
        )

    async def _set_weight_meta(self, payload: dict) -> dict:
        return await self._collective_rpc(
            "set_weight_meta",
            payload["names"],
            payload["dtypes"],
            payload["shapes"],
            payload["group_name"],
        )

    async def _set_weight_meta_lora(self, payload: dict) -> dict:
        return await self._collective_rpc(
            "set_weight_meta_lora",
            payload["names"],
            payload["dtypes"],
            payload["shapes"],
            payload["group_name"],
            payload["lora_name"],
            payload["lora_int_id"],
            payload["lora_target_modules"],
            payload["lora_rank"],
            payload["lora_alpha"],
            payload["lora_bias"],
            payload["base_model_name"],
        )

    async def _update_weights_xccl(self) -> dict:
        self._inproc_engine.pause_generation()
        result = await self._collective_rpc("update_weight_xccl")
        self._inproc_engine.resume_generation()
        return result

    async def _update_weights_lora_xccl(self) -> dict:
        self._inproc_engine.pause_generation()
        result = await self._collective_rpc("update_weight_lora_xccl")
        self._inproc_engine.resume_generation()
        return result

    async def _update_weights_disk(self, payload: dict) -> dict:
        return await self._collective_rpc("update_weights", payload["model_path"])

    @endpoint
    async def shutdown_engine(self):
        """Stop the generator and clean up local resources."""
        if self._inproc_engine is not None:
            self._inproc_engine.shutdown()
            self._inproc_engine = None

    @classmethod
    async def shutdown(cls, actor):
        """Shutdown generator and all resources."""
        try:
            await actor.shutdown_engine.call()
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
