"""GeneratorActor -- Monarch actor wrapping vLLM's LLMEngine (in-process).

Uses MonarchVLLMEngine (which wraps LLMEngine with InprocClient) instead of
AsyncLLM to avoid the EngineCore subprocess.  This ensures collective_rpc
Monarch RPCs execute in the actor's own process where the transport is
properly initialised, eliminating the deadlock that occurred when the
EngineCore ran as a non-Monarch subprocess.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from typing import Any, Optional

import cloudpickle
from monarch.actor import endpoint

from areal.api.cli_args import vLLMConfig
from areal.monarch_plugin.actor import AReaLMonarchActor

logger = logging.getLogger(__name__)


class GeneratorActor(AReaLMonarchActor):
    """Monarch actor embedding vLLM for in-process inference.

    Lifecycle
    ---------
    __init__   -> store vLLM CLI arg list
    setup()    -> serialise HostMesh, create MonarchVLLMEngine + workers
    handle_request(endpoint, payload) -> dispatch (generation / weight-sync)
    shutdown() -> cleanup
    """

    procs = 1
    with_gpus = True

    @staticmethod
    def build_vllm_cli_args(config, alloc_mode) -> list[str]:
        """Build the CLI arg list for vLLM."""
        args_dict = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=alloc_mode.gen.tp_size,
            pp_size=alloc_mode.gen.pp_size,
        )
        cli: list[str] = []
        for k, v in args_dict.items():
            if v is None or v is False or v == "" or (isinstance(v, list) and not v):
                continue
            flag = f"--{k.replace('_', '-')}"
            if v is True:
                cli.append(flag)
            elif isinstance(v, list):
                cli.append(flag)
                cli.extend(str(x) for x in v)
            else:
                cli.append(flag)
                cli.append(str(v))
        return cli

    def __init__(self, vllm_cli_args: list):
        self._cli_args = vllm_cli_args
        self.engine = None
        self.workers = None
        self._paused = False

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    @endpoint
    async def setup(self, host_mesh, worker_registry, device_ids: list):
        """Initialise MonarchVLLMEngine with AReaLMonarchExecutor.

        Parameters
        ----------
        host_mesh : HostMesh
            From ``this_host()`` in the orchestrator.
        worker_registry : WorkerRegistry actor mesh
            Bridge for MonarchExecutor to register workers.
        device_ids : list[str] | None
            NPU device IDs to allocate (e.g. ``["0"]``). If None,
            discovers from ``ASCEND_RT_VISIBLE_DEVICES``.
        """
        from vllm.engine.arg_utils import EngineArgs
        from vllm.entrypoints.llm import UsageContext
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        logger.info(f"[GeneratorActor] setup: requested device_ids={device_ids}")
        logger.info(f"[GeneratorActor] vLLM CLI args: {self._cli_args}")

        if device_ids is None:
            raw_ids = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")
            device_ids = [d.strip() for d in raw_ids.split(",") if d.strip()]
            logger.info(f"[GeneratorActor] Auto-discovered device_ids: {device_ids}")

        os.environ["VLLM_MONARCH_GPU_IDS"] = ",".join(str(d) for d in device_ids)

        serialized_hm = base64.b64encode(cloudpickle.dumps(host_mesh)).decode()
        os.environ["VLLM_MONARCH_HOST_MESH"] = serialized_hm

        serialized_reg = base64.b64encode(cloudpickle.dumps(worker_registry)).decode()
        os.environ["VLLM_MONARCH_WORKER_REGISTRY"] = serialized_reg

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        from vllm.entrypoints.openai.cli_args import make_arg_parser

        parser = FlexibleArgumentParser(description="vLLM for Monarch")
        parser = make_arg_parser(parser)
        parsed = parser.parse_args(self._cli_args)
        engine_args = EngineArgs.from_cli_args(parsed)
        vllm_config = engine_args.create_engine_config(UsageContext.LLM_CLASS)

        from areal.monarch_plugin.executor import AReaLMonarchExecutor

        vllm_config.parallel_config.distributed_executor_backend = (
            "areal.monarch_plugin.executor.AReaLMonarchExecutor"
        )
        vllm_config.scheduler_config.async_scheduling = False

        from areal.monarch_plugin.service.vllm_engine import MonarchVLLMEngine

        logger.info("[GeneratorActor] Creating MonarchVLLMEngine (in-process) ...")
        self.engine = MonarchVLLMEngine(
            vllm_config=vllm_config,
            executor_class=AReaLMonarchExecutor,
        )
        loop = asyncio.get_running_loop()
        self.engine.start(loop)
        logger.info("[GeneratorActor] MonarchVLLMEngine started")

        self.workers = await worker_registry.get_workers.call_one()
        if self.workers is None:
            raise RuntimeError("Workers not found in registry")
        logger.info(f"[GeneratorActor] Workers retrieved: {self.workers}")

    # -----------------------------------------------------------------
    # Universal request handler (mirrors areal_vllm_server.py routes)
    # -----------------------------------------------------------------

    @endpoint
    async def handle_request(self, ep: str, payload: dict) -> dict:
        """Dispatch a request based on the endpoint path."""
        if ep == "/v1/completions":
            return await self._generate_completion(payload)
        if ep == "/v1/chat/completions":
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

    # -----------------------------------------------------------------
    # Generation (/v1/completions)
    # -----------------------------------------------------------------

    async def _generate_completion(self, payload: dict) -> dict:
        """Run vLLM generation and return an OpenAI-compatible response dict."""
        from vllm.sampling_params import SamplingParams

        if self._paused:
            while self._paused:
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
            gen_prompt = {"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt
        elif messages is not None:
            gen_prompt = {"messages": messages}
        else:
            raise ValueError("payload must contain 'prompt' or 'messages'")

        request_output = None
        async for output in self.engine.generate(
            prompt=gen_prompt,
            sampling_params=params,
            request_id=request_id,
        ):
            request_output = output

        if request_output is None:
            return self._build_abort_response()

        return self._request_output_to_openai(request_output, prompt)

    @staticmethod
    def _build_abort_response() -> dict:
        return {
            "choices": [{
                "finish_reason": "abort",
                "logprobs": {"tokens": [], "token_logprobs": []},
                "text": "",
            }]
        }

    @staticmethod
    def _request_output_to_openai(request_output, prompt) -> dict:
        """Convert vLLM RequestOutput to AReaL-expected OpenAI dict."""
        comp_output = request_output.outputs[0]

        tokens = [f"token:{tid}" for tid in comp_output.token_ids]

        logprobs_list = []
        if comp_output.logprobs is not None:
            for tid, lp_dict in zip(comp_output.token_ids, comp_output.logprobs):
                if tid in lp_dict:
                    logprobs_list.append(lp_dict[tid].logprob)
                else:
                    logprobs_list.append(0.0)
        else:
            logprobs_list = [0.0] * len(comp_output.token_ids)

        return {
            "choices": [{
                "finish_reason": comp_output.finish_reason or "stop",
                "text": comp_output.text,
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": logprobs_list,
                },
            }]
        }

    # -----------------------------------------------------------------
    # Pause / Resume
    # -----------------------------------------------------------------

    async def _pause_generation(self) -> dict:
        self._paused = True
        self.engine.pause_generation()
        return {"success": True, "message": "Generation paused"}

    async def _continue_generation(self) -> dict:
        self.engine.resume_generation()
        self._paused = False
        return {"success": True, "message": "Generation resumed"}

    # -----------------------------------------------------------------
    # Weight sync -- forwarded to workers via collective_rpc
    # -----------------------------------------------------------------

    async def _collective_rpc(self, method: str, *args):
        """Call a method on all workers (offloaded to thread to avoid blocking)."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.workers.execute_method.call(method, *args).get(timeout=300),
        )
        ret_list = [v for _, v in result.items()]
        return self._build_response(ret_list)

    @staticmethod
    def _build_response(ret_list) -> dict:
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
        self.engine.pause_generation()
        result = await self._collective_rpc("update_weight_xccl")
        self.engine.resume_generation()
        return result

    async def _update_weights_lora_xccl(self) -> dict:
        self.engine.pause_generation()
        result = await self._collective_rpc("update_weight_lora_xccl")
        self.engine.resume_generation()
        return result

    async def _update_weights_disk(self, payload: dict) -> dict:
        return await self._collective_rpc("update_weights", payload["model_path"])

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------

    @endpoint
    async def shutdown(self) -> None:
        logger.info("[GeneratorActor] Shutting down ...")
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
        logger.info("[GeneratorActor] Shutdown complete")
