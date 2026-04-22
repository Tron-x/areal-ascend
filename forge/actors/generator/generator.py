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
        from vllm.sampling_params import RequestOutputKind, SamplingParams

        self._inproc_engine = None
        self._paused = False
        self.generator_version: int = 0
        self.workers = None
        self.vllm_config = None

        if self.engine_args is None:
            self.engine_args = EngineArgs()
        elif isinstance(self.engine_args, Mapping):
            import inspect

            valid_params = set(inspect.signature(EngineArgs.__init__).parameters.keys())
            filtered = {k: v for k, v in self.engine_args.items() if k in valid_params}
            self.engine_args = EngineArgs(**filtered)

        if self.sampling_params is None:
            self.sampling_params = SamplingParams()
        elif isinstance(self.sampling_params, Mapping):
            self.sampling_params = SamplingParams.from_optional(**self.sampling_params)
            self.sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

    def _ensure_vllm_config(self):
        """Lazily create vllm_config (deferred from __init__ for remote actors)."""
        if self.vllm_config is None:
            from vllm.entrypoints.llm import UsageContext

            self.vllm_config = self.engine_args.create_engine_config(
                UsageContext.LLM_CLASS
            )
        return self.vllm_config

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
        # launch() runs on the driver (local) node, so create_engine_config is safe here
        mesh_name = cls.mesh_name or "generator"

        provisioner = await _get_provisioner()

        if cls.hosts:
            host_mesh = await provisioner.get_host_mesh(mesh_name)
        else:
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

        if cls.hosts and provisioner.launcher:
            await provisioner.launcher.remote_setup(generator_proc)

        worker_registry = generator_proc.spawn("worker_registry", WorkerRegistry)

        actor_name = kwargs.pop("name", cls.__name__)
        generator = generator_proc.spawn(actor_name, cls, *args, **kwargs)
        generator._generator_proc = generator_proc
        generator._worker_registry = worker_registry
        # Expose the underlying host_mesh for callers (e.g. grpo.py's
        # torchstore weight-sync strategy) that need to spawn sibling
        # procs (storage volume, etc.) on the same generator host.
        # Use custom names that don't collide with ``ProcMesh._host_mesh``
        # / ``_head_host`` internals.
        generator._forge_gen_host_mesh = host_mesh
        generator._forge_gen_head_host = head_host

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

        self._ensure_vllm_config()
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
    async def get_worker_mesh(self):
        """Expose the internal vLLM WorkerWrapper ActorMesh to the caller.

        Used by weight-sync backends that need to fan out endpoint calls
        to every TP worker (e.g. CollectiveBroadcastBackend calling
        ``init_bcast_group`` / ``recv_and_load_flat``). WorkerRegistry
        already demonstrates that Monarch ActorMesh references are
        cross-process serializable, so returning this handle is safe.
        """
        if self.workers is None:
            raise RuntimeError("get_worker_mesh called before setup finished")
        return self.workers

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
    async def get_chat_template(self) -> dict:
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
    async def update_weights_sync(
        self, version: int, method: str = "nccl", payload: dict | None = None
    ) -> dict:
        """Unified weight update endpoint called by WeightSyncStrategy.

        Dispatches to the appropriate internal method based on ``method``.

        Args:
            version: Policy version after this update.
            method: One of ``"nccl"``, ``"checkpoint"``, ``"hixl"``.
            payload: Extra parameters (e.g. ``model_path`` for checkpoint).

        Returns:
            Status dict with ``success`` and ``message``.
        """
        payload = payload or {}

        if method == "nccl":
            result = await self._update_weights_xccl()
        elif method == "checkpoint":
            if "model_path" not in payload:
                return {
                    "success": False,
                    "message": "model_path required for checkpoint sync",
                }
            result = await self._update_weights_disk(payload)
        elif method == "hixl":
            result = await self._update_weights_hixl(payload)
        elif method == "torchstore":
            result = await self._update_weights_torchstore(version, payload)
        else:
            return {"success": False, "message": f"Unknown sync method: {method}"}

        if result.get("success"):
            self.generator_version = version
        return {**result, "version": version}

    async def _update_weights_hixl(self, payload: dict) -> dict:
        """HIXL one-sided weight update (stub for future implementation)."""
        self._inproc_engine.pause_generation()
        try:
            result = await self._collective_rpc("update_weight_hixl", payload)
        except Exception as e:
            self._inproc_engine.resume_generation()
            return {"success": False, "message": f"HIXL update failed: {e}"}
        self._inproc_engine.resume_generation()
        return result

    async def _update_weights_torchstore(self, version: int, payload: dict) -> dict:
        """Torchstore / Monarch RDMA weight update -- delegates to workers.

        The fast path: each vLLM worker runs ``pull_weights`` inside its
        own proc and writes the ``ts.get`` result straight into its NPU
        model parameters (HiXL RDMA inplace).  No CPU detour, no
        cloudpickle ``state_dict`` RPC payload through this Generator proc.

        ``payload`` must carry either:

        * ``items``: list of ``{"name", "key"}`` dicts (new ``WeightSyncService``
          / ``MultiVolTorchstoreBackend`` path), or
        * legacy ``param_names`` / ``param_shapes`` / ``param_dtypes``
          triplet (pre-Service callers).  We auto-translate that into the
          ``items`` form so the worker endpoint has a single contract.
        """
        import time

        # ----- Flat-buffer fast path ----------------------------------
        # If the backend sent ``flat_key`` + ``flat_plan`` we delegate
        # straight to WorkerWrapper.pull_weights_flat: one ``ts.get``
        # into an aligned flat buffer, then local unpack.  This
        # bypasses the 311-per-key RDMA overhead of the per-parameter
        # path below.
        flat_key = payload.get("flat_key")
        if flat_key is not None:
            return await self._update_weights_torchstore_flat(version, payload)

        items = payload.get("items")
        total_bytes_hint = int(payload.get("total_bytes") or 0)

        if items is None:
            # Legacy payload shape -- translate for the worker endpoint.
            from forge.engines.weight_sync.torchstore_sync import get_param_key

            names = payload.get("param_names") or []
            if not names:
                return {
                    "success": False,
                    "message": (
                        "torchstore sync: payload has neither "
                        "'items' nor 'param_names'; cannot proceed."
                    ),
                }
            items = [{"name": n, "key": get_param_key(version, n)} for n in names]

        self._inproc_engine.pause_generation()
        pull_t0 = time.perf_counter()
        try:
            pull_mesh = await self.workers.pull_weights.call(
                version=version, items=items
            )
            pull_s = time.perf_counter() - pull_t0
        except Exception as e:
            self._inproc_engine.resume_generation()
            logger.exception("torchstore sync failed")
            return {
                "success": False,
                "message": f"torchstore sync failed: {e}",
                "bytes": total_bytes_hint,
            }
        self._inproc_engine.resume_generation()

        # ``pull_mesh`` is a mesh-result dict; all workers return the same
        # aggregate shape.  Summarise using max() rather than sum() because
        # every worker pulls the same bytes in TP=1 (each loads its own copy
        # of the full state_dict); wire-bytes == per-worker-bytes, not
        # sum-across-workers.
        bytes_total = 0
        workers_load_s = 0.0
        num_keys = 0
        direct_count = 0
        fused_count = 0
        success_all = True
        for ref, payload in pull_mesh.items():
            if not isinstance(payload, dict):
                continue
            if not payload.get("success", False):
                success_all = False
            bytes_total = max(bytes_total, payload.get("bytes", 0))
            workers_load_s = max(workers_load_s, payload.get("workers_load_s", 0.0))
            num_keys = max(num_keys, payload.get("num_keys", 0))
            direct_count = max(direct_count, payload.get("direct_count", 0))
            fused_count = max(fused_count, payload.get("fused_count", 0))

        if not success_all:
            return {
                "success": False,
                "message": "one or more workers failed pull_weights",
                "bytes": bytes_total,
                "num_keys": num_keys,
                "pull_s": pull_s,
                "workers_load_s": workers_load_s,
            }

        logger.info(
            "torchstore sync v%d: %d keys (%d direct + %d fused), "
            "%.2f GB, total=%.2fs (pull+load combined in-worker, "
            "workers_load=%.2fs)",
            version,
            num_keys,
            direct_count,
            fused_count,
            bytes_total / (1024**3),
            pull_s,
            workers_load_s,
        )
        return {
            "success": True,
            "message": (
                f"torchstore sync: {direct_count} inplace + "
                f"{fused_count} fused-fallback"
            ),
            "bytes": bytes_total,
            "num_keys": num_keys,
            "pull_s": pull_s,
            "workers_load_s": workers_load_s,
        }

    async def _update_weights_torchstore_flat(
        self, version: int, payload: dict
    ) -> dict:
        """Flat-buffer variant of ``_update_weights_torchstore``.

        Delegates to ``WorkerWrapper.pull_weights_flat`` which does a
        single ``ts.get`` into an aligned NPU flat buffer and then
        unpacks views into model params.  Avoids the per-key pool
        staging overhead that dominates the per-parameter path.
        """
        import time

        flat_key = payload["flat_key"]
        plan = payload.get("flat_plan") or []
        total_bytes = int(payload.get("total_bytes") or 0)

        if not plan or total_bytes <= 0:
            # Guard against a bug upstream dropping rank-0's plan.
            # Returning success=False used to hide this and let the
            # service look fine; now we raise so the driver-side
            # sync loop notices and the operator sees the real cause.
            raise ValueError(
                f"_update_weights_torchstore_flat(v{version}): payload "
                f"missing plan (plan_len={len(plan)}) or total_bytes "
                f"({total_bytes}); the backend's ``_push_flat`` likely "
                "failed to collect rank-0's publish result."
            )

        shard_ranges = payload.get("shard_ranges") or []
        shard_key_fmt = payload.get("shard_key_fmt") or ""

        self._inproc_engine.pause_generation()
        pull_t0 = time.perf_counter()
        try:
            pull_mesh = await self.workers.pull_weights_flat.call(
                version=version,
                key=flat_key,
                plan=plan,
                total_bytes=total_bytes,
                shard_ranges=shard_ranges,
                shard_key_fmt=shard_key_fmt,
            )
            pull_s = time.perf_counter() - pull_t0
        except Exception as e:
            self._inproc_engine.resume_generation()
            logger.exception("torchstore flat sync failed")
            return {
                "success": False,
                "message": f"torchstore flat sync failed: {e}",
                "bytes": total_bytes,
            }
        self._inproc_engine.resume_generation()

        bytes_total = 0
        workers_load_s = 0.0
        num_keys = 0
        direct_count = 0
        fused_count = 0
        alloc_s = rdma_s = unpack_s = fused_load_s = 0.0
        success_all = True
        for ref, payload_r in pull_mesh.items():
            if not isinstance(payload_r, dict):
                continue
            if not payload_r.get("success", False):
                success_all = False
            bytes_total = max(bytes_total, payload_r.get("bytes", 0))
            workers_load_s = max(workers_load_s, payload_r.get("workers_load_s", 0.0))
            num_keys = max(num_keys, payload_r.get("num_keys", 0))
            direct_count = max(direct_count, payload_r.get("direct_count", 0))
            fused_count = max(fused_count, payload_r.get("fused_count", 0))
            alloc_s = max(alloc_s, payload_r.get("alloc_s", 0.0))
            rdma_s = max(rdma_s, payload_r.get("rdma_s", 0.0))
            unpack_s = max(unpack_s, payload_r.get("unpack_s", 0.0))
            fused_load_s = max(fused_load_s, payload_r.get("fused_load_s", 0.0))

        if not success_all:
            return {
                "success": False,
                "message": "one or more workers failed pull_weights_flat",
                "bytes": bytes_total,
                "num_keys": num_keys,
                "pull_s": pull_s,
                "workers_load_s": workers_load_s,
            }

        logger.info(
            "torchstore flat sync v%d: %d keys (%d direct + %d fused), "
            "%.2f GB, total=%.2fs [alloc=%.2fs rdma=%.2fs unpack=%.2fs "
            "fused_load=%.2fs]",
            version,
            num_keys,
            direct_count,
            fused_count,
            bytes_total / (1024**3),
            pull_s,
            alloc_s,
            rdma_s,
            unpack_s,
            fused_load_s,
        )
        return {
            "success": True,
            "message": (
                f"torchstore flat: {direct_count} inplace + {fused_count} fused"
            ),
            "bytes": bytes_total,
            "num_keys": num_keys,
            "pull_s": pull_s,
            "workers_load_s": workers_load_s,
        }

    @endpoint
    async def handle_request(self, ep: str, payload: dict) -> dict:
        """Dispatch a request based on the endpoint path.

        Compatible with MonarchVLLMEngine's RPC interface.
        Supports both new ``/forge/*`` routes and legacy ``/areal_*`` aliases.
        """
        _route_map = {
            "/v1/completions": self._generate_completion,
            "/v1/chat/completions": self._generate_completion,
            "/forge/generation/pause": self._pause_generation,
            "/forge/generation/resume": self._continue_generation,
            "/forge/weights/init_group": self._init_weights_update_group,
            "/forge/weights/set_meta": self._set_weight_meta,
            "/forge/weights/set_meta_lora": self._set_weight_meta_lora,
            "/forge/weights/update_nccl": self._update_weights_xccl,
            "/forge/weights/update_nccl_lora": self._update_weights_lora_xccl,
            "/forge/weights/update_checkpoint": self._update_weights_disk,
            # Legacy aliases (backward compat)
            "/areal_pause_generation": self._pause_generation,
            "/areal_continue_generation": self._continue_generation,
            "/areal_init_weights_update_group": self._init_weights_update_group,
            "/areal_set_update_weight_meta": self._set_weight_meta,
            "/areal_set_update_weight_meta_lora": self._set_weight_meta_lora,
            "/areal_update_weights_xccl": self._update_weights_xccl,
            "/areal_update_weights_lora_xccl": self._update_weights_lora_xccl,
            "/areal_update_weights": self._update_weights_disk,
        }

        handler = _route_map.get(ep)
        if handler is not None:
            if ep in (
                "/forge/weights/update_nccl",
                "/areal_update_weights_xccl",
                "/forge/weights/update_nccl_lora",
                "/areal_update_weights_lora_xccl",
                "/forge/generation/pause",
                "/areal_pause_generation",
                "/forge/generation/resume",
                "/areal_continue_generation",
            ):
                return await handler()
            return await handler(payload)
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
        result = await self.workers.execute_method.call(method, *args)
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
