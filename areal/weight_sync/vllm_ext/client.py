"""Trainer-side HTTP client for AReaL-style XCCL weight sync.

This is the framework-neutral counterpart to
``areal.weight_sync.vllm_ext.server_router`` (the FastAPI router that ships
with our forked vLLM server) and
``areal.weight_sync.vllm_ext.worker_extension.VLLMWorkerExtension`` (the
per-worker class that actually receives the broadcast).

The complete contract a *trainer* must follow is:

1. Once at startup, on trainer rank 0::

       client = WeightSyncClient("http://<server>:<port>",
                                 group_name="my_ws")
       group = client.init_communicator(
           master_addr=<this-host>,
           master_port=<free-port>,
           vllm_world_size=<TP * PP * DP of the server>,
           backend="hccl",  # or "nccl"/"xccl"; ``None`` auto-detects
       )

   Internally this fires ``POST /areal_init_weights_update_group`` and
   in parallel performs ``init_custom_process_group(rank=0,
   world_size=N+1, ...)`` so the trainer joins the cross-mesh group.

2. Each training step, on trainer rank 0::

       client.push_weights_xccl(model.named_parameters(),
                                chunk_mb=512)

   This pauses generation, buckets parameters by ``chunk_mb``, sends
   ``set_meta`` + ``update_weights_xccl`` for each bucket, drives
   ``dist.broadcast(...)`` on the cross-mesh group, then resumes.

3. On shutdown::

       client.close()

The intended consumers are *external* trainers (ms-swift /
trl.GRPOTrainer / your own) that don't want to import AReaL's
``RolloutController`` + Monarch RPC layer just to talk to a vLLM
server.  Forge users should keep using
``forge/engines/weight_sync/nccl_sync.py`` (Monarch-actor based) and
``forge/engines/weight_sync/backends/areal_xccl.py`` (Forge protocol
binding).

Multi-server note: this client deliberately manages exactly *one*
server / one process group.  For multi-replica vLLM rollouts,
instantiate one ``WeightSyncClient`` per server with a unique
``group_name`` and call ``push_weights_xccl`` on each in turn.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import requests
import torch
import torch.distributed as dist

from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.weight_sync.distributed import init_custom_process_group

logger = logging.getLogger("WeightSyncClient")


_TORCH_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def _dtype_to_str(dtype: torch.dtype) -> str:
    return _TORCH_DTYPE_TO_STR.get(dtype, str(dtype).split(".")[-1])


def _auto_backend() -> str:
    """Pick a torch.distributed backend matching the local accelerator.

    Order of preference: NPU (HCCL) > XPU (XCCL) > CUDA (NCCL) > Gloo.
    Mirrors ``areal.infra.platforms.current_platform.communication_backend``
    but does not require importing the platform registry, so this client
    can be dropped into environments where AReaL's platform module isn't
    fully wired (e.g. ms-swift's CLI entrypoint).
    """
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "hccl"
    except ImportError:
        pass

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xccl"

    if torch.cuda.is_available():
        return "nccl"

    return "gloo"


class WeightSyncError(RuntimeError):
    """Raised when the HTTP / collective handshake with the vLLM server fails."""


class WeightSyncClient:
    """Trainer-side HTTP + collective client for AReaL-style XCCL weight sync."""

    def __init__(
        self,
        server_url: str,
        *,
        group_name: str = "ws_default",
        request_timeout: float = 600.0,
        session: requests.Session | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.group_name = group_name
        self.request_timeout = request_timeout
        self._session = session or requests.Session()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ws-http")
        self._group: dist.ProcessGroup | None = None
        self._world_size: int | None = None
        self._backend: str | None = None

    # ------------------------------------------------------------------ utils

    def _post(self, endpoint: str, payload: dict[str, Any] | None = None) -> dict:
        url = f"{self.server_url}{endpoint}"
        try:
            response = self._session.post(
                url, json=payload or {}, timeout=self.request_timeout
            )
        except requests.RequestException as e:
            raise WeightSyncError(f"HTTP error calling {url}: {e}") from e

        try:
            body = response.json()
        except ValueError as e:
            raise WeightSyncError(
                f"Non-JSON response from {url} (status={response.status_code}): "
                f"{response.text[:512]}"
            ) from e

        if response.status_code != 200 or not body.get("success", False):
            raise WeightSyncError(
                f"Server returned failure for {endpoint} "
                f"(status={response.status_code}): {body}"
            )
        return body

    def _post_async(
        self, endpoint: str, payload: dict[str, Any] | None = None
    ) -> Future:
        return self._executor.submit(self._post, endpoint, payload)

    # ------------------------------------------------------------------ init

    def init_communicator(
        self,
        *,
        master_addr: str,
        master_port: int,
        vllm_world_size: int,
        backend: str | None = None,
        timeout: timedelta | None = None,
    ) -> dist.ProcessGroup:
        """Build the cross-mesh process group.

        The trainer rank that calls this method becomes rank 0 of the
        group; the vLLM workers occupy ranks ``1 .. vllm_world_size``.

        Args:
            master_addr: A hostname/IP reachable from every vLLM worker.
                Typically the trainer rank-0 host.
            master_port: A free TCP port on ``master_addr`` for the
                rendezvous TCPStore.
            vllm_world_size: Number of vLLM worker ranks to wait for
                (``TP * PP * DP`` of the target server).
            backend: torch.distributed backend.  Defaults to auto-detect:
                ``hccl`` on NPU, ``xccl`` on XPU, ``nccl`` on CUDA,
                ``gloo`` otherwise.
            timeout: Process-group rendezvous timeout.  Defaults to
                :data:`areal.utils.constants.DIST_GROUP_DEFAULT_TIMEOUT`.

        Returns:
            The local-side ``dist.ProcessGroup``; cached so subsequent
            ``push_weights_xccl`` calls reuse it.
        """
        if self._group is not None:
            raise WeightSyncError(
                f"Communicator already initialized for group_name={self.group_name!r}"
            )

        backend = backend or _auto_backend()
        timeout = timeout or DIST_GROUP_DEFAULT_TIMEOUT
        world_size = vllm_world_size + 1  # +1 for this trainer rank

        init_payload = {
            "master_address": master_addr,
            "master_port": str(master_port),
            "rank_offset": 1,  # vLLM workers occupy ranks 1..N
            "world_size": world_size,
            "backend": backend,
            "group_name": self.group_name,
        }

        logger.info(
            "init_communicator: server=%s group=%s backend=%s world=%d "
            "init_method=tcp://%s:%d",
            self.server_url,
            self.group_name,
            backend,
            world_size,
            master_addr,
            master_port,
        )

        # Server-side workers will block in init_custom_process_group until
        # this trainer joins as rank 0; fire HTTP request *first* (async),
        # then enter our own rendezvous.
        fut = self._post_async("/areal_init_weights_update_group", init_payload)

        self._group = init_custom_process_group(
            backend=backend,
            world_size=world_size,
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=0,
            group_name=self.group_name,
            timeout=timeout,
        )

        # Surface server-side failures (e.g. one TP rank crashed).
        fut.result(timeout=timeout.total_seconds())

        self._world_size = world_size
        self._backend = backend
        logger.info("init_communicator complete: group=%s", self.group_name)
        return self._group

    # ------------------------------------------------------------------ push

    def push_weights_xccl(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        *,
        chunk_mb: int = 512,
        pause_resume: bool = True,
    ) -> None:
        """Stream parameters to vLLM workers via collective broadcast.

        Args:
            named_tensors: Iterable of ``(name, tensor)`` pairs.  Tensors
                must already live on the local accelerator (HCCL/NCCL
                cannot broadcast CPU tensors); ``FSDP2`` users typically
                pre-call ``DTensor.full_tensor()`` to materialise full
                tensors on rank 0.
            chunk_mb: Approximate per-bucket size cap in MiB.  Buckets
                amortise the HTTP round-trip cost across many parameters.
            pause_resume: When True, calls
                ``/areal_pause_generation`` before the broadcast and
                ``/areal_continue_generation`` after.  Set to False if
                the caller already paused (e.g. nested updates).
        """
        if self._group is None:
            raise WeightSyncError(
                "init_communicator must be called before push_weights_xccl"
            )

        if pause_resume:
            self.pause_generation()

        try:
            chunk_bytes = chunk_mb * 1024 * 1024
            bucket: list[tuple[str, torch.Tensor]] = []
            bucket_bytes = 0

            for name, tensor in named_tensors:
                tensor_bytes = tensor.numel() * tensor.element_size()
                if bucket and bucket_bytes + tensor_bytes > chunk_bytes:
                    self._broadcast_bucket(bucket)
                    bucket = []
                    bucket_bytes = 0
                bucket.append((name, tensor))
                bucket_bytes += tensor_bytes

            if bucket:
                self._broadcast_bucket(bucket)
        finally:
            if pause_resume:
                self.continue_generation()

    def _broadcast_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        assert self._group is not None
        meta_payload = {
            "names": [name for name, _ in bucket],
            "dtypes": [_dtype_to_str(t.dtype) for _, t in bucket],
            "shapes": [list(t.shape) for _, t in bucket],
            "group_name": self.group_name,
        }

        # Order matters: ``update_weight_xccl`` on each worker reads the
        # per-worker attributes that ``set_weight_meta`` writes, so meta
        # must be acknowledged before we trigger the broadcast.  Once meta
        # is in, fire ``update_weights_xccl`` asynchronously so the worker
        # enters ``dist.broadcast`` while we drive the same call locally.
        self._post("/areal_set_update_weight_meta", meta_payload)
        update_fut = self._post_async("/areal_update_weights_xccl")

        for _, tensor in bucket:
            dist.broadcast(tensor, src=0, group=self._group, async_op=False)

        update_fut.result(timeout=self.request_timeout)

    # ------------------------------------------------------------------ lora

    def push_lora_adapter_xccl(
        self,
        peft_config: dict,
        named_lora_params: list[tuple[str, torch.Tensor]],
        *,
        lora_name: str,
        lora_int_id: int,
        base_model_name: str,
        pause_resume: bool = True,
    ) -> None:
        """Stream a LoRA adapter's ``(A, B)`` matrices to vLLM workers via XCCL.

        This is the LoRA-aware counterpart to :meth:`push_weights_xccl`.
        Unlike full-base broadcast, only the LoRA delta (typically <2% of
        full base size) traverses the wire, and vLLM applies it as a
        live adapter via ``LoRAModel.from_lora_tensors`` rather than a
        permanent merge into the base weights.

        Required in ``peft_config`` (LoraConfig serialized to dict; see
        :func:`peft.config.PeftConfigMixin.to_dict` or simply
        :func:`dataclasses.asdict` on a ``LoraConfig``):

        * ``r`` (int) -- LoRA rank
        * ``lora_alpha`` (int) -- LoRA scaling factor
        * ``target_modules`` (list[str] | str) -- which linear layers got
          adapted (e.g. ``["q_proj", "v_proj"]`` or ``"all-linear"``)
        * ``bias`` (str) -- one of ``"none"`` / ``"all"`` / ``"lora_only"``

        Args:
            peft_config: Serialized LoRA config (sets coerced to lists).
            named_lora_params: Iterable of ``(name, tensor)`` pairs for the
                LoRA matrices.  Names must follow PEFT convention so the
                worker-side ``LoRAModel.from_lora_tensors`` can map them
                back onto target modules
                (e.g. ``"base_model.model.layers.0.self_attn.q_proj.lora_A.weight"``).
                Tensors must already live on the local accelerator.
            lora_name: Logical name of the adapter (must match what
                vLLM's ``add_lora`` will be / was called with).  ms-swift
                uses the constant ``"swift_lora"``.
            lora_int_id: Integer id vLLM uses internally to address the
                adapter (ms-swift uses ``111``).  Same id is reused on
                every push so the adapter slot is recycled in place.
            base_model_name: HF id / path of the base model the adapter
                was trained against.  Stored on the worker to validate
                future incremental pushes.
            pause_resume: When True, brackets the push with
                ``/areal_pause_generation`` + ``/areal_continue_generation``.
                Set False if the caller already paused (nested updates).

        Raises:
            WeightSyncError: If the HTTP handshake or XCCL broadcast
                handshake fails on either side.
        """
        if self._group is None:
            raise WeightSyncError(
                "init_communicator must be called before push_lora_adapter_xccl"
            )

        bucket = list(named_lora_params)
        if not bucket:
            logger.warning("push_lora_adapter_xccl: empty LoRA params, skipping")
            return

        target_modules = peft_config.get("target_modules", [])
        if isinstance(target_modules, set):
            target_modules = list(target_modules)

        meta_payload = {
            "names": [name for name, _ in bucket],
            "dtypes": [_dtype_to_str(t.dtype) for _, t in bucket],
            "shapes": [list(t.shape) for _, t in bucket],
            "group_name": self.group_name,
            "lora_name": lora_name,
            "lora_int_id": lora_int_id,
            "lora_target_modules": target_modules,
            "lora_rank": int(peft_config.get("r", peft_config.get("lora_rank", 8))),
            "lora_alpha": int(peft_config.get("lora_alpha", 32)),
            "lora_bias": str(peft_config.get("bias", "none")),
            "base_model_name": base_model_name,
        }

        if pause_resume:
            self.pause_generation()

        try:
            self._post("/areal_set_update_weight_meta_lora", meta_payload)
            update_fut = self._post_async("/areal_update_weights_lora_xccl")

            for _, tensor in bucket:
                dist.broadcast(tensor, src=0, group=self._group, async_op=False)

            update_fut.result(timeout=self.request_timeout)
            logger.info(
                "push_lora_adapter_xccl OK: lora_name=%s int_id=%d params=%d",
                lora_name,
                lora_int_id,
                len(bucket),
            )
        finally:
            if pause_resume:
                self.continue_generation()

    # ------------------------------------------------------------------ disk

    def push_weights_from_disk(self, model_path: str) -> None:
        """Reload weights on every vLLM worker from a HuggingFace dir."""
        self._post("/areal_update_weights", {"model_path": str(model_path)})

    # ------------------------------------------------------------ pause/resume

    def pause_generation(self) -> None:
        self._post("/areal_pause_generation")

    def continue_generation(self) -> None:
        self._post("/areal_continue_generation")

    # ------------------------------------------------------------------ close

    def close(self) -> None:
        """Tear down the local process group and HTTP resources.

        We don't tear down the *server-side* group here -- the server
        survives across trainer restarts and reuses the group by
        ``group_name``.  Server admins should restart vLLM if they need
        to drop stale groups.
        """
        if self._group is not None:
            try:
                dist.destroy_process_group(self._group)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "destroy_process_group(%s) raised: %s", self.group_name, e
                )
            self._group = None

        self._executor.shutdown(wait=False)
        try:
            self._session.close()
        except Exception:
            pass

    # ------------------------------------------------------------ context mgr

    def __enter__(self) -> WeightSyncClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------- helpers


def iter_named_tensors_for_broadcast(
    model: torch.nn.Module,
    *,
    materialize_full: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` pairs ready to feed into push_weights_xccl.

    On FSDP2-sharded models, ``param.data`` is a DTensor; the broadcast
    needs the *full* unsharded tensor on rank 0.  When
    ``materialize_full=True`` and a parameter is a DTensor, this helper
    calls ``.full_tensor()``.  Caller is responsible for ensuring this
    is invoked on rank 0 only (or that the materialised tensor is the
    same on all trainer ranks if you intentionally don't gate by rank).
    """
    for name, param in model.named_parameters():
        tensor = param.data
        if materialize_full and hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        yield name, tensor


def slice_flattened_lora_tensor(
    flattened: torch.Tensor,
    metadatas: list[dict],
) -> list[tuple[str, torch.Tensor]]:
    """Slice a byte-offset flattened tensor bucket back into per-tensor pairs.

    Generic helper for the "pack many small tensors into one contiguous
    buffer + metadata sidecar" pattern that originated in SGLang's
    ``weight_sync/tensor_bucket.py`` and is now reused by several RL
    frameworks (TRL, ms-swift, etc.) to amortise the per-tensor
    broadcast overhead on small adapters like LoRA.

    Our XCCL worker broadcasts and applies tensors *individually* (one
    ``dist.broadcast`` per parameter), so callers who receive a
    flattened bucket must un-flatten before invoking
    :meth:`WeightSyncClient.push_lora_adapter_xccl` -- this helper is
    that un-flatten step.

    Schema contract (each ``metadata`` dict):

    * ``name``     -- target parameter name (string).
    * ``shape``    -- target tensor shape (sequence of int).
    * ``dtype``    -- target dtype as string; both
      ``"bfloat16"`` and ``"torch.bfloat16"`` accepted.
    * ``start_idx`` / ``end_idx`` -- byte offsets into the **uint8
      view** of ``flattened``.  Convention: the producer flattens via
      ``tensor.view(torch.uint8).reshape(-1)`` so both endpoints are
      byte-addressed, dtype-agnostic.

    No framework-specific imports -- the function only ever sees the
    primitive dict shape above.  Add a per-framework adapter at the
    glue layer if your producer uses different keys.

    Returns:
        List of ``(name, tensor)`` pairs where each tensor is a
        zero-copy *view* into ``flattened`` (mutating the view mutates
        the source buffer).  Safe to feed directly to
        ``dist.broadcast`` because views share device + dtype.
    """
    flat_u8 = flattened.view(torch.uint8).reshape(-1)
    out: list[tuple[str, torch.Tensor]] = []
    for meta in metadatas:
        name = meta["name"]
        shape = tuple(meta["shape"])
        dtype_str = str(meta["dtype"]).removeprefix("torch.")
        target_dtype = getattr(torch, dtype_str)
        start = int(meta["start_idx"])
        end = int(meta["end_idx"])
        view = flat_u8[start:end].view(target_dtype).reshape(shape)
        out.append((name, view))
    return out


__all__ = [
    "WeightSyncClient",
    "WeightSyncError",
    "iter_named_tensors_for_broadcast",
    "slice_flattened_lora_tensor",
]
