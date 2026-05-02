"""ms-swift <-> ``areal.weight_sync`` glue (Phase A: actor-driven).

Three installer functions, each idempotent and safe to call multiple
times.  All three are normally invoked indirectly by the Forge actors
(:mod:`forge.actors.msswift_trainer`, :mod:`forge.actors.msswift_rollout`)
rather than from a user script.

* :func:`install_server_patches` -- runs in the ms-swift rollout actor
  proc.  Wires custom ``/areal_*`` HTTP routes onto ``SwiftRolloutDeploy``
  that dispatch into vLLM workers via ms-swift's existing multiprocessing
  pipes, AND swaps in a ``_start_data_parallel_workers`` wrapper so each
  spawned vLLM child re-applies :func:`install_worker_patches` (required
  because ms-swift uses ``multiprocessing.set_start_method('spawn')``,
  which does not inherit driver-side monkey-patches).

* :func:`install_worker_patches` -- runs in **every** vLLM worker child
  process.  Forces ``worker_extension_cls`` to AReaL's class so the worker
  exposes ``init_update_weight_group`` / ``set_weight_meta`` /
  ``update_weight_xccl`` collective RPCs, and patches the
  ``vllm_ascend`` top_k_top_p kernel to its PyTorch fallback (the custom
  NPU kernel is broken in vllm-ascend 0.14.0rc1).

* :func:`install_client_patches` -- runs in the ms-swift trainer actor
  proc, before ``swift.pipelines.train.rlhf.rlhf_main`` is invoked.
  Replaces ``trl.VLLMClient.{init,update_named_param,reset_prefix_cache,
  close}_communicator`` with ``areal.weight_sync.vllm_ext.client.WeightSyncClient``
  delegations so the trainer pushes weights via collective HCCL rather
  than ms-swift's bundled ``PyHcclCommunicator`` (which is single-host
  only on NPU).

Why a glue module instead of forking ms-swift / vLLM
----------------------------------------------------
Per the framework first principles (``.cursor/rules/framework-first-principles.mdc``):
共性下沉 -- ``areal/weight_sync/`` is the shared collective backend, not
duplicated in each adapter.  ms-swift just needs three thin shims to
plug into it; those shims live here and only here.

Provenance: the original PoC version of this module lived under
``/tmp/swift_areal_smoke/swift_areal_glue.py``.  This file is a verbatim
move into the canonical Forge layout (``forge/engines/<backend>/``) so
actor / app code can import it without leaning on ``sys.path`` hacks.
"""

from __future__ import annotations

import logging
import os
import socket

# Wire-protocol schemas come from the L2 source of truth so that
# WeightSyncClient (trainer side), server_router (in-process areal
# vLLM server), and our glue shim (ms-swift's SwiftRolloutDeploy
# FastAPI app) can never desync on field names / types.  See
# ``areal/weight_sync/vllm_ext/protocol.py`` for the rationale.
#
# Aliasing under leading-underscore names preserves the historical
# import-site spellings inside this module without making the schemas
# look "private to glue" -- they aren't, they're shared.
from areal.weight_sync.vllm_ext.protocol import (
    UpdateGroupRequest as _UpdateGroupRequest,
)
from areal.weight_sync.vllm_ext.protocol import (
    UpdateWeightsFromXcclRequest as _SetWeightMetaRequest,
)
from areal.weight_sync.vllm_ext.protocol import (
    UpdateWeightsFromXcclRequestLora as _SetWeightMetaLoraRequest,
)
from areal.weight_sync.vllm_ext.protocol import (
    UpdateWeightsRequest as _UpdateWeightsRequest,
)

logger = logging.getLogger("forge.engines.msswift.glue")


# ============================================================================
# Cross-mesh weight-sync group registry (Monarch observability hook)
# ----------------------------------------------------------------------------
# Background: ms-swift's TRL ``GRPOTrainer`` builds a ``VLLMClient``
# inside its own process; our patched ``init_communicator`` (below)
# wires that client's collective HCCL/XCCL group via
# :class:`areal.weight_sync.vllm_ext.client.WeightSyncClient`.  The
# resulting :class:`torch.distributed.ProcessGroup` lives on trainer
# rank 0 and is *invisible* to Monarch by default -- there is no actor
# wrapping it because the group is intrinsically process-local
# (``torch.distributed`` cannot move a PG between processes).
#
# Per ``framework-first-principles.mdc`` 准则 3 ("any adapter must
# plug into Monarch"), we expose the group's lifecycle through endpoints
# on :class:`forge.actors.msswift_trainer.MsSwiftTrainerActor` instead of
# spawning a separate Actor proc.  The endpoints just call into the two
# helpers defined here, which read / write this module-level registry.
#
# Design pattern -- "actor-fy by surfacing endpoints, not by spawning
# yet another proc":  When an adapter's resource cannot be moved out of
# its owning process (process-local state, in-process callbacks,
# framework lifecycle hooks), do not pretend it can by wrapping it in
# its own Actor.  Instead, give the *owning* actor (which Monarch
# already manages) endpoints that observe and tear down the resource.
# This keeps the lifecycle bound to its natural owner while making it
# fully Monarch-visible.
# ============================================================================

_AREAL_ACTIVE_CLIENTS: list = []  # of WeightSyncClient -- avoid eager import


def _register_active_client(wsc) -> None:
    """Called from ``_patched_init_communicator`` after a successful build.

    Idempotent: a duplicate registration (e.g. two consecutive
    ``init_communicator`` calls without an intervening close) silently
    de-dupes by identity so the registry can never grow unbounded.
    """
    for existing in _AREAL_ACTIVE_CLIENTS:
        if existing is wsc:
            return
    _AREAL_ACTIVE_CLIENTS.append(wsc)


def _unregister_active_client(wsc) -> None:
    """Called from ``_patched_close_communicator``.  Tolerant of unknown wsc."""
    try:
        _AREAL_ACTIVE_CLIENTS.remove(wsc)
    except ValueError:
        pass


def get_cross_mesh_group_status() -> list[dict]:
    """Snapshot the cross-mesh groups currently held by this process.

    Returns one dict per active :class:`WeightSyncClient`::

        [{"group_name": "swift_ws_0",
          "server_url": "http://...:8000",
          "world_size": 5,
          "backend": "hccl",
          "initialized": True}, ...]

    Empty list when no group has been built (e.g. before the first
    weight sync, or on non-rank-0 trainer ranks where TRL never calls
    ``init_communicator``).  Cheap and side-effect-free -- safe to
    invoke from a Monarch endpoint at any time.
    """
    out: list[dict] = []
    for wsc in _AREAL_ACTIVE_CLIENTS:
        out.append(
            {
                "group_name": getattr(wsc, "group_name", "<unknown>"),
                "server_url": getattr(wsc, "server_url", "<unknown>"),
                "world_size": getattr(wsc, "_world_size", None),
                "backend": getattr(wsc, "_backend", None),
                "initialized": getattr(wsc, "_group", None) is not None,
            }
        )
    return out


def teardown_cross_mesh_groups() -> dict:
    """Force-close every registered cross-mesh group.

    Exposed for Monarch-driven emergency cleanup: when the driver
    detects a stuck weight sync (e.g. trainer is alive but every push
    times out), it can call this through the trainer actor's
    ``cross_mesh_group_teardown`` endpoint to release the
    ``torch.distributed`` PG and the underlying HCCL/XCCL handles
    *without* killing the whole trainer process.  After this returns,
    a subsequent ``init_communicator`` call from TRL will rebuild the
    group from scratch.

    Returns ``{"closed": int, "errors": int, "details": [...]}``.
    Safe to call when the registry is empty.
    """
    closed = 0
    errors = 0
    details: list[dict] = []
    # Iterate a snapshot so callbacks that mutate _AREAL_ACTIVE_CLIENTS
    # (e.g. _unregister_active_client invoked indirectly by close()) do
    # not desync our loop.
    for wsc in list(_AREAL_ACTIVE_CLIENTS):
        group_name = getattr(wsc, "group_name", "<unknown>")
        try:
            wsc.close()
            closed += 1
            details.append({"group_name": group_name, "ok": True})
        except Exception as e:  # noqa: BLE001
            errors += 1
            details.append(
                {
                    "group_name": group_name,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            logger.warning(
                "teardown_cross_mesh_groups: WeightSyncClient.close raised on "
                "group=%s: %s",
                group_name,
                e,
            )
    _AREAL_ACTIVE_CLIENTS.clear()
    return {"closed": closed, "errors": errors, "details": details}


# ---------------------------------------------------------------- npu sampler


def _apply_vllm_ascend_sampler_fallback() -> None:
    """Force ``vllm_ascend`` top_k_top_p to its PyTorch fallback.

    The custom NPU kernel ``aclnnApplyTopKTopPCustom`` in vllm-ascend
    0.14.0rc1 raises at runtime ("inner error reported above").  Until a
    fixed wheel is available we route sampling through the existing
    PyTorch implementation that vllm-ascend ships next to the broken
    kernel.  No-op when vllm-ascend is not installed (CUDA hosts).
    """
    try:
        from vllm_ascend.sample import sampler as _sampler_mod
    except ImportError:
        return
    if hasattr(_sampler_mod, "_apply_top_k_top_p_pytorch"):
        _sampler_mod.apply_top_k_top_p = _sampler_mod._apply_top_k_top_p_pytorch
        if hasattr(_sampler_mod, "AscendTopKTopPSampler"):
            _sampler_mod.AscendTopKTopPSampler.apply_top_k_top_p = staticmethod(
                _sampler_mod._apply_top_k_top_p_pytorch
            )
        logger.info("vllm_ascend top_k_top_p forced to pytorch fallback")


# ---------------------------------------------------------------- server side


_AREAL_WORKER_EXT = "areal.weight_sync.vllm_ext.worker_extension.VLLMWorkerExtension"


# HTTP request schemas (``_UpdateGroupRequest`` etc.) used by the routes
# below are imported at the top of this module from
# ``areal.weight_sync.vllm_ext.protocol`` -- the single L2 source of
# truth shared with WeightSyncClient and server_router.


def _ok(msg: str = "ok"):
    from fastapi.responses import JSONResponse

    return JSONResponse({"success": True, "message": msg}, status_code=200)


_WORKER_PATCHED = False


def install_worker_patches() -> None:
    """Patches that must run in *every* process owning a vLLM engine.

    ms-swift uses ``multiprocessing.set_start_method('spawn')`` so the
    monkey-patches we apply in the driver never reach ``llm_worker``
    children.  We install this function from the spawned worker entry
    point as well -- it is idempotent.
    """
    global _WORKER_PATCHED
    if _WORKER_PATCHED:
        return
    _apply_vllm_ascend_sampler_fallback()

    from swift.pipelines.infer import rollout as _rollout
    from swift.rlhf_trainers.utils import check_vllm_version_ge

    def _patched_get_engine(args, template=None, **kwargs):
        # ms-swift's get_infer_engine HARDCODES worker_extension_cls
        # (rollout.py:477) which clobbers anything passed via kwargs.  We
        # rebuild get_infer_engine entirely so our class wins.
        from swift.infer_engine import GRPOVllmEngine

        kwargs.update(
            {
                "model_id_or_path": args.model,
                "model_type": args.model_type,
                "revision": args.model_revision,
                "torch_dtype": args.torch_dtype,
                "template": template,
                "use_async_engine": args.vllm_use_async_engine,
                "max_lora_rank": args.vllm_max_lora_rank,
            }
        )
        kwargs.update(args.get_vllm_engine_kwargs())
        kwargs.update({"enable_lora": args.vllm_enable_lora})
        kwargs["logprobs_mode"] = (
            "processed_logprobs" if check_vllm_version_ge("0.10.2") else None
        )

        engine_kwargs = kwargs.get("engine_kwargs", {}) or {}
        engine_kwargs["worker_extension_cls"] = _AREAL_WORKER_EXT
        load_format = engine_kwargs.pop("load_format", "auto")
        kwargs["load_format"] = load_format

        if args.vllm_use_async_engine and args.vllm_data_parallel_size > 1:
            engine_kwargs["data_parallel_size"] = args.vllm_data_parallel_size

        kwargs["engine_kwargs"] = engine_kwargs
        logger.info(
            "[child pid=%d] _patched_get_engine: worker_extension_cls=%s, "
            "engine_kwargs.keys=%s",
            os.getpid(),
            _AREAL_WORKER_EXT,
            sorted(engine_kwargs.keys()),
        )
        return GRPOVllmEngine(**kwargs)

    _rollout.SwiftRolloutDeploy.get_infer_engine = staticmethod(_patched_get_engine)
    _WORKER_PATCHED = True
    logger.info(
        "[pid=%d] SwiftRolloutDeploy.get_infer_engine fully overridden -> %s",
        os.getpid(),
        _AREAL_WORKER_EXT,
    )


def _spawn_wrapped_llm_worker(args, dp_rank, master_port, conn):
    """Worker entry that re-installs patches in the spawned child."""
    install_worker_patches()
    from swift.pipelines.infer.rollout import llm_worker

    return llm_worker(args, dp_rank, master_port, conn)


def install_server_patches() -> None:
    """Wire SwiftRolloutDeploy to expose AReaL's weight-sync HTTP surface.

    ms-swift runs vLLM workers in *child processes* and dispatches via
    ``multiprocessing.Pipe``, so AReaL's stock router (which expects
    ``app.state.engine_client`` to be an in-process vLLM engine) cannot be
    mounted as-is.  We register equivalent ``/areal_*`` handlers that
    forward via ms-swift's ``connection.send({'type': 'fire_and_forget',
    'method': 'collective_rpc', 'kwargs': {...}})`` shape -- the wire
    format faced by trainer-side ``WeightSyncClient`` is identical.
    """
    install_worker_patches()

    from multiprocessing import Pipe, Process

    from swift.pipelines.infer import rollout as _rollout

    def _patched_start(self):
        # Mirror the original implementation, but route children through our
        # wrapper so they re-apply ``install_worker_patches`` before they
        # build the vLLM engine.  Async-engine path is left unpatched for
        # this PoC (we run sync engines only).
        for dp_rank in range(self.num_connections):
            parent_conn, child_conn = Pipe()
            target = (
                _spawn_wrapped_llm_worker
                if not self.use_async_engine
                else _rollout.llm_worker_entry
            )
            process = Process(
                target=target,
                args=(self.args, dp_rank, self.master_port, child_conn),
            )
            process.start()
            self.connections.append(parent_conn)
            self.processes.append(process)
        logger.info(
            "SwiftRolloutDeploy: %d data-parallel worker(s) spawned via wrapped entry",
            self.num_connections,
        )

    _rollout.SwiftRolloutDeploy._start_data_parallel_workers = _patched_start

    _orig_register = _rollout.SwiftRolloutDeploy._register_rl_rollout_app

    def _patched_register(self):
        _orig_register(self)
        app = self.app

        def _dispatch(method: str, args: tuple, *, kind: str = "fire_and_forget"):
            kwargs = {"method": method, "args": tuple(args)}
            for connection in self.connections:
                connection.send(
                    {"type": kind, "method": "collective_rpc", "kwargs": kwargs}
                )

        async def _init(req: _UpdateGroupRequest):
            logger.info(
                "/areal_init_weights_update_group: master=%s:%s world=%d backend=%s group=%s",
                req.master_address,
                req.master_port,
                req.world_size,
                req.backend,
                req.group_name,
            )
            _dispatch(
                "init_update_weight_group",
                (
                    req.master_address,
                    req.master_port,
                    req.rank_offset,
                    req.world_size,
                    req.backend,
                    req.group_name,
                ),
            )
            return _ok("init_update_weight_group dispatched")

        async def _set_meta(req: _SetWeightMetaRequest):
            _dispatch(
                "set_weight_meta",
                (req.names, req.dtypes, req.shapes, req.group_name),
            )
            return _ok("set_weight_meta dispatched")

        async def _update_xccl():
            _dispatch("update_weight_xccl", ())
            return _ok("update_weight_xccl dispatched")

        async def _update_disk(req: _UpdateWeightsRequest):
            _dispatch("update_weights", (req.model_path,))
            return _ok("update_weights dispatched")

        async def _set_meta_lora(req: _SetWeightMetaLoraRequest):
            # Carries the seven LoRA-specific fields
            # (lora_name/int_id/target_modules/rank/alpha/bias/base_model_name)
            # the worker-side ``set_weight_meta_lora`` needs in addition
            # to the names/dtypes/shapes that the non-LoRA path passes.
            # The order MUST match
            # ``areal.weight_sync.vllm_ext.worker_extension
            # .VLLMWorkerExtension.set_weight_meta_lora``.
            _dispatch(
                "set_weight_meta_lora",
                (
                    req.names,
                    req.dtypes,
                    req.shapes,
                    req.group_name,
                    req.lora_name,
                    req.lora_int_id,
                    req.lora_target_modules,
                    req.lora_rank,
                    req.lora_alpha,
                    req.lora_bias,
                    req.base_model_name,
                ),
            )
            return _ok("set_weight_meta_lora dispatched")

        async def _update_lora_xccl():
            _dispatch("update_weight_lora_xccl", ())
            return _ok("update_weight_lora_xccl dispatched")

        async def _pause():
            return _ok("pause_generation noop (ms-swift drives sync inference)")

        async def _continue():
            return _ok("continue_generation noop")

        app.post("/areal_init_weights_update_group")(_init)
        app.post("/areal_set_update_weight_meta")(_set_meta)
        app.post("/areal_update_weights_xccl")(_update_xccl)
        app.post("/areal_update_weights")(_update_disk)
        app.post("/areal_set_update_weight_meta_lora")(_set_meta_lora)
        app.post("/areal_update_weights_lora_xccl")(_update_lora_xccl)
        app.post("/areal_pause_generation")(_pause)
        app.post("/areal_continue_generation")(_continue)
        logger.info(
            "SwiftRolloutDeploy: /areal_* shim routes registered "
            "(init/set_meta/update_xccl/update/set_meta_lora/"
            "update_lora_xccl/pause/continue)"
        )

    _rollout.SwiftRolloutDeploy._register_rl_rollout_app = _patched_register


# ---------------------------------------------------------------- client side


def _resolve_master_addr() -> str:
    """Pick the NIC IP for the cross-mesh TCPStore master.

    Trainer rank 0 binds at ``addr:port``; rollout vLLM workers connect
    in via that same tuple.  We default to the host's primary IP rather
    than ``127.0.0.1`` so cross-host workers can actually reach us.
    """
    addr = os.environ.get("AREAL_WS_MASTER_ADDR")
    if addr:
        return addr
    try:
        host = socket.gethostbyname(socket.gethostname())
        if host and host != "127.0.0.1":
            return host
    except OSError:
        pass
    return os.environ.get("MASTER_ADDR") or "127.0.0.1"


def _free_port() -> int:
    port = os.environ.get("AREAL_WS_MASTER_PORT")
    if port:
        return int(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def install_client_patches() -> None:
    """Replace VLLMClient.{init,update_named_param,close}_communicator with WeightSyncClient calls.

    Installed inside the trainer actor before ``rlhf_main`` is invoked.
    See module docstring for the rationale (vllm-ascend's PyHccl single-
    host limit vs. ``init_custom_process_group`` multi-host).
    """
    _apply_vllm_ascend_sampler_fallback()

    from contextlib import contextmanager
    from datetime import timedelta

    import requests
    import torch
    from swift.rlhf_trainers import vllm_client as _vc

    from areal.weight_sync.vllm_ext.client import (
        WeightSyncClient,
        slice_flattened_lora_tensor,
    )

    # ms-swift uses fixed constants for the in-process LoRA adapter slot
    # vLLM addresses on the rollout server.  We import them so our
    # patched adapter pushes use the same name/id as ms-swift's native
    # path -- otherwise vLLM would see two adapters for the same logical
    # tuner and reject the second.
    try:
        from swift.rlhf_trainers.utils import (
            VLLM_LORA_INT_ID,
            VLLM_LORA_NAME,
        )
    except ImportError:
        # Older ms-swift releases predate these constants; fall back to
        # the documented values so we don't crash on import.  If they
        # ever change upstream, the resulting rollout error will be loud.
        VLLM_LORA_INT_ID = 111
        VLLM_LORA_NAME = "swift_lora"

    def _peft_config_to_dict(peft_config) -> dict:
        """Inline copy of ms-swift's ``peft_config_to_dict`` (avoids a hard
        dependency on the upstream helper, which has moved between
        modules across releases).  Coerces ``set`` target_modules to
        list so JSON serialisation works across the wire."""
        from dataclasses import asdict, is_dataclass

        if isinstance(peft_config, dict):
            cfg = dict(peft_config)
        elif is_dataclass(peft_config):
            cfg = asdict(peft_config)
        elif hasattr(peft_config, "to_dict"):
            cfg = peft_config.to_dict()
        else:
            raise TypeError(
                f"Unsupported peft_config type: {type(peft_config).__name__}"
            )
        tm = cfg.get("target_modules")
        if isinstance(tm, set):
            cfg["target_modules"] = list(tm)
        return cfg

    def _resolve_base_model_name(peft_config_dict: dict) -> str:
        """Pick the most stable base-model identifier from the peft dict.

        vLLM's ``LoRARequest`` stores ``base_model_name`` for
        compatibility checks against the model the engine was loaded
        with.  PEFT writes it under ``base_model_name_or_path``; we fall
        back to a sentinel rather than crashing because the field is
        descriptive, not load-bearing for our XCCL push path.
        """
        return str(
            peft_config_dict.get("base_model_name_or_path")
            or peft_config_dict.get("base_model_name")
            or "unknown_base_model"
        )

    @contextmanager
    def _suspend_torchelastic_agent_store():
        """Force PyTorch's tcp:// rendezvous to use start_daemon=(rank==0).

        When the trainer is launched under ``torchrun``, the agent sets
        ``TORCHELASTIC_USE_AGENT_STORE=True`` so subsequent
        ``init_process_group`` calls reuse the agent's TCPStore.  That
        forces *every* call through ``rendezvous.py:189`` with
        ``is_master=False``, which is correct for the trainer's main
        DDP group but breaks our cross-mesh group: trainer rank 0 needs
        to BIND, not connect to a non-existent daemon at our private
        master_port.  Temporarily unset the env var so PyTorch falls
        back to the rank-0-as-daemon branch (line 197+).
        """
        key = "TORCHELASTIC_USE_AGENT_STORE"
        prev = os.environ.pop(key, None)
        try:
            yield
        finally:
            if prev is not None:
                os.environ[key] = prev

    def _patched_init_communicator(self, device: int | str = 0) -> None:
        master_addr = _resolve_master_addr()
        self._areal_clients: list[WeightSyncClient] = []
        # Preserve old attribute so legacy code paths (e.g. atexit) survive.
        self.pynccl_comms = []

        timeout = timedelta(seconds=120)

        for i in range(self.num_servers):
            base = self.base_urls[i]
            r = self.sessions[i].get(f"{base}/get_world_size/")
            if r.status_code != 200:
                raise RuntimeError(f"Server {i} get_world_size failed: {r.text}")
            vllm_world_size = r.json()["world_size"]

            master_port = _free_port()
            logger.info(
                "VLLMClient[%d]: starting areal init_communicator "
                "(server=%s, master=%s:%d, vllm_world=%d, timeout=%ss)",
                i,
                base,
                master_addr,
                master_port,
                vllm_world_size,
                int(timeout.total_seconds()),
            )
            wsc = WeightSyncClient(
                base,
                group_name=f"swift_ws_{i}",
                request_timeout=600.0,
                session=self.sessions[i],
            )
            with _suspend_torchelastic_agent_store():
                wsc.init_communicator(
                    master_addr=master_addr,
                    master_port=master_port,
                    vllm_world_size=vllm_world_size,
                    backend=None,  # auto-detect: hccl on NPU
                    timeout=timeout,
                )
            self._areal_clients.append(wsc)
            # Make this group Monarch-visible: trainer actor exposes
            # ``cross_mesh_group_status`` / ``cross_mesh_group_teardown``
            # endpoints that read this registry (准则 3).
            _register_active_client(wsc)
            logger.info(
                "VLLMClient[%d]: areal init_communicator OK (server=%s, master=%s:%d, vllm_world=%d)",
                i,
                base,
                master_addr,
                master_port,
                vllm_world_size,
            )

    def _patched_update_named_param(self, name: str, weights: torch.Tensor) -> None:
        for wsc in self._areal_clients:
            # Single-element bucket -- preserves ms-swift's per-param semantics.
            wsc._broadcast_bucket([(name, weights)])  # noqa: SLF001

    def _patched_update_adapter_param(self, peft_config, lora_params) -> None:
        """LoRA-incremental sync, non-flattened path.

        Replaces ms-swift's native ``VLLMClient.update_adapter_param`` --
        which posts to the rollout server's own ``/update_adapter_param/``
        endpoint and then broadcasts via ``self.pynccl_comms`` (we never
        initialised that, by design).  Instead we route the LoRA tensors
        through our ``WeightSyncClient.push_lora_adapter_xccl`` so the
        already-established cross-mesh ``swift_ws`` HCCL group is
        reused, and the vLLM workers receive via the areal-side
        ``/areal_set_update_weight_meta_lora`` +
        ``/areal_update_weights_lora_xccl`` endpoints (which apply the
        adapter via ``LoRAModel.from_lora_tensors`` rather than
        merge-into-base).
        """
        peft_dict = _peft_config_to_dict(peft_config)
        base_model_name = _resolve_base_model_name(peft_dict)
        named_lora = (
            list(lora_params.items())
            if hasattr(lora_params, "items")
            else list(lora_params)
        )
        logger.info(
            "VLLMClient.update_adapter_param -> push_lora_adapter_xccl: "
            "params=%d lora_name=%s int_id=%d",
            len(named_lora),
            VLLM_LORA_NAME,
            VLLM_LORA_INT_ID,
        )
        for wsc in self._areal_clients:
            wsc.push_lora_adapter_xccl(
                peft_dict,
                named_lora,
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                base_model_name=base_model_name,
            )

    def _patched_update_adapter_flattened_param(
        self, peft_config, metadatas, flattened_tensor
    ) -> None:
        """LoRA-incremental sync, flattened-bucket path.

        ms-swift's default (``enable_flattened_weight_sync=True``) packs
        every LoRA matrix into one contiguous tensor and posts the
        ``FlattenedTensorMetadata`` list separately.  Our XCCL worker
        broadcasts and applies tensors *individually*, so we slice the
        bucket back into per-tensor views via
        :func:`areal.weight_sync.vllm_ext.client.slice_flattened_lora_tensor`
        before delegating to the same code path as the non-flattened
        method.  Bandwidth on the wire is identical to native ms-swift
        (only the LoRA delta), just with N broadcasts instead of 1.
        """
        meta_dicts = [
            (
                m.model_dump()
                if hasattr(m, "model_dump")
                else m.dict()
                if hasattr(m, "dict")
                else dict(m)
            )
            for m in metadatas
        ]
        peft_dict = _peft_config_to_dict(peft_config)
        base_model_name = _resolve_base_model_name(peft_dict)
        named_lora = slice_flattened_lora_tensor(flattened_tensor, meta_dicts)
        logger.info(
            "VLLMClient.update_adapter_flattened_param -> push_lora_adapter_xccl: "
            "params=%d (un-flattened) lora_name=%s int_id=%d",
            len(named_lora),
            VLLM_LORA_NAME,
            VLLM_LORA_INT_ID,
        )
        for wsc in self._areal_clients:
            wsc.push_lora_adapter_xccl(
                peft_dict,
                named_lora,
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                base_model_name=base_model_name,
            )

    def _patched_reset_prefix_cache(self):
        # Keep legacy ms-swift /reset_prefix_cache/ endpoint -- it isn't part
        # of the weight-sync protocol and we don't ship a replacement.
        for i in range(self.num_servers):
            try:
                r = self.sessions[i].post(f"{self.base_urls[i]}/reset_prefix_cache/")
                if r.status_code != 200:
                    logger.warning(
                        "reset_prefix_cache server %d returned %d: %s",
                        i,
                        r.status_code,
                        r.text[:200],
                    )
            except requests.RequestException as e:
                logger.warning("reset_prefix_cache server %d raised: %s", i, e)

    def _patched_close_communicator(self):
        for wsc in getattr(self, "_areal_clients", []):
            try:
                wsc.close()
            except Exception as e:  # pragma: no cover
                logger.warning("WeightSyncClient.close raised: %s", e)
            finally:
                # Mirror the registration in _patched_init_communicator
                # so the Monarch-visible registry stays consistent even
                # when close() raises.
                _unregister_active_client(wsc)
        self._areal_clients = []
        self.pynccl_comms = []

    _vc.VLLMClient.init_communicator = _patched_init_communicator
    _vc.VLLMClient.update_named_param = _patched_update_named_param
    _vc.VLLMClient.update_adapter_param = _patched_update_adapter_param
    _vc.VLLMClient.update_adapter_flattened_param = (
        _patched_update_adapter_flattened_param
    )
    _vc.VLLMClient.reset_prefix_cache = _patched_reset_prefix_cache
    _vc.VLLMClient.close_communicator = _patched_close_communicator

    logger.info(
        "VLLMClient patched: init_communicator/update_named_param/"
        "update_adapter_param/update_adapter_flattened_param/close_communicator "
        "now delegate to areal.weight_sync.vllm_ext.client.WeightSyncClient"
    )


__all__ = [
    "install_server_patches",
    "install_worker_patches",
    "install_client_patches",
    "get_cross_mesh_group_status",
    "teardown_cross_mesh_groups",
]
