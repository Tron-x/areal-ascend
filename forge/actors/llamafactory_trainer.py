"""LlamaFactoryTrainerActor -- run LlamaFactory's ``run_exp`` inside a Monarch actor.

**Conforms to:** :class:`forge.core.protocols.SPMDTrainerProtocol`
(Adapter Path; partial -- ``setup_env`` + ``run`` are present, ``teardown``
is **TODO**).  See :class:`forge.actors.msswift_trainer.MsSwiftTrainerActor`
for the canonical full conformance.

Mirror of :class:`forge.actors.titan_trainer.TitanTrainerActor`, but for the
LlamaFactory training stack.  The integration contract is intentionally
identical to TorchTitan's:

    Driver  -- spawns one actor per (host, NPU) cell.
    Actor   -- reads its YAML, sets up FSDP env vars, calls
               ``llamafactory.train.tuner.run_exp``.

Why mirror TorchTitan instead of inventing a new shape?
    The whole point of putting LlamaFactory behind a Monarch actor is to
    make it *swappable* with TorchTitan from the orchestration layer's
    point of view -- same ``run`` endpoint, same ``setup_env`` wiring,
    same return shape.  Future RL drivers shouldn't care which SPMD
    backend they're driving.

Translating ``accelerate launch --config_file fsdp2.yaml ...``
-------------------------------------------------------------
Upstream LlamaFactory is launched via ``accelerate launch
--config_file <fsdp2_config.yaml> src/train.py <lf_yaml>``.  The
``accelerate`` wrapper does two things before invoking ``train.py``:

1. Sets ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` /
   ``MASTER_PORT`` from its own launcher state.
2. Translates the FSDP config YAML into a fixed set of env vars
   (``ACCELERATE_USE_FSDP``, ``FSDP_VERSION``, ``FSDP_AUTO_WRAP_POLICY``,
   ...).  ``accelerate.Accelerator`` later reads these env vars when the
   training script constructs it.

We replicate exactly that contract here:

* (1) is handled by the inherited :class:`SPMDActor._setup_env` (driver
  calls ``actor.setup_env(addr, port)``).
* (2) is handled by :func:`_export_fsdp2_env` in this file -- a small
  parser keyed off the same field names ``accelerate.commands.launch``
  uses.  Keeping the mapping local (instead of importing accelerate's
  internal launcher) lets the actor file stay importable on a CPU-only
  driver host.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time
from typing import Any

from monarch._src.spmd.actor import SPMDActor
from monarch.actor import endpoint

logger = logging.getLogger(__name__)


# Fields we forward verbatim from the accelerate FSDP YAML's
# ``fsdp_config:`` block to env vars.  Mirrors the assignments in
# ``accelerate.utils.launch.prepare_multi_gpu_env`` (the only place
# accelerate actually reads them).  Keys not in this map are silently
# ignored so we tolerate older / newer accelerate schemas without
# breaking.
_FSDP_FIELD_TO_ENV: dict[str, str] = {
    "fsdp_version": "FSDP_VERSION",
    "fsdp_auto_wrap_policy": "FSDP_AUTO_WRAP_POLICY",
    "fsdp_transformer_layer_cls_to_wrap": "FSDP_TRANSFORMER_CLS_TO_WRAP",
    "fsdp_backward_prefetch": "FSDP_BACKWARD_PREFETCH",
    "fsdp_state_dict_type": "FSDP_STATE_DICT_TYPE",
    "fsdp_offload_params": "FSDP_OFFLOAD_PARAMS",
    "fsdp_reshard_after_forward": "FSDP_RESHARD_AFTER_FORWARD",
    "fsdp_forward_prefetch": "FSDP_FORWARD_PREFETCH",
    "fsdp_use_orig_params": "FSDP_USE_ORIG_PARAMS",
    "fsdp_cpu_ram_efficient_loading": "FSDP_CPU_RAM_EFFICIENT_LOADING",
    "fsdp_sync_module_states": "FSDP_SYNC_MODULE_STATES",
    "fsdp_activation_checkpointing": "FSDP_ACTIVATION_CHECKPOINTING",
    "fsdp_min_num_params": "FSDP_MIN_NUM_PARAMS",
    "fsdp_sharding_strategy": "FSDP_SHARDING_STRATEGY",
}

# Boolean fields whose env-var representation accelerate spells in
# lowercase ("true" / "false").  Everything else is stringified as-is.
_FSDP_BOOL_FIELDS: frozenset[str] = frozenset(
    {
        "fsdp_offload_params",
        "fsdp_reshard_after_forward",
        "fsdp_forward_prefetch",
        "fsdp_use_orig_params",
        "fsdp_cpu_ram_efficient_loading",
        "fsdp_sync_module_states",
        "fsdp_activation_checkpointing",
    }
)


def _load_yaml_or_dict(spec: str | os.PathLike | dict, *, label: str) -> dict:
    """Coerce a YAML spec to a dict, accepting either a path or a parsed dict.

    Driver-side callers pre-parse the algorithm-author YAMLs and ship
    dicts to the actor so workers don't need filesystem access to those
    files (single source of truth = the launcher host).  Legacy
    callers that still pass a path keep working: we just open it here.

    ``label`` is a short tag ("lf", "accelerate", ...) used only in
    the ValueError raised on a bad type so the breakage is debuggable.
    """
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, (str, os.PathLike)):
        import yaml

        with open(spec) as fh:
            return yaml.safe_load(fh) or {}
    raise TypeError(
        f"{label} YAML spec must be dict or path, got {type(spec).__name__}"
    )


def _export_fsdp2_env(accelerate_cfg: dict | str | os.PathLike) -> dict[str, str]:
    """Translate an ``accelerate`` config (dict or YAML path) into env vars.

    Parses just the subset of ``accelerate``'s schema we care about:

    * ``distributed_type: FSDP``        -> ``ACCELERATE_USE_FSDP=true``
    * ``mixed_precision: bf16|fp16|...``-> ``ACCELERATE_MIXED_PRECISION=...``
    * ``fsdp_config.<field>``           -> ``FSDP_<FIELD>`` (per
      :data:`_FSDP_FIELD_TO_ENV`)

    Returns the dict of env vars that were set, for logging.
    Side-effect: mutates :data:`os.environ`.

    Accepts either a parsed dict (driver-side preferred) or a filesystem
    path (legacy).  When a path is given, it is opened and parsed here.

    Notes:
        * ``cpu_ram_efficient_loading`` requires ``sync_module_states=True``
          (accelerate enforces this at launch time).  We mirror the check
          so we fail fast at the actor instead of deep inside FSDP wrap.
        * ``num_processes`` / ``num_machines`` from the YAML are
          intentionally **ignored** -- the Monarch driver owns rank
          topology, the YAML is purely for FSDP semantics.
    """
    cfg = _load_yaml_or_dict(accelerate_cfg, label="accelerate")

    set_vars: dict[str, str] = {}
    if str(cfg.get("distributed_type", "")).upper() != "FSDP":
        raise ValueError(
            "accelerate config: distributed_type must be FSDP "
            f"(got {cfg.get('distributed_type')!r}); "
            "this actor is FSDP-only by design."
        )
    set_vars["ACCELERATE_USE_FSDP"] = "true"

    mp = cfg.get("mixed_precision", "no")
    set_vars["ACCELERATE_MIXED_PRECISION"] = str(mp)

    fsdp_block = cfg.get("fsdp_config", {}) or {}
    for yaml_key, env_key in _FSDP_FIELD_TO_ENV.items():
        if yaml_key not in fsdp_block:
            continue
        value = fsdp_block[yaml_key]
        if yaml_key in _FSDP_BOOL_FIELDS:
            set_vars[env_key] = str(bool(value)).lower()
        else:
            set_vars[env_key] = str(value)

    if (
        set_vars.get("FSDP_CPU_RAM_EFFICIENT_LOADING") == "true"
        and set_vars.get("FSDP_SYNC_MODULE_STATES", "false") != "true"
    ):
        raise ValueError(
            "accelerate config: fsdp_cpu_ram_efficient_loading=true "
            "requires fsdp_sync_module_states=true (accelerate constraint)."
        )

    os.environ.update(set_vars)
    return set_vars


class LlamaFactoryTrainerActor(SPMDActor):
    """Run ``llamafactory.train.tuner.run_exp`` once per actor instance.

    Subclasses :class:`monarch._src.spmd.actor.SPMDActor` to inherit
    rank / device env wiring + the ``setup_env(master_addr, master_port)``
    endpoint already vetted by Monarch's SPMD path.  We add a single
    ``run()`` endpoint that does the LlamaFactory-specific work.

    Endpoint contract is intentionally identical to
    :class:`forge.actors.titan_trainer.TitanTrainerActor.run` modulo
    args, so a future ``forge launch --mode llamafactory-train`` driver
    can mirror ``forge.apps.titan_pretrain`` with no shape divergence.
    """

    @endpoint
    def run(
        self,
        lf_yaml: str | dict,
        accelerate_yaml: str | dict,
        cwd: str,
        overrides: dict[str, Any] | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        use_modelscope: bool = False,
        extra_env: dict[str, str] | None = None,
        lf_yaml_origin: str | None = None,
    ) -> dict[str, Any]:
        """Drive one full LlamaFactory training run on this rank.

        Args:
            lf_yaml: LlamaFactory training spec.  Accepts either a
                pre-parsed ``dict`` (driver-side preferred -- ships
                a single source-of-truth across hosts so workers
                don't need filesystem access) or a path to a YAML
                file (legacy, kept for the ``this_host()`` driver and
                manual smokes).  Equivalent to ``src/train.py <yaml>``
                upstream once resolved to a dict.
            accelerate_yaml: Accelerate FSDP2 config.  Same dict-or-path
                contract as ``lf_yaml``.  Translated into env vars
                BEFORE ``run_exp`` imports accelerate / transformers,
                so FSDP wrap picks them up.  Equivalent to
                ``accelerate launch --config_file ...`` upstream.
            cwd: Working directory before ``run_exp`` is called.  Must
                be the LlamaFactory repo root because LF resolves
                ``data/dataset_info.json`` relative to it.
            overrides: Extra (key, value) pairs merged on top of the
                LF YAML dict before dispatch.  Lets a driver flip
                ``max_steps``, ``output_dir``, ``model_name_or_path``,
                etc., without forking the YAML for every smoke run.
            master_addr / master_port: If provided (driver-supplied)
                set the rendezvous via ``self._setup_env``.  When
                ``None`` the caller is expected to have invoked
                ``setup_env`` beforehand.
            use_modelscope: When ``True``, set ``USE_MODELSCOPE_HUB=1``
                inside this actor proc *before* the ``llamafactory``
                import.  Required when (a) HuggingFace is unreachable
                from this host AND (b) the driver lives in a different
                process tree (e.g. SSHJob workers don't inherit the
                driver's env).  Single-host ``this_host()`` drivers
                that already export this in their parent shell can
                leave it ``False``.
            extra_env: Arbitrary env-var overrides applied alongside
                ``use_modelscope`` (after FSDP env, before LF import).
                Escape hatch for one-offs (HF_ENDPOINT, MODELSCOPE_CACHE,
                etc.) without churning this signature.
            lf_yaml_origin: Optional human label for ``lf_yaml`` -- e.g.
                the driver's filesystem path -- folded into the actor's
                log line so users can map a worker log back to the
                source file.  Purely cosmetic; safe to leave ``None``.

        Returns:
            ``{"rank": int, "host": str, "elapsed_s": float, "ok": True}``
            so the driver can sanity-check that every rank actually
            reached training completion.

        Notes:
            * Heavy imports (``torch``, ``torch_npu``, ``transformers``,
              ``llamafactory``) happen INSIDE ``run()`` so this module
              stays importable on CPU-only orchestration hosts.
            * ``ACCELERATE_USE_FSDP`` etc. MUST be exported before the
              first ``import accelerate`` in this proc.  We do that via
              :func:`_export_fsdp2_env` before any ``llamafactory``
              import.
            * Same ordering rule applies to ``USE_MODELSCOPE_HUB``:
              must be set before ``import llamafactory`` because LF's
              hub-resolver is keyed on it at module-load time.
        """

        if master_addr is not None and master_port is not None:
            self._setup_env(master_addr, master_port)

        os.chdir(cwd)

        if use_modelscope:
            os.environ.setdefault("USE_MODELSCOPE_HUB", "1")
        if extra_env:
            os.environ.update({str(k): str(v) for k, v in extra_env.items()})

        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        world_size = int(os.environ.get("WORLD_SIZE", "-1"))
        host = socket.gethostname()

        fsdp_env = _export_fsdp2_env(accelerate_yaml)
        if rank == 0:
            logger.info(
                "LlamaFactoryTrainerActor exported FSDP env: %s",
                {k: v for k, v in sorted(fsdp_env.items())},
            )

        # Force NPU device pinning per LOCAL_RANK before any torch / LF
        # import touches a device.  This matches ``accelerate launch``
        # which sets ASCEND_RT_VISIBLE_DEVICES per worker.  We instead
        # let every proc see every NPU and pin via torch_npu.set_device,
        # which is what ``transformers.Trainer`` does on NPU when
        # LOCAL_RANK is set.
        import torch
        import torch_npu  # noqa: F401  -- registers the npu backend

        if local_rank >= 0 and torch.npu.is_available():
            torch.npu.set_device(local_rank)

        # Resolve the LF YAML to a dict of training args, apply the
        # caller's overrides, and dispatch.  We pass a dict (not a list
        # of CLI args) because ``run_exp`` already accepts dict input
        # via its ``read_args`` shim, which keeps us off sys.argv -- a
        # must on Monarch where sys.argv is owned by the bootstrap.
        from llamafactory.train.tuner import run_exp

        lf_args: dict[str, Any] = _load_yaml_or_dict(lf_yaml, label="lf")

        if overrides:
            lf_args.update(overrides)

        logger.info(
            "LlamaFactoryTrainerActor rank=%d local_rank=%d world_size=%d "
            "host=%s lf_yaml=%s",
            rank,
            local_rank,
            world_size,
            host,
            lf_yaml_origin or "<inline-dict>",
        )

        t0 = time.time()
        err: BaseException | None = None
        try:
            run_exp(args=lf_args)
        except BaseException as e:  # noqa: BLE001
            err = e
            logger.exception("LlamaFactoryTrainerActor rank=%d crashed", rank)
            raise
        finally:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning(
                    "LlamaFactoryTrainerActor rank=%d destroy_process_group raised %r",
                    rank,
                    cleanup_exc,
                )

        elapsed = time.time() - t0
        try:
            torch.npu.synchronize()
        except Exception:  # noqa: BLE001
            pass

        sys.stdout.flush()
        sys.stderr.flush()

        return {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "host": host,
            "elapsed_s": elapsed,
            "ok": err is None,
            "lf_yaml": lf_yaml_origin or "<inline-dict>",
        }
