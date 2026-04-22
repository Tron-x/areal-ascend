"""GRPO training entry point using Forge actors.

Two backend paths:

- **TrainBackend** (legacy/AReaL): synchronous ``trainer.train_step(step)``
  with built-in rollout + weight sync.
- **TrainEngine** (new/FSDP/Megatron): uses ``BatchAdapter`` for external
  batch, ``WeightSyncStrategy`` for weight transfer.

Engine-agnostic -- the backend is selected via ``ForgeConfig.backend_type``
(default ``"areal"``), configurable to any engine in ``forge/engines/``.

Usage::

    python -m forge.apps.grpo \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml
"""

from __future__ import annotations

import asyncio
import logging
import os

from forge.actors.generator import Generator
from forge.actors.reward import RewardActor
from forge.actors.trainer import TrainerActor
from forge.bootstraps import ensure_ascend_custom_opp_path
from forge.core.types import Episode
from forge.engines import create_batch_adapter, create_config_bridge, create_engine
from forge.provisioner import init_provisioner, shutdown

logger = logging.getLogger("ForgeGRPO")


# Storage spawn / bootstrap helpers used to live here; they moved into the
# backend implementations (see
# ``forge/engines/weight_sync/backends/torchstore_multi_vol.py::_storage_bootstrap_factory``)
# now that weight-sync goes through ``WeightSyncService`` + pluggable backend.


async def _create_weight_sync(forge_cfg, trainer, generator):
    """Create and initialize a weight-sync driver for TrainEngine backends.

    Two dispatch paths depending on ``FORGE_WEIGHT_SYNC``:

    * ``"torchstore"`` (and future values "p2p_rdma" / "dedicated_ps" /
      "areal_xccl"):  go through the new ``WeightSyncService`` +
      ``WeightSyncBackend`` abstraction.  The service owns the
      parallelism-aware orchestration; backends plug in under it.  This is
      the path we want all new callers on.

    * ``"nccl"`` / ``"checkpoint"`` / ``"hixl"`` (legacy):  fall through
      to the old ``WeightSyncStrategy`` factory.  Kept for one release so
      existing deployments don't break.

    Backend selection for the Service path is controlled by
    ``FORGE_WEIGHT_SYNC_BACKEND`` (default ``"torchstore_multi_vol"``):
    see ``forge/engines/weight_sync/backends/__init__.py``.
    """
    method_str = os.environ.get("FORGE_WEIGHT_SYNC", "nccl")

    if method_str == "torchstore":
        return await _create_weight_sync_service(forge_cfg, trainer, generator)
    return await _create_legacy_weight_sync(forge_cfg, trainer, generator, method_str)


async def _spawn_storage_mesh(
    host_mesh_name: str,
    num_vols: int,
    npu_base: int,
):
    """Provision a torchstore storage ``ProcMesh`` on a named host mesh.

    This used to live inside ``MultiVolTorchstoreBackend.initialize`` which
    (a) hard-coded the host to the trainer's and (b) hid the NPU pinning
    bootstrap behind the backend API.  Moving it up here turns "where does
    storage run" into a **driver-level decision** -- today picked by an
    env var, tomorrow by the ``meshes:`` YAML block, next year by a
    Slurm/K8s launcher.  The backend just consumes whatever ``ProcMesh``
    it's handed.

    Args:
        host_mesh_name: name of a HostMesh already registered with the
            provisioner (``"trainer"`` / ``"generator"`` / ``"ps"`` /
            arbitrary custom name).  Must resolve via
            ``provisioner.get_host_mesh(name)``.
        num_vols: number of storage volumes to spawn on that host mesh
            (one proc per volume; a 4-rank trainer typically spawns 4
            volumes so LocalRankStrategy routes cleanly).
        npu_base: first physical NPU id to pin volumes to on the host
            (volume ``i`` lands on NPU ``npu_base + i``).  The trainer's
            own NPUs must not overlap this range.
    """
    # Bootstrap migration note (see forge/docs/weight_sync.md §7.1 Bug B):
    # the legacy ``_storage_bootstrap_factory`` closure used to be handed
    # to ``spawn_procs(bootstrap=...)`` so every storage proc could set
    # ``ASCEND_RT_VISIBLE_DEVICES`` + HiXL/HCCL env vars before any
    # ``import torch_npu`` happened.  Our Monarch build silently skips
    # that callback (observed 2026-04-22 via marker-file instrumentation:
    # no marker files produced on either host), so every storage vol
    # ended up running without the mask and all 4 HiXL engines converged
    # on the same physical NPU.  We now mirror torchforge's workaround:
    # spawn procs with no bootstrap, then spawn a small ``StorageEnvSetter``
    # actor on the mesh and call its ``setup`` endpoint, which fires in
    # each proc before torchstore imports.
    from forge.engines.weight_sync._env_setter import StorageEnvSetter
    from forge.provisioner import _get_provisioner

    provisioner = await _get_provisioner()
    storage_hosts = await provisioner.get_host_mesh(host_mesh_name)
    storage_mesh = storage_hosts.spawn_procs(
        per_host={"procs": num_vols},
        name="torchstore_storage_multi_vol",
    )

    setter = storage_mesh.spawn("_storage_env_setter", StorageEnvSetter)
    # pool_mb reads the env that run_multinode.sh / YAML forwards.
    pool_mb = int(os.environ.get("TORCHSTORE_MONARCH_RDMA_POOL_MB", "8192"))
    setup_mesh = await setter.setup.call(npu_base=npu_base, pool_mb=pool_mb)

    # Collect the per-rank confirmation so we can print a single
    # driver-visible line that proves the env actually landed (as
    # opposed to the old "spawn worked but bootstrap silently skipped"
    # situation where we only had the caller-side print of intent).
    per_rank_info = []
    for _ref, payload in setup_mesh.items():
        if isinstance(payload, dict):
            per_rank_info.append(payload)
    per_rank_info.sort(key=lambda d: d.get("local_rank", -1))
    print(
        f"[WeightSync] spawned {num_vols} storage volumes on host mesh "
        f"{host_mesh_name!r} (NPU range {npu_base}..{npu_base + num_vols - 1}); "
        f"env setup confirmed: {per_rank_info}",
        flush=True,
    )
    return storage_mesh


async def _create_weight_sync_service(forge_cfg, trainer, generator):
    """New path: WeightSyncService + pluggable backend."""
    from forge.engines.weight_sync.backends import create_backend
    from forge.engines.weight_sync.service import (
        ParallelLayout,
        WeightSyncService,
    )

    backend_name = os.environ.get("FORGE_WEIGHT_SYNC_BACKEND", "torchstore_multi_vol")
    # Backend-specific options from env (kept narrow so YAML stays clean).
    backend_kwargs: dict = {}
    # Both torchstore_multi_vol and collective_broadcast share the same
    # trainer-side HiXL put path, so they take the same storage-mesh
    # provisioning knobs. The shared block is below; the bcast backend
    # layers its HCCL-group knobs on top.
    if backend_name in ("torchstore_multi_vol", "collective_broadcast"):
        # Placement knobs: these used to be baked into the backend; now
        # the driver provisions the storage mesh and hands it in, so the
        # backend becomes topology-agnostic.  Env vars still take
        # precedence so we can A/B "same host as trainer" vs
        # "different host" without touching YAML.
        #
        #   FORGE_STORAGE_HOST_MESH   -- name of the host mesh to spawn
        #                                storage volumes on.  Defaults to
        #                                the trainer host, which
        #                                reproduces the legacy colocated
        #                                layout bit-for-bit.  Setting
        #                                "generator" (or any other
        #                                registered host name) moves
        #                                storage off the trainer.
        #   TORCHSTORE_STORAGE_NPU_BASE -- first NPU id on that host for
        #                                the volumes.  Defaults to
        #                                train_world (right after the
        #                                trainer's NPUs on the trainer
        #                                host; for a dedicated storage
        #                                host 0 is usually correct).
        #
        # When ``FORGE_STORAGE_SPAWN_IN_BACKEND=1`` we skip external
        # spawning entirely and let the backend fall back to its legacy
        # self-spawn path -- useful for the very first smoke while we
        # bed this refactor in.
        npu_base_env = os.environ.get("TORCHSTORE_STORAGE_NPU_BASE")
        pool_mb_env = os.environ.get("TORCHSTORE_MONARCH_RDMA_POOL_MB")
        storage_host_name = os.environ.get("FORGE_STORAGE_HOST_MESH", "trainer")
        spawn_in_backend = os.environ.get("FORGE_STORAGE_SPAWN_IN_BACKEND", "0") == "1"

        if pool_mb_env is not None:
            backend_kwargs["pool_mb"] = int(pool_mb_env)

        if spawn_in_backend:
            # Legacy path: backend spawns storage itself on the trainer
            # host.  Honors TORCHSTORE_STORAGE_NPU_BASE the old way.
            if npu_base_env is not None:
                backend_kwargs["storage_npu_base"] = int(npu_base_env)
        else:
            # New path: driver spawns storage and injects the mesh.
            num_vols = forge_cfg.train_world_size
            npu_base = int(npu_base_env) if npu_base_env is not None else num_vols
            storage_mesh = await _spawn_storage_mesh(
                host_mesh_name=storage_host_name,
                num_vols=num_vols,
                npu_base=npu_base,
            )
            backend_kwargs["storage_mesh"] = storage_mesh

        # Bcast-specific extras: which vol is the src, optional master
        # port override for the HCCL TCPStore. Defaults are fine.
        if backend_name == "collective_broadcast":
            src_env = os.environ.get("FORGE_BCAST_SRC_VOL_IDX")
            port_env = os.environ.get("FORGE_BCAST_MASTER_PORT")
            if src_env is not None:
                backend_kwargs["bcast_src_vol_idx"] = int(src_env)
            if port_env is not None:
                backend_kwargs["bcast_master_port"] = int(port_env)

    try:
        backend = create_backend(backend_name, **backend_kwargs)
    except Exception as e:
        print(
            f"[WeightSync] failed to create backend {backend_name!r}: "
            f"{type(e).__name__}: {e}.  Disabling sync.",
            flush=True,
        )
        import traceback

        traceback.print_exc()
        return None

    # Layout comes from ``forge_cfg`` (filled by the config bridge from
    # ``allocation_mode``).  Env-var overrides are still honored for
    # ad-hoc experiments, but should not be needed in steady state --
    # flipping ``allocation_mode`` in YAML is now the canonical knob.
    layout = ParallelLayout(
        train_world=forge_cfg.train_world_size,
        gen_world=forge_cfg.gen_world_size,
        gen_tp=int(os.environ.get("FORGE_GEN_TP") or forge_cfg.gen_tp_size or 1),
        gen_pp=int(os.environ.get("FORGE_GEN_PP") or forge_cfg.gen_pp_size or 1),
        ps_world=int(os.environ.get("FORGE_PS_WORLD", "0")),
        trainer_mesh_name="trainer",
        generator_mesh_name="generator",
        ps_mesh_name="ps",
    )

    service = WeightSyncService(
        backend=backend,
        trainer_actor=trainer,
        generator_actor=generator,
        layout=layout,
        config={"forge_cfg": forge_cfg},
    )
    try:
        await service.initialize()
        print(
            f"[WeightSync] initialized: method=torchstore, backend={backend_name}, "
            f"layout=train={layout.train_world} gen={layout.gen_world} "
            f"tp={layout.gen_tp}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[WeightSync] service init failed: {type(e).__name__}: {e}. "
            "Continuing without sync.",
            flush=True,
        )
        import traceback

        traceback.print_exc()
        return None
    return service


async def _create_legacy_weight_sync(forge_cfg, trainer, generator, method_str):
    """Old path: direct WeightSyncStrategy factory (nccl/checkpoint/hixl)."""
    from forge.core.weight_sync import WeightSyncConfig, WeightSyncMethod
    from forge.engines.weight_sync import create_weight_sync

    method = WeightSyncMethod(method_str)
    config = WeightSyncConfig(
        method=method,
        checkpoint_dir=os.path.join(forge_cfg.fileroot, "weight_sync"),
        master_addr=forge_cfg.master_addr,
        master_port=forge_cfg.master_port + 100,
        world_size=forge_cfg.train_world_size + forge_cfg.gen_world_size,
        rank_offset=forge_cfg.train_world_size,
        backend="hccl" if os.environ.get("ASCEND_VISIBLE_DEVICES") else "nccl",
    )
    strategy = create_weight_sync(method_str)
    try:
        await strategy.initialize(trainer, generator, config)
        print(f"[WeightSync] initialized (legacy): method={method_str}", flush=True)
    except Exception as e:
        print(
            f"[WeightSync] legacy init failed: {type(e).__name__}: {e}. "
            "Continuing without sync.",
            flush=True,
        )
        import traceback

        traceback.print_exc()
        return None
    return strategy


def _load_dataset_prompts(dataset_path: str, split: str = "train"):
    """Load prompts and answers from a HuggingFace dataset."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_path, "main", split=split)
        items = []
        for row in ds:
            q = row.get("question", row.get("problem", row.get("prompt", "")))
            a = row.get("answer", row.get("solution", ""))
            items.append({"prompt": q, "answer": a})
        return items
    except Exception as e:
        logger.warning(f"Failed to load dataset {dataset_path}: {e}. Using dummy data.")
        return [
            {"prompt": f"What is {i} + {i * 2}?", "answer": str(i * 3)}
            for i in range(100)
        ]


def _apply_chat_template(prompt: str, tokenizer) -> str:
    """Format prompt with chat template if available."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt


async def _engine_training_loop(
    trainer,
    generator,
    reward,
    batch_adapter,
    weight_sync,
    max_steps: int,
    start_step: int = 0,
    forge_cfg=None,
    raw_cfg=None,
):
    """Training loop for TrainEngine backends (TorchTitan, FSDP).

    Full pipeline: dataset → tokenize → generate → reward → adapt → train → sync.
    """
    import time

    dataset_path = "openai/gsm8k"
    batch_size = 8
    n_samples = 4
    max_new_tokens = 1024

    if raw_cfg is not None:
        ds_cfg = raw_cfg.get("train_dataset", {})
        if hasattr(ds_cfg, "path"):
            dataset_path = ds_cfg.path
        elif isinstance(ds_cfg, dict):
            dataset_path = ds_cfg.get("path", dataset_path)
        gen_cfg = raw_cfg.get("gconfig", {})
        if hasattr(gen_cfg, "n_samples"):
            n_samples = gen_cfg.n_samples
        elif isinstance(gen_cfg, dict):
            n_samples = gen_cfg.get("n_samples", n_samples)
        if hasattr(gen_cfg, "max_new_tokens"):
            max_new_tokens = gen_cfg.max_new_tokens
        elif isinstance(gen_cfg, dict):
            max_new_tokens = gen_cfg.get("max_new_tokens", max_new_tokens)

    print(f"[Train-Engine] Loading dataset: {dataset_path}", flush=True)
    dataset_items = _load_dataset_prompts(dataset_path)
    print(f"[Train-Engine] Loaded {len(dataset_items)} prompts", flush=True)

    tokenizer = None
    model_path = forge_cfg.model_path if forge_cfg else ""
    if model_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            print(f"[Train-Engine] Tokenizer loaded from {model_path}", flush=True)
        except Exception as e:
            logger.warning(f"[Train-Engine] Tokenizer load failed: {e}")

    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        n=n_samples,
        logprobs=1,
    )

    prompt_idx = 0
    step = start_step
    while step < max_steps:
        step_start = time.time()
        print(f"[Train-Engine] Step {step}/{max_steps}", flush=True)

        prompts_with_answers = []
        for _ in range(batch_size):
            item = dataset_items[prompt_idx % len(dataset_items)]
            prompt_idx += 1
            prompts_with_answers.append(item)

        rollout_start = time.time()
        episodes = []
        total_reward = 0.0

        for item in prompts_with_answers:
            formatted_prompt = _apply_chat_template(item["prompt"], tokenizer)

            gen_mesh = await generator.generate.call(
                formatted_prompt,
                sampling_params=sampling_params,
            )
            _, gen_results = next(iter(gen_mesh.items()))
            if not isinstance(gen_results, list):
                gen_results = [gen_results]

            for gen_result in gen_results:
                text = gen_result.get("text", "")
                token_ids = gen_result.get("token_ids", [])
                logprobs = gen_result.get("logprobs", [])
                if not isinstance(logprobs, list):
                    logprobs = [0.0] * len(token_ids)
                version = gen_result.get("generator_version", -1)

                r = 0.0
                try:
                    reward_mesh = await reward.compute_reward.call(
                        prompt=item["prompt"],
                        completion=text,
                        task_data={"answer": item.get("answer", "")},
                    )
                    _, r = next(iter(reward_mesh.items()))
                except Exception as e:
                    logger.debug(f"Reward failed: {e}")

                total_reward += float(r)
                episodes.append(
                    Episode(
                        episode_id=f"ep_{step}_{len(episodes)}",
                        prompt=item["prompt"],
                        response=text,
                        reward=float(r),
                        policy_version=version,
                        token_ids=token_ids,
                        generator_logprobs=logprobs,
                        loss_mask=[1] * len(token_ids),
                        versions=[version] * len(token_ids),
                    )
                )

        rollout_time = time.time() - rollout_start
        avg_reward = total_reward / max(len(episodes), 1)
        print(
            f"[Train-Engine] Step {step} rollout: {len(episodes)} episodes, "
            f"reward={avg_reward:.4f}, time={rollout_time:.1f}s",
            flush=True,
        )

        batch = batch_adapter.adapt(episodes)

        train_start = time.time()
        result_mesh = await trainer.train_on_engine_batch.call(batch, step)
        _, result = next(iter(result_mesh.items()))
        train_time = time.time() - train_start

        if weight_sync is not None:
            sync_start = time.time()
            try:
                await weight_sync.push(step + 1)
                sync_time = time.time() - sync_start
                print(f"[Train-Engine] Weight sync: {sync_time:.1f}s", flush=True)
            except Exception as e:
                logger.warning(f"[Train-Engine] Weight sync failed: {e}")

        step_time = time.time() - step_start
        print(
            f"[Train-Engine] Step {step} complete: {result}, "
            f"reward={avg_reward:.4f}, time={step_time:.1f}s "
            f"(rollout={rollout_time:.1f}, train={train_time:.1f})",
            flush=True,
        )
        step += 1


async def grpo_main(
    config=None,
    run_id: int = 0,
    provisioner_config=None,
    backend_override: str | None = None,
    titan_args: dict | None = None,
):
    """Async GRPO training orchestration with Forge actors.

    Architecture::

        Generator.as_actor()     -- inference (vLLM / SGLang / ...)
        RewardActor.as_actor()   -- reward computation
        TrainerActor.as_actor()  -- training (AReaL / Slime / TorchTitan / ...)

        Training loop:
            trainer.train_step(step) -> internal rollout + train
            -> weight sync to Generator
    """
    ensure_ascend_custom_opp_path()

    bridge = create_config_bridge(backend="areal")
    forge_cfg, raw_cfg, alloc_mode = bridge.parse_and_build(run_id=run_id)

    bridge.setup_name_resolve(raw_cfg)

    is_recover_run = forge_cfg.backend_config.get("is_recover_run", False)
    if not is_recover_run:
        bridge.save_metadata(raw_cfg)

    os.makedirs(forge_cfg.resolve_log_dir(), exist_ok=True)

    if backend_override:
        forge_cfg.backend_type = backend_override
        if backend_override == "titan":
            ta = titan_args or {}
            forge_cfg.backend_config = {
                "model_name": ta.get("model_name", "qwen3"),
                "model_flavor": ta.get("model_flavor", "1.7B"),
                "hf_model_path": forge_cfg.model_path,
                "max_steps": forge_cfg.backend_config.get(
                    "max_steps",
                    raw_cfg.get("total_train_steps", 100)
                    if hasattr(raw_cfg, "get")
                    else 100,
                ),
                "loss_type": "grpo",
                "lr": 1.7e-5,
                "dtype": "bfloat16",
                "seq_len": 4096,
                "local_batch_size": 4,
                "dp_shard": -1,
                "tp": 1,
                "pp": 1,
                "loss_config": {"beta": 0.0},
            }

    print(
        f"GRPO: experiment={forge_cfg.experiment_name}, "
        f"trial={forge_cfg.trial_name}, run_id={run_id}, "
        f"backend={forge_cfg.backend_type}"
    )

    if bridge.is_llm_server_only(alloc_mode):
        logger.info("LLM_SERVER_ONLY mode -- serving only, no training")
        return

    await init_provisioner(provisioner_config)

    is_distributed = provisioner_config is not None and (
        provisioner_config.launcher_config is not None
        and provisioner_config.launcher_config.launcher.value != "local"
    )

    # Generator topology is driven entirely by ``forge_cfg`` (which is
    # filled from ``allocation_mode`` by the config bridge).  Flipping
    # the YAML ``allocation_mode`` is the single source of truth:
    #   vllm:d1p1t1  -> procs=1, gpus_per_proc=1  (legacy default)
    #   vllm:d1p1t4  -> procs=1, gpus_per_proc=4  (one vLLM engine, TP=4)
    #   vllm:d4p1t1  -> procs=4, gpus_per_proc=1  (4 independent vLLM replicas)
    #   vllm:d2p1t2  -> procs=2, gpus_per_proc=2  (mixed TP+DP)
    # ``engine_args.tensor_parallel_size`` was already resolved to
    # ``gen_tp_size * gen_pp_size`` by the bridge, so vLLM's internal
    # worker-spawn matches the device mask we install here.
    gen_procs = max(1, forge_cfg.gen_dp_size)
    gen_gpus_per_proc = max(1, forge_cfg.gen_tp_size * forge_cfg.gen_pp_size)
    generator = await Generator.options(
        procs=gen_procs,
        gpus_per_proc=gen_gpus_per_proc,
        with_gpus=True,
        mesh_name="generator",
        hosts=1 if is_distributed else None,
    ).as_actor(
        engine_args=forge_cfg.engine_args,
    )

    use_rm_gpu = forge_cfg.reward_mode in ("model", "hybrid")
    reward = await RewardActor.options(
        procs=1, with_gpus=use_rm_gpu, mesh_name="reward"
    ).as_actor()
    if forge_cfg.reward_fn_path:
        await reward.setup.call(forge_cfg.reward_fn_path)
    if forge_cfg.reward_model_path:
        await reward.setup_model.call(
            model_path=forge_cfg.reward_model_path,
            device=forge_cfg.reward_model_device,
            dtype=forge_cfg.reward_model_dtype,
            max_batch_size=forge_cfg.reward_model_max_batch_size,
            max_length=forge_cfg.reward_model_max_length,
        )
    if forge_cfg.reward_mode != "rule":
        await reward.set_mode.call(
            mode=forge_cfg.reward_mode,
            rule_weight=forge_cfg.reward_rule_weight,
            model_weight=forge_cfg.reward_model_weight,
        )

    use_engine = forge_cfg.backend_type != "areal"
    weight_sync = None

    if use_engine:
        engine = create_engine(
            backend=forge_cfg.backend_type,
            config={
                "model_path": forge_cfg.model_path,
                "max_steps": forge_cfg.backend_config.get("max_steps", 100),
                **forge_cfg.backend_config,
            },
        )
        trainer = await TrainerActor.options(
            procs=forge_cfg.train_world_size, with_gpus=True, mesh_name="trainer"
        ).as_actor(engine=engine)
    else:
        xccl_alloc = bridge.resolve_xccl_alloc_mode(
            raw_cfg, alloc_mode, train_world_size=forge_cfg.train_world_size
        )
        backend = create_engine(
            backend=forge_cfg.backend_type,
            cli_args=forge_cfg.training_args,
            env_vars=forge_cfg.trainer_env,
            rank=-1,
            world_size=forge_cfg.train_world_size,
            master_addr=forge_cfg.master_addr,
            master_port=forge_cfg.master_port,
            generator_actor=generator,
            reward_actor=reward,
            agent_actor=None,
            xccl_weight_update_alloc_mode=xccl_alloc,
        )
        trainer = await TrainerActor.options(
            procs=forge_cfg.train_world_size, with_gpus=True, mesh_name="trainer"
        ).as_actor(backend=backend)

    info_mesh = await trainer.initialize.call()
    _, info = next(iter(info_mesh.items()))
    max_steps = info.get("max_steps", 0)
    start_step = info.get("start_step", 0)
    print(
        f"[GRPO] Trainer ready: max_steps={max_steps}, start={start_step}, backend={forge_cfg.backend_type}",
        flush=True,
    )

    # Titan historically managed its own weight sync, so it was excluded here.
    # The torchstore strategy is backend-agnostic (it only needs
    # ``state_dict_for_sync()`` on the engine side), so the titan path now
    # opts in when ``FORGE_WEIGHT_SYNC=torchstore``.
    if use_engine and weight_sync is None:
        method_str = os.environ.get("FORGE_WEIGHT_SYNC", "nccl")
        titan_allowed = method_str == "torchstore"
        if forge_cfg.backend_type not in ("titan",) or titan_allowed:
            weight_sync = await _create_weight_sync(forge_cfg, trainer, generator)

    step = start_step
    try:
        if use_engine:
            print(
                f"[GRPO] Starting engine training loop, max_steps={max_steps}",
                flush=True,
            )
            batch_adapter = create_batch_adapter(backend=forge_cfg.backend_type)
            await _engine_training_loop(
                trainer,
                generator,
                reward,
                batch_adapter,
                weight_sync,
                max_steps=max_steps,
                start_step=start_step,
                forge_cfg=forge_cfg,
                raw_cfg=raw_cfg,
            )
            print("[GRPO] Engine training loop completed", flush=True)
            step = max_steps
        else:
            while step < max_steps:
                logger.info(f"[Train] Step {step}/{max_steps}")
                result_mesh = await trainer.train_step.call(step)
                results = list(result_mesh.items())
                _, result = results[0]
                logger.info(f"[Train] Step {step} complete: {result}")
                step += 1
    except Exception as e:
        print(f"[GRPO] FAILED at step {step}: {type(e).__name__}: {e}", flush=True)
        import traceback

        traceback.print_exc()
        raise
    finally:
        if weight_sync is not None:
            await weight_sync.shutdown()
        logger.info("Shutting down all actors...")
        await shutdown()

    logger.info(f"GRPO training complete. Total steps: {step}")


def _load_yaml_launcher_block(
    forge_config_path: str | None,
    areal_config_path: str | None,
) -> tuple[dict, str | None]:
    """Eagerly extract the ``launcher:`` block from YAML.

    We read the ``launcher:`` block ourselves rather than going through
    AReaL's ``config_bridge.parse_and_build`` because Monarch transport
    configuration (``configure(TcpWithHostname)``) and the
    ``ProvisionerConfig`` we hand to ``grpo_main`` both have to be set
    up **before** any actor is created, while the AReaL config bridge
    fires inside ``grpo_main``.  The ``launcher:`` block is small,
    self-contained, and doesn't use cross-section Hydra interpolation,
    so a standalone read is safe.

    Two YAML paths are searched in priority order:

    1. ``--forge-config FILE`` -- a forge-specific YAML dedicated to
       launcher topology.  Recommended: keeps AReaL's experiment YAMLs
       clean so the same AReaL recipe can be reused with different
       launcher configs (local / bare_metal / slurm).
    2. ``--config FILE`` -- AReaL's experiment YAML.  If it happens
       to carry a top-level ``launcher:`` key we pick that up too,
       which lets users inline launcher config in a single file when
       they don't care about the separation.

    Returns ``(block_dict, source_path)`` where ``source_path`` is the
    file the block came from (for logging), or ``({}, None)`` if no
    usable block was found.  Callers layer CLI / env overrides on top.
    """
    from pathlib import Path

    for source_label, p in (
        ("--forge-config", forge_config_path),
        ("--config", areal_config_path),
    ):
        if not p:
            continue
        path = Path(p)
        if not path.is_file():
            continue
        try:
            from omegaconf import OmegaConf

            raw = OmegaConf.load(str(path))
            block = raw.get("launcher", None)
            if block is None:
                continue
            return (
                OmegaConf.to_container(block, resolve=True),
                f"{source_label}={p}",
            )
        except Exception as e:
            print(
                f"[WARN] _load_yaml_launcher_block: failed to parse "
                f"{source_label}={p!r}: {e}",
                flush=True,
            )
            continue
    return {}, None


def _find_cli_path(argv: list[str], flag: str) -> str | None:
    """Return the value of ``--flag VALUE`` or ``--flag=VALUE`` in argv."""
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def main():
    import sys

    from monarch._src.actor.actor_mesh import context

    provisioner_config = None

    bare_metal_args = {}
    mesh_placement: dict[str, int] = {}
    backend_override = None
    titan_args = {}
    remaining_argv = []
    argv = sys.argv[1:]

    # Read YAML launcher: block up front.  CLI flags and env vars
    # will override YAML values where they conflict (priority:
    # CLI > env > YAML > dataclass defaults) -- this keeps
    # ``run_multinode.sh`` able to inject dynamic values
    # (e.g. hostfile-derived workers list) on top of a checked-in
    # YAML that describes the canonical topology.
    #
    # --forge-config takes priority; falling back to --config lets
    # a single-file YAML (launcher + AReaL recipe merged) work too.
    forge_config_path = _find_cli_path(argv, "--forge-config")
    areal_config_path = _find_cli_path(argv, "--config")
    yaml_launcher, yaml_launcher_source = _load_yaml_launcher_block(
        forge_config_path, areal_config_path
    )
    if yaml_launcher:
        print(
            f"[launcher] loaded launcher block from {yaml_launcher_source}",
            flush=True,
        )
    i = 0
    while i < len(argv):
        if argv[i] == "--forge-config":
            # Consumed by _load_yaml_launcher_block via
            # ``_find_cli_path(argv, "--forge-config")`` above; we
            # strip it from the argv handed to AReaL's config_bridge
            # so argparse doesn't choke on the unknown flag.
            i += 2
        elif argv[i] == "--bare-metal-workers":
            bare_metal_args["workers"] = argv[i + 1].split(",")
            i += 2
        elif argv[i] == "--bare-metal-master-addr":
            bare_metal_args["master_addr"] = argv[i + 1]
            i += 2
        elif argv[i] == "--bare-metal-worker-port":
            bare_metal_args["worker_port"] = int(argv[i + 1])
            i += 2
        elif argv[i] == "--mesh-placement":
            # Parse: "trainer=0,generator=1,storage=0" into
            # {"trainer": 0, "generator": 1, "storage": 0}.
            # Maps logical mesh names to worker indices in the bare-metal
            # launcher's worker array.  Future launchers (slurm, k8s)
            # will accept the same --mesh-placement form but interpret
            # the right-hand side as a mesh / node selector, not an
            # index.  Keep this parser dumb and string-typed so
            # alternate forms (JSON dict, scheme prefix, etc.) can slot
            # in without restructuring the CLI.
            for item in argv[i + 1].split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" not in item:
                    raise ValueError(
                        f"--mesh-placement expects 'name=worker_idx' "
                        f"pairs, got {item!r}"
                    )
                name, v = item.split("=", 1)
                mesh_placement[name.strip()] = int(v.strip())
            i += 2
        elif argv[i] == "--backend":
            backend_override = argv[i + 1]
            i += 2
        elif argv[i] == "--model-name":
            titan_args["model_name"] = argv[i + 1]
            i += 2
        elif argv[i] == "--model-flavor":
            titan_args["model_flavor"] = argv[i + 1]
            i += 2
        else:
            remaining_argv.append(argv[i])
            i += 1

    # Env-var fallback for mesh placement so run_multinode.sh can pass
    # this through without tweaking the command line.  Parsed identically
    # to --mesh-placement.
    env_placement = os.environ.get("FORGE_MESH_PLACEMENT", "").strip()
    if env_placement and not mesh_placement:
        for item in env_placement.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                name, v = item.split("=", 1)
                mesh_placement[name.strip()] = int(v.strip())

    # --- Merge YAML launcher: block (lowest precedence) -----------------
    #
    # YAML acts as the baseline "canonical topology" -- CLI / env can
    # override individual fields for one-off experiments without
    # editing the YAML.  The block schema (minimum viable, extend as
    # new launcher types come online):
    #
    #     launcher:
    #       type: bare_metal            # future: slurm | k8s | local
    #       bare_metal:
    #         workers: [tcp://..., ...]
    #         master_addr: ...
    #         worker_port: 22222
    #       meshes:
    #         trainer:   {host_idx: 0}
    #         generator: {host_idx: 1}
    #         storage:   {host_idx: 0}
    #       weight_sync:                # optional, consumed by forge driver
    #         backend: torchstore_multi_vol
    #         storage_mesh: storage     # -> meshes.storage above
    #         pool_mb: 8192
    #         storage_npu_base: null    # null = auto (train_world_size)
    if yaml_launcher:
        yaml_bare_metal = yaml_launcher.get("bare_metal") or {}
        if not bare_metal_args.get("workers") and yaml_bare_metal.get("workers"):
            bare_metal_args["workers"] = list(yaml_bare_metal["workers"])
        if not bare_metal_args.get("master_addr") and yaml_bare_metal.get(
            "master_addr"
        ):
            bare_metal_args["master_addr"] = yaml_bare_metal["master_addr"]
        if (
            "worker_port" not in bare_metal_args
            and yaml_bare_metal.get("worker_port") is not None
        ):
            bare_metal_args["worker_port"] = int(yaml_bare_metal["worker_port"])

        yaml_meshes = yaml_launcher.get("meshes") or {}
        for name, spec in yaml_meshes.items():
            if name in mesh_placement:
                continue  # CLI/env already decided this one
            if isinstance(spec, dict) and "host_idx" in spec:
                mesh_placement[name] = int(spec["host_idx"])
            elif isinstance(spec, int):
                mesh_placement[name] = int(spec)

        # weight_sync -> env vars (the downstream readers in
        # _create_weight_sync_service / _spawn_storage_mesh are still
        # env-driven after Step 2; pushing YAML into env keeps a
        # single read path).  Existing env takes precedence so an
        # operator shell setting still wins over YAML.
        yaml_ws = yaml_launcher.get("weight_sync") or {}
        _env_fallback = {
            "FORGE_WEIGHT_SYNC_BACKEND": yaml_ws.get("backend"),
            "FORGE_STORAGE_HOST_MESH": yaml_ws.get("storage_mesh"),
            "TORCHSTORE_MONARCH_RDMA_POOL_MB": yaml_ws.get("pool_mb"),
            "TORCHSTORE_STORAGE_NPU_BASE": yaml_ws.get("storage_npu_base"),
        }
        for env_name, val in _env_fallback.items():
            if val is None:
                continue
            if not os.environ.get(env_name):
                os.environ[env_name] = str(val)

    if bare_metal_args:
        from forge.types import Launcher, LauncherConfig, ProvisionerConfig

        provisioner_config = ProvisionerConfig(
            launcher_config=LauncherConfig(
                launcher=Launcher.BARE_METAL,
                meshes=mesh_placement,
                **bare_metal_args,
            )
        )
        from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
        from monarch._src.actor.actor_mesh import configure

        configure(default_transport=ChannelTransport.TcpWithHostname)

    sys.argv = [sys.argv[0]] + remaining_argv

    context()

    asyncio.run(
        grpo_main(
            run_id=0,
            provisioner_config=provisioner_config,
            backend_override=backend_override,
            titan_args=titan_args,
        )
    )


if __name__ == "__main__":
    main()
