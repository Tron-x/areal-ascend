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


def _storage_bootstrap(dev_id: int):
    """Bootstrap for a single-NPU torchstore storage proc.

    The storage volume lives in NPU device memory so HiXL RDMA can register
    it and both trainer and generator engines can do one-sided put/get
    straight into/out of it.  This is the only supported storage topology:
    RoCE across hosts, 2 MB-aligned NPU buffers, single staging pool per
    storage volume.

    ``dev_id`` selects the physical NPU (typically an idle one — e.g. 7 —
    so the storage HiXL engine doesn't share an HCCL port range with the
    trainer's FSDP comm group or the generator's vLLM comm group).
    """
    import os as _os

    def _bootstrap():
        _os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        _os.environ["MONARCH_NPU_DEVICE"] = "0"
        _os.environ["MONARCH_HIXL_TRANSPORT"] = "roce"
        _os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        _os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        _os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        _os.environ["TORCHSTORE_MONARCH_RDMA_EAGER_D2H"] = "0"
        _os.environ["TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE"] = "npu:0"
        _os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_POOL_MB", "8192")
        _os.environ.setdefault("LOCAL_RANK", "0")
        _os.environ.setdefault("RANK", "0")
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


async def _spawn_storage_mesh(dev_id: int, mesh_name: str = "generator"):
    """Spawn a single torchstore StorageVolume proc on the generator host.

    Topology (phase-1 goal: measure cross-node HiXL RoCE bandwidth):

        trainer NPU 0-3 (node 23)  ──HiXL RoCE write──►  storage (node 26 NPU `dev_id`)
                                                                        │
        generator NPU 0 (node 26)  ◄──intra-node HiXL read──────────────┘

    We deliberately put storage on the **generator** host (not the trainer
    host) for two reasons:

    1. The trainer host runs FSDP's HCCL comm group on NPUs 0-3.  Adding a
       HiXL engine on an NPU on the same host forces intra-node HiXL
       Connect(trainer_npu → storage_npu) which races with the FSDP PG at
       the CANN RA driver level and surfaces as ``RaHdcTypicalMrReg ret=-13
       / HcclCommPrepare 0x13 / hixl_connect 503900``.
    2. Putting storage next to the generator makes the heavy trainer→storage
       path pure **cross-node RoCE** -- which is exactly the bandwidth we
       want phase-1 to measure -- and keeps the intra-node generator→storage
       hop short and collision-free (generator uses NPU 0, storage uses an
       idle NPU e.g. 7, so their HiXL engines don't share an HCCL port
       range with anything else).

    One storage volume, only trainer rank 0 pushes, generator pulls the
    full state_dict.  HiXL handshake matrix: 4 cross-node pairs
    (trainer_rank_i ↔ storage) + 1 intra-node pair (generator ↔ storage).

    Args:
        dev_id: physical NPU ID to pin the storage proc to (typically 7 -
            an idle card on the generator host).
        mesh_name: host-mesh name to colocate with (defaults to ``generator``).

    Returns a Monarch ``ProcMesh`` suitable for ``ts.initialize(mesh=...)``,
    or ``None`` on failure (caller should skip torchstore sync).
    """
    try:
        from forge.provisioner import _get_provisioner

        provisioner = await _get_provisioner()
        trainer_hosts = await provisioner.get_host_mesh(mesh_name)
        storage_mesh = trainer_hosts.spawn_procs(
            per_host={"procs": 1},
            name="torchstore_storage",
            bootstrap=_storage_bootstrap(dev_id),
        )
        print(
            f"[WeightSync] spawned 1 NPU storage proc on trainer host mesh "
            f"'{mesh_name}' pinned to NPU {dev_id} (HiXL/RoCE, mesh={storage_mesh}).",
            flush=True,
        )
        return storage_mesh
    except Exception as e:
        print(
            f"[WeightSync] failed to spawn storage mesh on trainer host: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        import traceback

        traceback.print_exc()
        return None


async def _create_weight_sync(forge_cfg, trainer, generator):
    """Create and initialize a WeightSyncStrategy for TrainEngine backends.

    Selection is via ``FORGE_WEIGHT_SYNC`` env var:
        - ``"nccl"`` (default): HCCL/NCCL collective broadcast
        - ``"checkpoint"``: save+reload via disk
        - ``"hixl"``: one-sided HiXL RDMA (stub)
        - ``"torchstore"``: torchstore + Monarch RDMA (HiXL on NPU).  Reuses
          the generator actor's proc_mesh as the storage volume mesh so the
          StorageVolume lives on the same host (and HiXL engine pair) as
          the generator and can leverage the shared Monarch-RDMA staging pool.
    """
    from forge.core.weight_sync import WeightSyncConfig, WeightSyncMethod
    from forge.engines.weight_sync import create_weight_sync

    method_str = os.environ.get("FORGE_WEIGHT_SYNC", "nccl")
    method = WeightSyncMethod(method_str)

    extra: dict | None = None
    if method == WeightSyncMethod.TORCHSTORE:
        # Single NPU storage proc on the trainer host: rank 0 pushes, generator
        # pulls, both via HiXL RoCE one-sided RDMA into/out of the storage's
        # NPU device memory.  Pin it to an idle NPU (default 7) so it doesn't
        # share an HCCL port range with the trainer's FSDP comm group.
        storage_npu = int(os.environ.get("TORCHSTORE_STORAGE_NPU", "7"))
        storage_host = os.environ.get("TORCHSTORE_STORAGE_HOST_MESH", "generator")
        storage_mesh = await _spawn_storage_mesh(
            dev_id=storage_npu, mesh_name=storage_host
        )
        if storage_mesh is None:
            print(
                "[WeightSync] FORGE_WEIGHT_SYNC=torchstore requested but "
                f"could not spawn storage proc on host mesh '{storage_host}' "
                "(see logs above). Disabling sync.",
                flush=True,
            )
            return None
        extra = {
            "storage_mesh": storage_mesh,
            "num_storage_volumes": 1,
        }
        print(
            f"[WeightSync] method=torchstore; 1 NPU storage proc on "
            f"'{storage_host}' host NPU {storage_npu} ({storage_mesh}).",
            flush=True,
        )

    config = WeightSyncConfig(
        method=method,
        checkpoint_dir=os.path.join(forge_cfg.fileroot, "weight_sync"),
        master_addr=forge_cfg.master_addr,
        master_port=forge_cfg.master_port + 100,
        world_size=forge_cfg.train_world_size + forge_cfg.gen_world_size,
        rank_offset=forge_cfg.train_world_size,
        backend="hccl" if os.environ.get("ASCEND_VISIBLE_DEVICES") else "nccl",
        extra=extra,
    )

    strategy = create_weight_sync(method_str)
    try:
        await strategy.initialize(trainer, generator, config)
        print(f"[WeightSync] initialized: method={method_str}", flush=True)
    except Exception as e:
        print(
            f"[WeightSync] init failed: {type(e).__name__}: {e}. "
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

    generator = await Generator.options(
        procs=1,
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


def main():
    import sys

    from monarch._src.actor.actor_mesh import context

    provisioner_config = None

    bare_metal_args = {}
    backend_override = None
    titan_args = {}
    remaining_argv = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--bare-metal-workers":
            bare_metal_args["workers"] = argv[i + 1].split(",")
            i += 2
        elif argv[i] == "--bare-metal-master-addr":
            bare_metal_args["master_addr"] = argv[i + 1]
            i += 2
        elif argv[i] == "--bare-metal-worker-port":
            bare_metal_args["worker_port"] = int(argv[i + 1])
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

    if bare_metal_args:
        from forge.types import Launcher, LauncherConfig, ProvisionerConfig

        provisioner_config = ProvisionerConfig(
            launcher_config=LauncherConfig(
                launcher=Launcher.BARE_METAL,
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
