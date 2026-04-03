"""AReaL Monarch Plugin -- TorchForge-style actor framework for distributed RL.

Architecture layers:
  - ``controller/``: AReaLForgeActor base, Provisioner, Service layer
  - ``actors/``: Generator, Trainer, Reward, Agent, Sandbox, ReplayBuffer
  - ``apps/``: Entry points (GRPO, Agent RL)
  - ``types.py``: Core dataclasses (ProcessConfig, ServiceConfig, TrainBatch)
  - ``weight_sync_v2.py``: WeightStore abstraction (Disk, XCCL)

Quick start::

    # GRPO training
    python -m areal.monarch_plugin.apps.grpo.main \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml

    # Agentic RL training
    python -m areal.monarch_plugin.apps.agent_rl.main \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_agent.yaml

Legacy launcher (backward compatible)::

    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml
"""
