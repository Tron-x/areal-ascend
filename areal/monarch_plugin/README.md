# Monarch Plugin for AReaL

基于 [Meta Monarch](https://github.com/pytorch-labs/monarch) 框架的 AReaL 分布式 RL 插件。
采用 TorchForge 架构模式（声明式 Actor + Service + Provisioner），将训练、推理、奖励计算、经验回放等组件拆分为独立的 Monarch Actor，通过 RPC 通信实现全分离架构。支持任意卡数配置、多机部署和 Ascend NPU。

## 架构总览 (v2 - TorchForge Style)

```
Controller (async main)
  ├── Provisioner (GPU allocation, ProcMesh lifecycle, EnvSetter)
  │
  ├── Generator Service (replicated, load-balanced)
  │     ├── Replica 0: AsyncLLM + AReaLMonarchExecutor
  │     │     └── WorkerWrapper x N (tensor parallel, NPU/GPU)
  │     └── Replica 1: AsyncLLM + AReaLMonarchExecutor
  │           └── WorkerWrapper x N
  │
  ├── Trainer Actor (FSDP, multi-GPU/NPU)
  │     ├── FSDPEngine
  │     └── PPOTrainer
  │
  ├── Agent Service (session-affine routing for multi-turn)
  │     └── AgentActor (generate → sandbox → reward loop)
  │
  ├── Sandbox Service (replicated subprocess execution)
  │     └── SandboxActor x N
  │
  ├── Reward Service (replicated, load-balanced)
  │     └── RewardActor x N
  │
  ├── ReplayBuffer Actor
  │     └── Age/count eviction, versioned sampling
  │
  └── ComputeAdvantages Actor
        └── GRPO group-norm or GAE
```

## 新架构目录结构

```
areal/monarch_plugin/
  controller/
    __init__.py              # Re-exports AReaLForgeActor, Provisioner
    actor.py                 # AReaLForgeActor base class (options/as_service/as_actor)
    provisioner.py           # Provisioner + DeviceProxy + GpuManager + EnvSetter
    service/
      __init__.py            # Re-exports Service, ServiceInterface
      service.py             # Service controller (replicas, health loop)
      interface.py           # ServiceInterface + ServiceEndpoint (route/fanout)
      replica.py             # Replica lifecycle (init, recover, stop)
      router.py              # LeastLoaded, RoundRobin, Session routers
      metrics.py             # ServiceMetrics aggregation
  actors/
    __init__.py
    generator/
      __init__.py
      generator.py           # Generator actor (AsyncLLM + MonarchExecutor)
      executor.py            # AReaLMonarchExecutor (Ascend-aware, vLLM Executor)
      worker.py              # WorkerWrapper, WorkerRegistry, FutureWrapper
    trainer.py               # TrainerActor (wraps FSDPEngine cleanly)
    reward.py                # RewardActor + MonarchRewardWrapper
    agent.py                 # AgentActor + MonarchAgentWorkflow
    sandbox.py               # SandboxActor (subprocess code execution)
    replay_buffer.py         # ReplayBuffer with eviction/sampling policies
    advantages.py            # ComputeAdvantages actor
  apps/
    grpo/
      main.py                # GRPO training entry point
    agent_rl/
      main.py                # Agentic RL entry point (multi-turn + tool use)
  types.py                   # ProcessConfig, ServiceConfig, LauncherConfig, TrainBatch
  weight_sync.py             # XCCL weight sync alloc_mode (legacy, still used)
  weight_sync_v2.py          # WeightStore abstraction (Disk / XCCL backends)
  bootstraps.py              # Platform-agnostic bootstrap factories
  monarch_inf_engine.py      # MonarchVLLMEngine (legacy bridge)
  __init__.py
```

## 核心设计原则 (from TorchForge)

1. **声明式 Actor**: `MyActor.options(procs=4, with_gpus=True).as_service()` — 资源在类级别声明
2. **Provisioner 管理资源**: 全局单例跟踪 GPU 分配、ProcMesh 生命周期和关停
3. **Service = 副本 + 健康 + 路由**: 自动故障恢复、负载均衡、会话亲和
4. **Controller 驱动协调**: 异步 `main()` 直接编排 actors — 无需独立的 Pipeline 类
5. **DeviceProxy 硬件抽象**: CUDA、NPU、XPU 一套代码路径

## 快速开始

### GRPO 训练 (新入口)

```bash
python -m areal.monarch_plugin.apps.grpo.main \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "allocation_mode=vllm:d4p1t1+d4p1t1" \
    "+total_train_steps=2"
```

### Agentic RL 训练 (新入口)

```bash
python -m areal.monarch_plugin.apps.agent_rl.main \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_agent.yaml \
    "++enable_thinking=true"
```

### Legacy 入口 (backward compatible)

```bash
python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "+total_train_steps=2"
```

## 支持的配置

### 单机

| 配置 | allocation_mode | NPU 数 | 说明 |
|------|----------------|--------|------|
| 1+1 | `vllm:d1p1t1+d1p1t1` | 2 | 最小验证配置 |
| 2+2 | `vllm:d2p1t1+d2p1t1` | 4 | 2 卡推理 + 2 卡训练 |
| 4+4 | `vllm:d4p1t1+d4p1t1` | 8 | 默认 8 卡配置 |
| 1(TP=4)+4 | `vllm:d1p1t4+d4p1t1` | 8 | 推理用 TP 并行 |

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.11+ |
| PyTorch | 2.9.0+ |
| torch_npu | 2.9.0+ (Ascend) |
| vLLM | 0.14.0+ |
| CANN | 9.0.0+ |
| Monarch | 从源码编译 |
| 芯片 | Ascend 910B / NVIDIA GPU |

## 关键设计决策

**为什么采用 TorchForge 架构？**
- 声明式 Actor 配置替代手动 ProcMesh 管理
- Service 层提供健康监控、负载均衡和故障恢复
- Provisioner 全局管理 GPU 分配，避免资源竞争
- SessionRouter 保证多轮对话路由到同一 Generator 副本（KV cache 局部性）

**为什么保留 MonarchVLLMEngine？**
- 作为 AReaL PPOTrainer 和新 Generator 之间的桥接层
- 确保现有 FSDPEngine 训练循环无需修改
- 后续可逐步迁移到纯 Monarch RPC 通信

**WeightStore 抽象为什么重要？**
- TorchStore 在 Ascend 环境可能不可用
- DiskWeightStore 提供可靠的 fallback
- XCCLWeightStore 利用 HCCL/NCCL 高带宽
- 统一接口使后续切换透明
