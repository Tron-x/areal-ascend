# Monarch Plugin for AReaL

基于 [Meta Monarch](https://github.com/pytorch-labs/monarch) 框架的 AReaL 分布式 RL 插件，将训练、推理、奖励计算、经验回放等组件拆分为独立的 Monarch Actor，通过 RPC 通信实现全分离架构。

## 架构总览

```
MonarchOrchestrator (主进程, 无 NPU)
  ├── GeneratorProcMesh (NPU 0)
  │     └── GeneratorActor — vLLM AsyncLLM 推理引擎
  ├── RewardProcMesh (CPU)
  │     └── RewardActor — reward 函数计算
  ├── SandboxProcMesh (CPU)
  │     └── SandboxActor — 隔离 Python 代码执行
  ├── AgentProcMesh (CPU)
  │     └── AgentActor — 多轮 Agent 编排 (Generator + Sandbox + Reward)
  ├── ReplayBufferProcMesh (CPU)
  │     └── ReplayBufferActor — 异步经验回放缓冲
  ├── RolloutProcMesh (CPU)
  │     └── RolloutActor — 独立 rollout 生产 (自带 dataloader, 不加载模型)
  └── TrainingProcMesh (NPU 1)
        └── TrainerActor — FSDP 训练 (仅消费 batch, 不做 rollout)
```

所有 Actor 间通过 **Monarch RPC** 通信，无 HTTP 依赖。

## 文件说明

| 文件 | 说明 |
|------|------|
| `launcher.py` | 主入口，负责创建 ProcMesh、spawn Actor、编排训练流程 |
| `generator_actor.py` | 封装 vLLM `AsyncLLM`，提供 `/v1/completions` 等推理端点 |
| `executor.py` | `AReaLMonarchExecutor` — 自定义 vLLM Executor，在 Monarch Worker ProcMesh 中运行 |
| `monarch_inf_engine.py` | `MonarchVLLMEngine` — 替换 AReaL 的 `RemotevLLMEngine`，通过 Monarch RPC 路由请求 |
| `reward_actor.py` | `RewardActor` + `MonarchRewardWrapper`，CPU 上运行 reward 计算 |
| `sandbox_actor.py` | `SandboxActor` — 隔离子进程执行 Python 代码（带超时） |
| `agent_actor.py` | `AgentActor` + `MonarchAgentWorkflow` — 多轮 Agent 交互编排 |
| `replay_buffer_actor.py` | `ReplayBufferActor` — 异步经验缓冲，支持版本感知的过期淘汰 |
| `rollout_actor.py` | `RolloutActor` — 独立 rollout 生产，拥有自己的 dataloader 和 WorkflowExecutor |
| `scripts/run_1x1.sh` | 执行脚本：1 卡推理 + 1 卡训练 |
| `scripts/run_4x4.sh` | 执行脚本：4 卡推理 + 4 卡训练 |

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.9.0 |
| torch_npu | 2.9.0 |
| vLLM | 0.14.0 |
| CANN | 9.0.0-beta.1 |
| ATB (NNAL) | 9.0.0-beta.1 |
| Monarch | 从源码编译 ([npu 分支](https://github.com/pytorch-labs/monarch)) |
| 芯片 | Ascend 910B |

## 快速开始

### 1. 环境准备

```bash
conda activate monarch_ascend

# 加载 CANN 9.0 环境（ATB 必须与 CANN 版本匹配）
source /path/to/cann-9.0.0-beta.1/set_env.sh
source /path/to/cann-9.0.0-beta.1/nnal/atb/set_env.sh

# HuggingFace 镜像（可选）
export VLLM_USE_MODELSCOPE=true
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 使用执行脚本

`scripts/` 目录下提供了两个开箱即用的执行脚本，自动配置环境并启动训练：

**1+1 模式（1 卡推理 + 1 卡训练，共 2 NPU）：**

```bash
# 默认 3 步训练
bash areal/monarch_plugin/scripts/run_1x1.sh

# 自定义参数
bash areal/monarch_plugin/scripts/run_1x1.sh --steps 5 --model /path/to/model
```

**4+4 模式（4 卡推理 + 4 卡训练，共 8 NPU）：**

```bash
# 默认 2 步训练
bash areal/monarch_plugin/scripts/run_4x4.sh

# 自定义参数
bash areal/monarch_plugin/scripts/run_4x4.sh --steps 5 --model /path/to/model
```

脚本支持的参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--steps N` | 训练步数 | 1+1: 3, 4+4: 2 |
| `--model PATH` | 模型路径（本地或 HuggingFace） | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--cann PATH` | CANN 安装目录 | `/root/hzz/cann-9.0.0-beta.1` |

### 3. 手动启动

也可以直接调用 launcher，灵活传递 Hydra 参数：

```bash
# 1+1: 1 卡推理 + 1 卡训练
python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "allocation_mode=vllm:d1p1t1+d1p1t1" \
    "cluster.n_gpus_per_node=2" \
    "actor.path=/path/to/Qwen2.5-1.5B-Instruct" \
    "+total_train_steps=3"

# 4+4: 4 卡推理 + 4 卡训练 (yaml 默认配置)
python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "+total_train_steps=2"
```

Hydra 参数说明：

| 参数 | 说明 |
|------|------|
| `allocation_mode=vllm:d1p1t1+d1p1t1` | NPU 分配：1 卡推理 + 1 卡训练 |
| `allocation_mode=vllm:d4p1t1+d4p1t1` | NPU 分配：4 卡推理 + 4 卡训练（yaml 默认） |
| `cluster.n_gpus_per_node=N` | 节点可用 NPU 数量 |
| `actor.path=...` | 模型路径（本地或 HuggingFace） |
| `+total_train_steps=N` | 限制训练步数（测试用） |

### 4. 预期输出

```
MonarchPlugin INFO: GeneratorActor ready (Monarch RPC mode)
MonarchPlugin INFO: RewardActor spawned
MonarchPlugin INFO: SandboxActor spawned
MonarchPlugin INFO: AgentActor spawned
MonarchPlugin INFO: ReplayBufferActor spawned (max_size=8)
MonarchPlugin INFO: RolloutActor spawned
MonarchPlugin INFO: TrainerActor ready: max_steps=3, start_step=0
MonarchPlugin INFO: RolloutActor ready: steps_per_epoch=29
MonarchPlugin INFO: Starting async pipeline (true parallelism): steps 0 -> 3
MonarchPlugin INFO: [Rollout] Step 0 batch added to buffer (buffer_size=1)
MonarchPlugin INFO: [Step 1/3] epoch=0, epoch_step=0
...
MonarchPlugin INFO: Training completed successfully.
```

## 架构演进

| Phase | 内容 | Actor 数 |
|-------|------|----------|
| Phase 3 | Monarch 作为胶水层，GeneratorActor + TrainerActor | 2 |
| Phase 4 | 解耦编排，独立 RewardActor | 3 |
| Phase 5 | AgentActor + SandboxActor，多轮 Agent 场景 | 5 |
| Phase 6 | ReplayBufferActor，异步 rollout + 训练流水线 | 6 |
| Phase 6b | RolloutActor 独立化，真正的流水线并行 | 8 |

## 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `aclnnAddRmsNormBias not in libopapi.so` | vllm_ascend 自定义算子未加载 | launcher 启动时设置 `ASCEND_CUSTOM_OPP_PATH` |
| `signal only works in main thread` | `math_verify` 用 `signal.alarm()` 做超时 | `RewardActor.setup()` 中 monkey-patch `math_verify.utils.timeout` |
| `ActorMesh has attribute that collides with endpoint` | Monarch 保留属性名 `size` | 将 endpoint 重命名为 `buffer_size` |
| `mixes both async and sync endpoints` | Monarch 不允许 Actor 混用 async/sync | 统一为全 sync 或全 async endpoint |
| `async_scheduling only supports mp, uni, or external_launcher` | vLLM 白名单校验 executor | 显式设置 `async_scheduling = False` |

## 关键设计决策

**为什么用 Monarch RPC 而不是 HTTP？**
- 同一 runtime 下统一编排，无需服务发现
- 结构化消息传递，天然支持 Python 对象
- Actor 生命周期由 Monarch 管理，自动清理

**为什么 RolloutActor 跑在 CPU 上？**
- Rollout 的核心工作是调度：从 dataloader 取数据 → 发 RPC 给 GeneratorActor 做推理 → 收集结果
- 计算密集部分（模型推理）在 GeneratorActor 的 NPU 上完成
- CPU ProcMesh 不占用 NPU 资源，实现 rollout 和 training 真正并行

**为什么需要 ReplayBufferActor？**
- 解耦 rollout 生产和 training 消费的速率差异
- 支持版本感知的过期淘汰（staleness control）
- 实现流水线并行：rollout step N+1 与 training step N 重叠执行
