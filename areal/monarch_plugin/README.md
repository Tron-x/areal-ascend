# Monarch Plugin for AReaL

基于 [Meta Monarch](https://github.com/pytorch-labs/monarch) 框架的 AReaL 分布式 RL 插件，将训练、推理、奖励计算、经验回放等组件拆分为独立的 Monarch Actor，通过 RPC 通信实现全分离架构。支持任意卡数配置和多机部署。

## 架构总览

```
MonarchOrchestrator (主进程, 无 NPU)
  ├── GeneratorProcMesh (NPU 0..N-1)
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
  └── TrainingProcMesh (NPU N..N+M-1)
        ├── TrainerActor[rank=0] ─┐
        ├── ...                   ├── FSDP via HCCL all-reduce
        └── TrainerActor[rank=M] ─┘
```

所有 Actor 间通过 **Monarch RPC** 通信，无 HTTP 依赖。

## 支持的配置

### 单机

| 配置 | allocation_mode | NPU 数 | 说明 |
|------|----------------|--------|------|
| 1+1 | `vllm:d1p1t1+d1p1t1` | 2 | 最小验证配置 |
| 2+2 | `vllm:d2p1t1+d2p1t1` | 4 | 2 卡推理 + 2 卡训练 |
| 1+3 | `vllm:d1p1t1+d3p1t1` | 4 | 偏重训练并行度 |
| 4+4 | `vllm:d4p1t1+d4p1t1` | 8 | 默认 8 卡配置 |
| 2+6 | `vllm:d2p1t1+d6p1t1` | 8 | 偏重训练并行度 |
| 1(TP=4)+4 | `vllm:d1p1t4+d4p1t1` | 8 | 推理用 TP 并行 |

### 多机（需要 MONARCH_WORKERS）

| 配置 | 节点数 | 说明 |
|------|--------|------|
| 2 × (4+4) | 2 | 每节点 4 推理 + 4 训练（对称模式） |
| inf 节点 + train 节点 | 2 | 角色分离模式（MONARCH_NODE_ROLES） |

## 文件说明

### 编排层（声明式 Actor 生命周期管理）

| 文件 | 说明 |
|------|------|
| `launcher.py` | 主入口（~260 行），解析配置 → 构建 specs → 调用 registry spawn/init → 启动 pipeline |
| `actor_spec.py` | `ActorSpec` / `ActorContext` 数据类 — 每个 actor 的声明式描述（资源、依赖、构造参数） |
| `actor_registry.py` | `ActorRegistry` — 拓扑排序、有序 spawn/init、反序 shutdown 的生命周期管理器 |
| `specs.py` | `make_actor_specs()` — 根据配置构建所有 ActorSpec 的唯一入口 |
| `pipeline.py` | `run_training_pipeline()` — 异步 rollout→replay_buffer→training 流水线 |
| `bootstraps.py` | 平台无关的 bootstrap 工厂（Ascend NPU / CUDA） |

### Actor 实现

| 文件 | 说明 |
|------|------|
| `generator_actor.py` | 封装 vLLM `AsyncLLM`，提供推理端点 |
| `executor.py` | `AReaLMonarchExecutor` — 自定义 vLLM Executor，在 Monarch Worker ProcMesh 中运行 |
| `monarch_inf_engine.py` | `MonarchVLLMEngine` — 替换 AReaL 的 `RemotevLLMEngine`，通过 Monarch RPC 路由请求 |
| `trainer_actor.py` | `TrainerActor` — in-process FSDP 训练，拆分 rollout/train_on_batch 端点 |
| `reward_actor.py` | `RewardActor` + `MonarchRewardWrapper`，CPU 上运行 reward 计算 |
| `sandbox_actor.py` | `SandboxActor` — 隔离子进程执行 Python 代码（带超时） |
| `agent_actor.py` | `AgentActor` + `MonarchAgentWorkflow` — 多轮 Agent 交互编排 |
| `replay_buffer_actor.py` | `ReplayBufferActor` — 异步经验缓冲，支持版本感知的过期淘汰 |
| `rollout_actor.py` | `RolloutActor` — 独立 rollout 生产，拥有自己的 dataloader 和 WorkflowExecutor |

### 基础设施

| 文件 | 说明 |
|------|------|
| `topology.py` | 集群拓扑与设备放置抽象（单机/多机统一接口） |
| `weight_sync.py` | XCCL weight sync alloc_mode 解析与修正 |
| `scripts/run.sh` | 通用执行脚本（支持任意 N+M 配置） |
| `scripts/run_1x1.sh` | 快捷脚本：1 卡推理 + 1 卡训练 |
| `scripts/run_4x4.sh` | 快捷脚本：4 卡推理 + 4 卡训练 |
| `scripts/run_multi_node.sh` | 多机执行脚本 |

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

### 2. 使用通用脚本（推荐）

`scripts/run.sh` 支持任意 N+M 配置，通过 `--inf` 和 `--train` 指定推理和训练卡数：

```bash
# 1+1: 1 卡推理 + 1 卡训练 (2 NPU)
bash areal/monarch_plugin/scripts/run.sh --inf 1 --train 1

# 4+4: 4 卡推理 + 4 卡训练 (8 NPU)
bash areal/monarch_plugin/scripts/run.sh --inf 4 --train 4

# 2+6: 2 卡推理 + 6 卡训练 (8 NPU)
bash areal/monarch_plugin/scripts/run.sh --inf 2 --train 6

# 1+3: 1 卡推理 + 3 卡训练 (4 NPU)
bash areal/monarch_plugin/scripts/run.sh --inf 1 --train 3

# 带 TP 并行的推理: 1 inf (TP=4) + 4 train (8 NPU)
bash areal/monarch_plugin/scripts/run.sh --inf 4 --train 4 --tp 4

# 自定义模型和步数
bash areal/monarch_plugin/scripts/run.sh --inf 4 --train 4 --steps 10 --model /path/to/model
```

通用脚本参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--inf N` | 推理 NPU 数 | 1 |
| `--train N` | 训练 NPU 数 | 1 |
| `--tp N` | 推理 Tensor Parallel 度 | 1 |
| `--pp N` | 推理 Pipeline Parallel 度 | 1 |
| `--steps N` | 训练步数 | 3 |
| `--model PATH` | 模型路径 | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--cann PATH` | CANN 安装目录 | `/root/hzz/cann-9.0.0-beta.1` |

### 3. 快捷脚本

针对常用配置的快捷脚本：

```bash
# 1+1 (2 NPU, 默认 3 步)
bash areal/monarch_plugin/scripts/run_1x1.sh

# 4+4 (8 NPU, 默认 2 步)
bash areal/monarch_plugin/scripts/run_4x4.sh
```

### 4. 多机部署

需要先在每个 worker 节点启动 Monarch worker 进程：

```bash
# 在每个 worker 节点执行：
python -c "
from monarch._src.actor.bootstrap import run_worker_loop_forever
run_worker_loop_forever(address='tcp://0.0.0.0:29600', ca='trust_all_connections')
"
```

然后在 driver 节点执行：

```bash
# 2 节点, 每节点 4+4 (对称模式)
bash areal/monarch_plugin/scripts/run_multi_node.sh \
    --workers "tcp://node0:29600,tcp://node1:29600" \
    --nodes 2 --inf 4 --train 4

# 角色分离模式: node0 全推理, node1 全训练
MONARCH_NODE_ROLES=inference,training \
bash areal/monarch_plugin/scripts/run_multi_node.sh \
    --workers "tcp://node0:29600,tcp://node1:29600" \
    --nodes 2 --inf 8 --train 8
```

### 5. 手动启动

直接调用 launcher，灵活传递 Hydra 参数：

```bash
# 1+1
python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "allocation_mode=vllm:d1p1t1+d1p1t1" \
    "cluster.n_gpus_per_node=2" \
    "+total_train_steps=3"

# 2+6
python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "allocation_mode=vllm:d2p1t1+d6p1t1" \
    "+total_train_steps=3"

# 4+4 (yaml 默认配置)
python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "+total_train_steps=2"

# 多机 (需要 MONARCH_WORKERS 环境变量)
MONARCH_WORKERS=tcp://node0:29600,tcp://node1:29600 \
python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "cluster.n_nodes=2" \
    "+total_train_steps=2"
```

Hydra 参数说明：

| 参数 | 说明 |
|------|------|
| `allocation_mode=vllm:dXpYtZ+dApBtC` | NPU 分配（推理 DP×PP×TP + 训练 DP×PP×TP） |
| `cluster.n_gpus_per_node=N` | 每节点 NPU 数量 |
| `cluster.n_nodes=N` | 节点数量（多机时使用） |
| `actor.path=...` | 模型路径 |
| `+total_train_steps=N` | 限制训练步数 |

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
| Phase 7 | 声明式 launcher 重构：ActorSpec + ActorRegistry + 拓扑排序 | 8 |

## 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `aclnnAddRmsNormBias not in libopapi.so` | vllm_ascend 自定义算子未加载 | launcher 启动时设置 `ASCEND_CUSTOM_OPP_PATH` |
| `signal only works in main thread` | `math_verify` 用 `signal.alarm()` 做超时 | `RewardActor.setup()` 中 monkey-patch `math_verify.utils.timeout` |
| `ActorMesh has attribute that collides with endpoint` | Monarch 保留属性名 `size` | 将 endpoint 重命名为 `buffer_size` |
| `mixes both async and sync endpoints` | Monarch 不允许 Actor 混用 async/sync | 统一为全 sync 或全 async endpoint |
| `async_scheduling only supports mp, uni, or external_launcher` | vLLM 白名单校验 executor | 显式设置 `async_scheduling = False` |

## 拓扑层设计 (topology.py)

`topology.py` 提供了 `ClusterTopology` 抽象，统一处理单机和多机的设备放置：

```
allocation_mode + cluster config
        │
        ▼
  ClusterTopology.from_config()
        │
        ▼
  DevicePlacement
    ├── inference:  RolePlacement (哪些节点/设备做推理)
    ├── training:   RolePlacement (哪些节点/设备做训练)
    ├── cpu_services: RolePlacement (CPU Actor 放置)
    └── master_addr: str (FSDP MASTER_ADDR)
```

单机时 `ClusterTopology` 自动使用 `this_host()`；多机时通过 `MONARCH_WORKERS` 环境变量发现 worker 节点，使用 `attach_to_workers()` 创建跨节点 `HostMesh`。

### 多机放置策略

- **对称模式**（默认）：每个节点按相同比例分配推理和训练设备
- **角色分离模式**：通过 `MONARCH_NODE_ROLES=inference,training` 指定节点角色

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

**为什么要做声明式 launcher 重构？**
- 旧 launcher 是 1100+ 行的单一函数，spawn/init/shutdown 逻辑交织
- 新架构：每个 actor 用 `ActorSpec` 声明式描述（资源、依赖、构造参数），`ActorRegistry` 自动拓扑排序管理生命周期
- launcher 降到 ~260 行，只做配置解析 → specs 构建 → registry 调用 → pipeline 启动
- 新增 actor 只需在 `specs.py` 加一个 `ActorSpec`，无需改 launcher 逻辑
