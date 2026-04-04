# Forge -- Framework-Agnostic Agentic RL Orchestration

Forge is a Monarch-native orchestration layer for large-scale agentic reinforcement
learning. It provides pluggable abstractions for training engines, inference engines,
agent strategies, and tools, so that algorithm researchers can focus on RL logic while
infrastructure handles the distributed execution.

## Architecture

```
forge/
├── core/           Zero-dependency protocols, types, config
│   ├── types.py    Episode, TrainBatch, Completion, AgentAction,
│   │               ProcessConfig, ServiceConfig, LauncherConfig
│   ├── protocols.py TrainEngine, InferenceEngine, RewardFn, AgentLogic
│   ├── config.py   ForgeConfig
│   └── chat_template.py ChatTemplate + presets (ChatML, Llama)
│
├── actors/         Monarch actor wrappers (thin shells over protocols)
│   ├── generator/  vLLM Generator (inference)
│   ├── trainer.py  TrainerActor (training, delegates to engine)
│   ├── agent.py    AgentActor (multi-turn orchestrator)
│   ├── reward.py   RewardActor
│   ├── replay_buffer.py  Episode-level async buffer
│   ├── group_buffer.py   GRPO group rollout + Windowed FIFO
│   └── sandbox.py  Subprocess code execution
│
├── engines/        Pluggable training/inference backends
│   ├── __init__.py create_engine() / create_batch_adapter() / create_config_bridge()
│   ├── areal/     AReaL engine (PPOTrainer + FSDPEngine + vLLM) — legacy TrainBackend
│   ├── fsdp/      Native FSDP2 engine (TrainEngine + BatchAdapter)
│   └── weight_sync/  WeightSyncStrategy implementations
│       ├── nccl_sync.py       NCCL/HCCL collective broadcast
│       ├── checkpoint_sync.py Filesystem checkpoint transfer
│       └── hixl_sync.py       HIXL one-sided RDMA (stub)
│
├── rl/             Framework-agnostic RL algorithms
│   ├── advantage.py  GRPO advantage computation
│   ├── collate.py    Episode → TrainBatch conversion
│   ├── rewards.py    reward_to_go, process_reward, composite_reward
│   └── loss/         GRPOLoss, DAPOLoss + composable primitives
│
├── tools/          Tool system for ReTool / TIR
│   ├── protocol.py   Tool, ActionParser, ToolSpec, ToolCall
│   ├── registry.py   ToolRegistry (register/discover/execute)
│   ├── parsers.py    CodeBlock, ToolCall, FunctionCall, Composite
│   └── python_sandbox.py  Safe Python execution
│
├── agents/         AgentLogic implementations
│   ├── react.py    SimpleReActAgent (code execution loop)
│   ├── retool.py   ReToolAgent (multi-format tool calling)
│   └── external.py ExternalAgentRunner (CLI-Native mode)
│
├── observability/  Forge-native metrics and timing
│   ├── metrics.py  record_metric, accumulators, cross-rank reduce
│   └── tracer.py   CPU timer with step breakdown
│
├── service/        Service layer (multi-replica, routing)
│   ├── model_proxy.py        Unified LLM interface
│   ├── model_proxy_server.py HTTP API + trajectory recording
│   ├── service.py            Replica management
│   └── router.py             Load balancing + session affinity
│
├── apps/           Orchestration entry points
│   ├── grpo.py     Synchronous GRPO training
│   └── agent_rl.py Agentic RL (sync + async pipeline)
│
└── provisioner.py  Multi-node resource management
                    Local / Slurm / K8s-preallocated modes
```

## Design Principles

1. **Protocol-driven**: actors depend only on `core/protocols.py`, never on concrete
   engine implementations.
1. **Engine-agnostic**: swap training/inference backends by implementing `TrainEngine` /
   `InferenceEngine` protocols.
1. **Zero framework coupling**: `core/`, `rl/`, `tools/`, `agents/` have zero imports
   from AReaL, Slime, or any training framework.
1. **Monarch-native**: uses Actor/ProcMesh/HostMesh for MPMD+SPMD distributed execution
   across heterogeneous GPU clusters.

## Design References

- **TorchForge** (Meta): ForgeActor lifecycle, Service layer, ReplayBuffer, async
  rollout/train pipeline, RL loss library
- **MiniMax Forge**: Windowed FIFO scheduling, composite reward, Gateway + Data Pool
  middleware architecture
- **ROLL** (Alibaba): GroupQueue, Agent Framework decoupling, ToolEnvWrapper,
  ActionParser, staleness expiry
- **Slime** (MiniMax): ReTool generate loop, token-level loss_mask, PythonSandbox,
  multi-rollout threads
- **veRL** (ByteDance): Token-in/token-out pattern, ReTool recipe

## Adding a New Engine

1. Create `forge/engines/<name>/` with `__init__.py`
1. Implement `TrainEngine` protocol (from `forge.core.protocols`)
1. Implement `BatchAdapter` for your engine's tensor layout
1. Register in `forge/engines/__init__.py` factory
1. Apps use `create_engine(backend="<name>")` -- zero app changes

## Weight Sync

Training engines use `WeightSyncStrategy` to transfer weights to the
Generator. Three strategies are available:

| Strategy | Use case | Env var |
|----------|----------|---------|
| `NCCLWeightSync` | Co-located train+infer (same cluster) | `FORGE_WEIGHT_SYNC=nccl` |
| `CheckpointWeightSync` | Cross-cluster / elastic scheduling | `FORGE_WEIGHT_SYNC=checkpoint` |
| `HIXLWeightSync` | NPU clusters with Monarch HIXL | `FORGE_WEIGHT_SYNC=hixl` |

The legacy AReaL backend handles weight sync internally (built into
`TrainBackend.sync_weights`). The new `TrainEngine` protocol separates
weight description (`get_weights_spec()`) from weight data
(`state_dict_for_sync()`), letting the sync strategy decide the transport.
