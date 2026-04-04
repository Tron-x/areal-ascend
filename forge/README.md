# Forge -- Framework-Agnostic Agentic RL Orchestration

Forge is a Monarch-native orchestration layer for large-scale agentic
reinforcement learning. It provides pluggable abstractions for training
engines, inference engines, agent strategies, and tools, so that
algorithm researchers can focus on RL logic while infrastructure
handles the distributed execution.

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
│   ├── sandbox.py  Subprocess code execution
│   └── rollout_producer.py  Optional rollout driver
│
├── engines/        Pluggable training/inference backends
│   ├── __init__.py create_engine() / create_config_bridge() factory
│   └── areal/     AReaL engine (PPOTrainer + FSDPEngine + vLLM)
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

1. **Protocol-driven**: actors depend only on `core/protocols.py`,
   never on concrete engine implementations.
2. **Engine-agnostic**: swap training/inference backends by
   implementing `TrainEngine` / `InferenceEngine` protocols.
3. **Zero framework coupling**: `core/`, `rl/`, `tools/`, `agents/`
   have zero imports from AReaL, Slime, or any training framework.
4. **Monarch-native**: uses Actor/ProcMesh/HostMesh for MPMD+SPMD
   distributed execution across heterogeneous GPU clusters.

## Design References

- **TorchForge** (Meta): ForgeActor lifecycle, Service layer,
  ReplayBuffer, async rollout/train pipeline, RL loss library
- **MiniMax Forge**: Windowed FIFO scheduling, composite reward,
  Gateway + Data Pool middleware architecture
- **ROLL** (Alibaba): GroupQueue, Agent Framework decoupling,
  ToolEnvWrapper, ActionParser, staleness expiry
- **Slime** (MiniMax): ReTool generate loop, token-level loss_mask,
  PythonSandbox, multi-rollout threads
- **veRL** (ByteDance): Token-in/token-out pattern, ReTool recipe

## Adding a New Engine

1. Create `forge/engines/<name>/` with `__init__.py`
2. Implement `TrainEngine` protocol (or legacy `TrainBackend`)
3. Register in `forge/engines/__init__.py` factory
4. Apps use `create_engine(backend="<name>")` -- zero app changes
