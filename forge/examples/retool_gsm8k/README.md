# ReTool GSM8K -- Forge Agentic RL Example

Multi-turn tool-integrated reasoning on GSM8K math problems.

The model learns to:

1. Think about a math problem
1. Write Python code to compute the answer (`<code>print(...)</code>`)
1. Read the execution result
1. Give the final answer (`Answer: \boxed{42}`)

## Architecture

```
Forge Orchestration Layer:
  ReToolAgent        → parse <code> blocks + detect \boxed{} answers
  CompositeParser    → extract tool calls from any format
  ToolRegistry       → route to PythonSandbox
  PythonSandbox      → execute code in subprocess
  GroupBuffer         → collect n_samples responses per prompt
  ReplayBuffer       → async rollout/train pipeline

AReaL Training Engine (backend):
  Generator (vLLM)   → text generation on NPU 0-3
  TrainerActor (FSDP) → GRPO training on NPU 4-7
  Weight sync (XCCL)  → push updated weights to Generator
```

## Usage

```bash
# Sync mode (default, same as grpo.py but with Forge agent layer)
bash forge/examples/retool_gsm8k/run.sh

# Async pipeline mode
bash forge/examples/retool_gsm8k/run.sh --async

# Custom model
bash forge/examples/retool_gsm8k/run.sh --model /path/to/model --steps 10
```

## What's Different from `examples/math/gsm8k_rl.py`

| Aspect         | `examples/math/gsm8k_rl.py`   | This example                         |
| -------------- | ----------------------------- | ------------------------------------ |
| Entry point    | `forge.apps.grpo`             | `forge.apps.agent_rl`                |
| Rollout        | AReaL internal `RLVRWorkflow` | Forge `ReToolAgent` + `ToolRegistry` |
| Tool execution | None (pure text reasoning)    | `PythonSandbox` code execution       |
| Loss mask      | All tokens = 1                | LLM=1, tool output=0                 |
| Multi-turn     | Single turn                   | Up to `max_turns` rounds             |
| Agent logic    | Hardcoded in workflow         | Pluggable `AgentLogic` protocol      |
