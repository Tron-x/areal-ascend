# Agentic RL Architecture — Design Note

**Status**: design draft, no code yet. Extends `role_abstraction_design.md` with
agentic-specific roles and phases.

**Context**: the stated goal is an **agentic RL system**, not a traditional
`prompt → completion → reward` pipeline. That changes what "role" means, what
"transport" must cover, and which phases go first.

______________________________________________________________________

## 1. Traditional RL vs Agentic RL — what actually differs

| Axis               | Traditional RL (what forge/AReaL assumed) | Agentic RL (target)                               |
| ------------------ | ----------------------------------------- | ------------------------------------------------- |
| Rollout shape      | `prompt → completion` (1 shot)            | `agent ↔ env` loop — tool call, obs, act, …       |
| Reward             | Final only                                | **Process + Final** mixed                         |
| Episode length     | Fixed, short                              | Variable, tens of turns × thousands of tokens     |
| Tools              | None                                      | Independent services (sandbox, search, RAG, …)    |
| State              | Stateless prompt                          | KV cache / scratchpad / memory across turns       |
| Replay             | `(prompt, completion, reward)`            | **Trajectory** (turns × actions × observations)   |
| Concurrency        | Batch-synchronous                         | **Async**: episodes advance independently         |
| Weight sync target | vLLM workers only                         | vLLM **plus** agent-runner caches / tool contexts |

The traditional assumption is baked into `forge/apps/grpo.py` today. For agentic
scenarios every row on the right must be a first-class concept in the YAML + runtime,
not a hack.

______________________________________________________________________

## 2. Audit: what forge/AReaL already has

The agentic foundation is more complete than the infra layer suggested. Inventory:

| Layer        | Component                                                                                          | State                             |
| ------------ | -------------------------------------------------------------------------------------------------- | --------------------------------- |
| Agent logic  | `forge/agents/`: `HarborAgentLogic`, `ReToolAgent`, `SimpleReActAgent`, `ExternalAgentRunner`      | 4 implementations; protocol-based |
| Tools        | `forge/tools/`: `ToolRegistry` (registry mode), `ToolSpec`, `python_sandbox`                       | **Already registry-based**        |
| Workflows    | `areal/workflow/`: `RolloutWorkflow` + `RLVRWorkflow` / `MultiTurnWorkflow` / `VisionRLVRWorkflow` | Protocol-based; **no registry**   |
| Integrations | `areal/workflow/`: `openai_agent/`, `anthropic/`, `langchain/`                                     | Present                           |
| Examples     | `forge/examples/retool_gsm8k/`, `forge/examples/harbor/`                                           | End-to-end agentic runs exist     |

**Key finding**: tool is already a registry (pre-existing, "Inspired by ROLL"). Today's
reward-registry work was really catching reward up to where tools already were.

Three gaps remain before the pieces compose into an "agentic RL platform":

1. **No Agent registry / no Workflow registry** — switching agent or workflow today
   edits Python. A YAML-level short name is missing.
1. **Tool server is not a role** — `python_sandbox` runs inside the generator proc.
   Can't scale it independently, can't pin it to a separate host, can't share it across
   replicas.
1. **Trajectory / episode is not an explicit data type** — no shared `Trajectory`
   dataclass, no replay-buffer role, no clear boundary between "raw rollout" and
   "training batch". Makes process reward awkward.

______________________________________________________________________

## 3. Agentic-specific role extensions

Add four new role types on top of `role_abstraction_design.md §2`:

### 3.1 `agent_runner`

Runs the agent loop (ReAct / ReTool / Harbor / custom). Distinct from `generator`
because it owns the **loop orchestration** (turn counter, stop condition, tool-call
parsing), while `generator` owns **token generation** (vLLM forward pass).

```yaml
roles:
  agent_runner:
    hardware: {type: cpu, min_memory_gb: 16}   # agent logic is light
    placement: {count: 4}
    procs: 8                                    # high concurrency for async episodes
    role_type: agent_runner
    agent: retool                               # registry short-name
    agent_config:
      max_turns: 16
      tool_choice: auto
```

Why separate from generator: a single vLLM worker (TP=4) can feed **hundreds** of
concurrent agent-runner procs via async HTTP; binding agent loop to generator proc would
waste NPU.

### 3.2 `tool_server`

Hosts one or more tools (sandbox, search, RAG, code-exec). Independent placement,
independent scaling.

```yaml
roles:
  code_sandbox:
    hardware: {type: cpu, min_memory_gb: 32}   # no NPU needed
    placement: {count: 2}
    procs: 16                                   # 32 concurrent sandboxes total
    role_type: tool_server
    tools: [python_sandbox]                     # registry short-names

  search_backend:
    hardware: {type: cpu, min_memory_gb: 64}
    placement:
      node_selector: {group: data}              # land on data-heavy node
    procs: 4
    role_type: tool_server
    tools: [web_search, rag_retrieval]
```

Transport: `agent_runner → tool_server` is `rpc` (Monarch) or `http` depending on
backend. Key property: tool response **must not block generator**; async dispatch
required.

### 3.3 `workflow_coordinator`

Owns the per-episode orchestration: given prompt, instantiate agent_runner, connect it
to generator + tool_servers, collect trajectory, emit to replay buffer. Today this logic
lives in `RolloutWorkflow.arun_episode` and is implicitly single-proc; making it a role
lets it scale horizontally.

```yaml
roles:
  workflow:
    hardware: {type: cpu, min_memory_gb: 8}
    placement: {count: 1}
    procs: 32                                   # fan out across episodes
    role_type: workflow_coordinator
    workflow: rlvr_retool                       # registry short-name
```

### 3.4 `trajectory_store`

Replay buffer as a first-class role. Holds structured `Trajectory` objects (not raw
tensors). Trainer pulls from it; workflow pushes into it.

```yaml
roles:
  trajectory_store:
    hardware: {type: cpu, min_memory_gb: 256}
    placement: {count: 1}
    procs: 4                                    # shard by episode_id hash
    role_type: trajectory_store
    capacity: 100000                            # trajectories
```

______________________________________________________________________

## 4. Reward design: final + process side-by-side

`reward_granularity` question has no single correct answer — a good agentic framework
supports both simultaneously without user-visible choice:

```yaml
rewards:
  - name: gsm8k              # registered via @register_reward
    scope: final             # applied to full trajectory's final answer
    weight: 1.0

  - name: tool_call_valid    # applied per turn
    scope: process
    weight: 0.1

  - name: length_penalty
    scope: process
    weight: -0.01
```

Implementation hint:

- `RewardPipeline` iterates `rewards:` list. For each:
  - `scope: final` → call once on `trajectory.final_response`.
  - `scope: process` → call on each `turn` in `trajectory.turns`.
- Accumulates into `trajectory.step_rewards[step] += w * r` for process, and
  `trajectory.episode_reward += w * r` for final.
- Trainer consumes `step_rewards` directly when PRM; falls back to `episode_reward` when
  no process signal present.

This is the smallest change that makes process reward **opt-in per reward function**
without breaking final-only callers.

______________________________________________________________________

## 5. Transport extensions for agentic

`role_abstraction_design.md §3 Transport` lists
`rpc / rpc_streaming / one_sided_rdma / collective_broadcast / collective_scatter`. Add:

| mode            | backend       | use case                                   |
| --------------- | ------------- | ------------------------------------------ |
| `http`          | `tool_openai` | OpenAI-compatible tool server (search, …)  |
| `rpc_streaming` | `monarch`     | tool outputs streaming back chunk-by-chunk |
| `async_pub_sub` | `monarch`     | workflow → trajectory_store (non-blocking) |

Agentic transport choices are typically "per message, not per episode" — the same
`agent_runner → tool_server` pair may use `http` for some tools and `rpc` for others.
Transport declaration happens at **tool registration time**, not at YAML level.

______________________________________________________________________

## 6. Target YAML (full agentic example)

```yaml
roles:
  trainer:        {hardware: {type: npu, model: 910b}, placement: {count: 1}, procs: 4, role_type: training_backend}
  generator:      {hardware: {type: npu, model: 910b}, placement: {count: 1}, procs: 1, gpus_per_proc: 4, role_type: inference_engine}
  storage:        {hardware: {type: npu, requires_rdma: true}, placement: {colocate_with: trainer}, procs: 4, role_type: weight_storage}

  agent_runner:
    hardware: {type: cpu, min_memory_gb: 16}
    placement: {count: 2}
    procs: 8
    role_type: agent_runner
    agent: retool                           # @register_agent
    agent_config: {max_turns: 16}

  code_sandbox:
    hardware: {type: cpu, min_memory_gb: 32}
    placement: {count: 1}
    procs: 16
    role_type: tool_server
    tools: [python_sandbox]                 # @register_tool

  workflow:
    hardware: {type: cpu}
    placement: {count: 1}
    procs: 32
    role_type: workflow_coordinator
    workflow: rlvr_retool                   # @register_workflow

  trajectory_store:
    hardware: {type: cpu, min_memory_gb: 256}
    placement: {count: 1}
    procs: 4
    role_type: trajectory_store

transports:
  - {name: weight_push, from: trainer, to: storage, mode: one_sided_rdma, backend: hixl}
  - {name: weight_pull, from: storage, to: generator, mode: collective_broadcast, backend: hccl}
  - {name: gen_call,    from: agent_runner, to: generator, mode: http, backend: openai}
  - {name: tool_call,   from: agent_runner, to: code_sandbox, mode: rpc, backend: monarch}
  - {name: traj_push,   from: workflow, to: trajectory_store, mode: async_pub_sub, backend: monarch}
  - {name: traj_sample, from: trainer, to: trajectory_store, mode: rpc_streaming, backend: monarch}

rewards:
  - {name: gsm8k,           scope: final,   weight: 1.0}
  - {name: tool_call_valid, scope: process, weight: 0.1}
```

Reading this YAML tells you *everything* about the run: what's on what hardware, who
talks to whom over what transport, and how reward is shaped.

______________________________________________________________________

## 7. Phase plan (revised — agentic-first)

The original `role_abstraction_design.md §7` phases (R1 role parse → R2 placement → R3
transport) stay valid, but they're not the first things to ship. What algorithm
engineers hit immediately is **swapping agents and workflows**. Phase A-series addresses
that.

### Phase A1 — Agent & Workflow registry *(≈3 days, independently shippable)*

- `@register_agent(name)` + `get_agent(name)` in `forge/agents/__init__.py`.
- Apply to 4 existing agents (`react`, `retool`, `harbor`, `external`).
- `@register_workflow(name)` + `get_workflow(name)` in `areal/workflow/__init__.py`.
- Apply to `rlvr`, `multi_turn`, `vision_rlvr`.
- `grpo.py` reads `agent:` / `workflow:` short names from YAML; same YAML> env> default
  resolver used for reward.
- `python -m forge list-agents` / `list-workflows` CLI.

**Ship condition**: algorithm engineer can write `agent: my_custom_agent` + drop
`@register_agent("my_custom_agent")` in a Python file under `forge/agents/`; no other
code changes.

### Phase A2 — Tool as independent role *(in progress)*

**Landed (A2 MVP):**

- `forge/tools/server_config.py::resolve_tool_server_options` reads
  `launcher.roles.tool_server` from the R1 schema and returns concrete spawn parameters
  (`procs`, `num_replicas`, `mesh_name`, `tool_types`, `host_idx`, `extras`).
- `forge.apps.agent_rl._spawn_tool_server` replaces the hardcoded
  `SandboxActor.options(procs=1, mesh_name="sandbox").as_actor()` call. When
  `roles.tool_server.placement.count > 1` (or `extras.num_replicas` is set) the sandbox
  is spawned via `as_service` (load-balanced via `endpoint.route`); the legacy
  single-actor path is the default so existing YAMLs see zero behavior change.
- Backward-compatible: missing `roles.tool_server` falls back to `ToolServerOptions`'
  defaults which match the pre-A2 hardcoded call.
- Example YAML: `forge/configs/agentic_rl_with_tool_server.example.yaml`.
- Unit tests: `tests/test_forge_roles.py` (14 cases, no GPU / Monarch).

**Landed (A2-full):**

- `@register_tool("short_name")` decorator + `get_tool` / `available_tools` /
  `ensure_loaded` in `forge/tools/__init__.py` (mirrors the reward/agent/workflow
  registries).
- `PythonTool` registered under `python_sandbox`.
- `SandboxActor` accepts `tool_types=[...]` at construction, builds a per-actor
  :class:`ToolRegistry` lazily in worker proc, exposes a new
  `execute_tool(name, arguments)` endpoint. `execute_code` is kept intact so existing
  `AgentActor._call_sandbox` callers keep working.
- `resolve_tool_server_options` defaults `mesh_name` to the role name when a role entry
  is authored, so `BareMetalLauncher.get_host_mesh(mesh_name)` lands the tool server on
  the `host_idx` written by the R2-narrow bidirectional bridge -- no extra wiring
  needed.
- `forge list-tools` CLI subcommand.

**Still owed:**

- Actually ship non-python tools (browser, bash, ...) with safety / resource-isolation
  review. Framework is ready; it's a matter of writing `@register_tool`-decorated
  classes.
- Transport pluggability (HTTP / gRPC alongside Monarch RPC) -- R3 territory, deferred.

**Ship condition (A2-full, met)**: `code_sandbox` can be spawned as an N-replica service
on a dedicated host via
`roles.tool_server: {placement: {host_idx: N, count: K}, tools: [python_sandbox]}` with
zero code changes; `AgentActor` still invokes it via `sandbox.execute_code.route(...)`.

### Phase A3 — Trajectory + process reward *(landed A3 MVP)*

**Landed:**

- `forge/core/trajectory.py`: `Turn`, `Observation`, `Trajectory` dataclasses with
  `to_dict` / `from_dict` roundtrip and `from_single_turn` factory for single-turn
  backfit.
- `@register_reward(name, scope="final" | "process")`: scope defaults to `"final"` so
  every pre-A3 reward (`gsm8k`, `geometry3k`, `clevr_count_70k`, ...) keeps working with
  zero changes. `get_reward_scope(name)` + `rewards_by_scope(scope)` expose the new
  metadata.
- `forge/reward/pipeline.py::RewardPipeline`: composable aggregator that handles final +
  process rewards in one pass, with optional per-reward `weights`. Writes
  `trajectory.final_reward`, `trajectory.step_rewards`, `trajectory.reward_breakdown`,
  and `turn.reward` back in place (skippable via `write_back=False`).
- `forge/reward/process_rewards.py`: built-in `tool_call_valid` (+0.1 on a successful
  tool invocation) and `turn_efficiency` (-0.01 per assistant turn) as canonical
  per-turn reward examples.
- `Episode.step_rewards: list[float] | None` -- forward-compat hook so a single-turn
  `Episode` can still carry per-turn rewards when produced by a multi-turn workflow.

**Still owed (A3-follow-up):**

- Wire `step_rewards` into the PPO / GRPO trainer side: today the value rides through
  `Episode` unchanged but the loss function still only consumes the scalar `reward`.
  Enabling per-step advantages is an algorithm-side change, tracked separately.
- Rollout workflows (`MultiTurnWorkflow`, `ReTool`, `Harbor`) should emit `Trajectory`
  objects natively instead of hand-rolled dicts. Today `Trajectory.from_single_turn`
  provides an easy bridge.

**Ship condition (met)**: `RewardPipeline(["gsm8k", "tool_call_valid"])` composes final
\+ process rewards and produces a `step_rewards` vector ready for trainer consumption.
Unit tests `tests/test_forge_trajectory_reward.py` cover the full cross-product (19
green).

### Phases R1 / R2 / R3 (role abstraction infra) — continue in parallel

- R1 (Role parse + legacy meshes): unblocked, can ship alongside A1.
- R2 (Placement scheduler): needed for A2's node_selector to be honored on
  non-homogeneous clusters.
- R3 (Transport abstraction): needed for A2's RPC and A3's async pub-sub.

Revised ordering (reflecting the Monarch-boundary correction):

**A1 → R1 → A2-MVP → R2-narrow → A2-full → A3 → R3**.

- **A1** (agent/workflow registry) -- *landed*.
- **R1** (role schema parse) -- *landed*.
- **A2-MVP** (YAML-driven tool_server spawn + as_service/as_actor dispatch) -- *landed*;
  depends only on R1 and does not touch Monarch.
- **R2-narrow** (bidirectional role/mesh bridge fix) -- *landed*.
  `BareMetalLauncher.get_host_mesh(name)` now honors `roles.<name>.placement.host_idx`
  uniformly for pure-roles, pure-meshes, AND mixed authorship, via the
  `_sync_roles_and_meshes` bridge inside `LauncherConfig`. The launcher code itself was
  not touched -- the bridge guarantees whatever read path it uses (the legacy `meshes`
  map) sees every placement the user authored. K8s / Slurm support stays delegated to
  Monarch's native `KubernetesJob` / `SlurmJob`.
- **A2-full** -- multi-tool registry + `tool_server` actually running on `host_idx`
  provided by R2-narrow (today we log it; next step wires it through the
  `SandboxActor.options(hosts=...)` path).
- **A3** -- `Trajectory` + process reward.
- **R3** -- transport abstraction; deferred until A2-full / A3 need more than Monarch
  RPC + RDMA.

______________________________________________________________________

## 8. What we are explicitly NOT doing in this arc

- LLM-as-judge / reward model as an agent itself. Add later once `agent_runner` role
  exists; trivially expressible.
- Hierarchical agents (agent calling another agent). Out of scope; requires trajectory
  nesting.
- Multi-agent RL (MARL, self-play). Explicitly a different architecture.
- Tool sandboxing with security isolation (gVisor, nsjail). Ops concern, not framework
  concern.

These are all reachable from the A1-A3 + R1-R3 foundation; none block it.

______________________________________________________________________

## 9. First concrete decision needed

Before writing code:

- [ ] Confirm **A1 goes first** (Agent/Workflow registry, ≈3 days, pure ergonomics, zero
  runtime risk).
- [ ] Confirm agent short-names in YAML: `retool`, `react`, `harbor`, `external` (or
  rename).
- [ ] Confirm workflow short-names: `rlvr`, `multi_turn`, `vision_rlvr`.
- [ ] Decide whether `list-agents` / `list-workflows` share one CLI (`list-registry`) or
  stay separate for discoverability.

Once A1 lands, the algorithm-engineer friction metric becomes:

> "Write a custom agent + wire it into GRPO training: **new Python file + 3 YAML lines +
> 0 framework edits**."

That's the user-visible bar. Everything below that (role placement, transport,
trajectory) is infra and can evolve without algorithm engineers noticing.
