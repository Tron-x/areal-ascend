# Forge Architecture — Current State

> Snapshot as of Phase A1 + R1 + A2-full + R2-narrow + A3-MVP landed.
>
> Source of truth for "what's actually wired up today" — as opposed to
> [`agentic_rl_architecture.md`](agentic_rl_architecture.md) which carries the
> longer-term vision + phase plan.

The diagrams in this file use [Mermaid](https://mermaid.js.org/); GitHub / VS Code /
Cursor render them inline.

## 1. Layer stack

```mermaid
flowchart TB
    subgraph YAML["User YAML (e.g. agentic_rl_with_tool_server.example.yaml)"]
        Y1["roles: trainer, generator, storage, tool_server, ..."]
        Y2["reward: gsm8k · agent: retool · workflow: multi_turn"]
    end

    subgraph APPS["Forge Apps (algorithm entry points)"]
        A1["forge/apps/grpo.py<br/>(single-turn GRPO)"]
        A2["forge/apps/agent_rl.py<br/>(agentic multi-turn)"]
    end

    subgraph REGS["Registries (user extension points)"]
        R1["forge.reward<br/>@register_reward(scope)"]
        R2["forge.agents<br/>@register_agent"]
        R3["areal.workflow<br/>@register_workflow"]
        R4["forge.tools<br/>@register_tool"]
    end

    subgraph TYPES["Core data types"]
        T1["Episode (single-turn)"]
        T2["Trajectory · Turn · Observation<br/>(multi-turn, A3)"]
        T3["RewardPipeline + RewardBreakdown"]
    end

    subgraph CORE["Forge Core (config / roles / bridge)"]
        C1["LauncherConfig<br/>.roles / .meshes<br/>bidirectional bridge (R2-narrow)"]
        C2["resolve_tool_server_options<br/>(forge/tools/server_config.py)"]
        C3["BareMetalLauncher<br/>get_host_mesh(name)"]
    end

    subgraph MONARCH["Monarch (unchanged primitives)"]
        M1["JobTrait family<br/>SSHJob / KubernetesJob / SlurmJob / LocalJob"]
        M2["HostMesh · ProcMesh · Actor · @endpoint"]
        M3["attach_to_workers / bootstrap / RDMABuffer"]
        M4["NPU HiXL adapter (one-sided RDMA)"]
    end

    subgraph LOW["Torch / TorchStore / HCCL / HiXL / SGLang / vLLM"]
    end

    YAML --> APPS
    APPS --> REGS
    APPS --> TYPES
    APPS --> CORE
    CORE --> MONARCH
    MONARCH --> LOW
```

## 2. Registries at a glance

```mermaid
flowchart LR
    subgraph RW["forge.reward"]
        rw1["@register_reward('gsm8k')"]
        rw2["@register_reward('tool_call_valid',<br/>scope='process')"]
        rw3["get_reward / rewards_by_scope"]
    end
    subgraph AG["forge.agents"]
        ag1["@register_agent('react')"]
        ag2["@register_agent('retool')"]
        ag3["@register_agent('harbor')"]
        ag4["@register_agent('external')"]
    end
    subgraph WF["areal.workflow"]
        wf1["@register_workflow('rlvr')"]
        wf2["@register_workflow('multi_turn')"]
        wf3["@register_workflow('vision_rlvr')"]
    end
    subgraph TL["forge.tools"]
        tl1["@register_tool('python_sandbox')"]
        tl2["(browser / bash / ... future)"]
    end

    CLI["python -m forge<br/>list-rewards · list-agents<br/>list-workflows · list-tools"]

    RW --> CLI
    AG --> CLI
    WF --> CLI
    TL --> CLI
```

## 3. Data types — Episode vs. Trajectory

```mermaid
classDiagram
    class Episode {
        +str prompt
        +str response
        +float reward
        +list~float~ step_rewards?      ← A3 hook
        +dict reward_breakdown
        +list~int~ token_ids
        +list~float~ generator_logprobs
        +to_dict() / from_dict()
    }

    class Trajectory {
        +str trajectory_id
        +str prompt
        +Any target
        +list~Turn~ turns
        +float final_reward
        +list~float~ step_rewards
        +dict reward_breakdown
        +assistant_turns()
        +final_response()
        +sync_step_rewards()
        +from_single_turn()
    }

    class Turn {
        +int turn_idx
        +TurnRole role
        +str content
        +list~ToolCall~ tool_calls
        +list~Observation~ observations
        +float reward
        +list~int~ token_ids
    }

    class Observation {
        +str text
        +bool success
        +dict metadata
        +from_tool_result()
    }

    class ToolCall {
        +str name
        +dict arguments
        +str raw_text
    }

    Trajectory "1" --> "*" Turn
    Turn "1" --> "*" ToolCall
    Turn "1" --> "*" Observation
```

## 4. Reward pipeline (final + process scopes)

```mermaid
sequenceDiagram
    autonumber
    participant App as agent_rl.py
    participant Pipe as RewardPipeline
    participant Reg as forge.reward registry
    participant Fn1 as gsm8k_reward_fn<br/>(scope=final)
    participant Fn2 as tool_call_valid<br/>(scope=process)
    participant Traj as Trajectory

    App->>Pipe: RewardPipeline(['gsm8k', 'tool_call_valid'])
    Pipe->>Reg: get_reward + get_reward_scope (×2)
    App->>Pipe: score(traj)
    Pipe->>Fn1: fn(prompt, response, target=...)
    Fn1-->>Pipe: 1.0   (scalar)
    Pipe->>Fn2: fn(trajectory)
    Fn2-->>Pipe: [0.1, 0.0, 0.1]   (per-turn)
    Pipe->>Pipe: final = Σ(scalar·w) + Σ(steps·w)<br/>step_totals += w · steps
    Pipe->>Traj: write back<br/>final_reward, step_rewards,<br/>reward_breakdown, turns[i].reward
    Pipe-->>App: RewardBreakdown(final, step_rewards,<br/>per_reward, per_reward_steps)
```

## 5. Placement chain — tool_server on a dedicated host

```mermaid
flowchart TB
    Y["<b>YAML</b><br/>roles:<br/>&nbsp;&nbsp;tool_server:<br/>&nbsp;&nbsp;&nbsp;&nbsp;procs: 1<br/>&nbsp;&nbsp;&nbsp;&nbsp;placement: {host_idx: 1, count: 2}<br/>&nbsp;&nbsp;&nbsp;&nbsp;tools: [python_sandbox]"]

    LC["<b>LauncherConfig.__post_init__</b><br/>_sync_roles_and_meshes()<br/>(R2-narrow, bidirectional)"]

    subgraph BRIDGE["After bridge"]
        B1["roles['tool_server']<br/>=RoleConfig(placement.host_idx=1,<br/>tools=['python_sandbox'], ...)"]
        B2["meshes['tool_server']<br/>={'host_idx': 1}"]
    end

    RES["<b>resolve_tool_server_options</b><br/>(forge/tools/server_config.py)<br/>defaults mesh_name = role_name"]

    OPT["<b>ToolServerOptions</b><br/>procs=1 · num_replicas=2<br/>mesh_name='tool_server'<br/>tool_types=('python_sandbox',)<br/>host_idx=1"]

    SP["<b>_spawn_tool_server</b> (agent_rl.py)<br/>SandboxActor.options(procs=1,<br/>&nbsp;&nbsp;mesh_name='tool_server', num_replicas=2)<br/>.as_service(tool_types=['python_sandbox'])"]

    PROV["<b>Provisioner.get_host_mesh('tool_server')</b><br/>→ self.meshes['tool_server'].host_idx = 1"]

    MON["<b>Monarch</b><br/>spawn on worker[1]"]

    ACTOR["<b>SandboxActor replica</b> (×2)<br/>builds ToolRegistry<br/>mounts PythonTool<br/>exposes execute_tool / execute_code"]

    Y --> LC
    LC --> B1
    LC --> B2
    B1 --> RES
    B2 --> PROV
    RES --> OPT
    OPT --> SP
    SP --> PROV
    PROV --> MON
    MON --> ACTOR
```

## 6. Role catalog — which role is where

```mermaid
flowchart LR
    subgraph Host0["Host 0 (NPU)"]
        H0A["trainer<br/>FSDP2 / Megatron"]
    end
    subgraph Host1["Host 1 (NPU)"]
        H1A["generator<br/>vLLM / SGLang (TP>1)"]
    end
    subgraph HostS["Storage host (NPU + RDMA)"]
        HS["storage<br/>torchstore + HiXL"]
    end
    subgraph HostC["CPU host (A2-full new)"]
        HC["tool_server<br/>SandboxActor + ToolRegistry"]
    end
    subgraph Future["Future (not yet implemented)"]
        F1["agent_runner"]
        F2["reward_model"]
        F3["replay_buffer"]
        F4["workflow_coordinator"]
    end

    H0A <-- "weight sync<br/>(HiXL / HCCL broadcast)" --> HS
    H1A <-- "pull params" --> HS
    H1A <-- "tool call RPC" --> HC
```

## 7. End-to-end request flow (agentic GRPO)

```mermaid
sequenceDiagram
    autonumber
    participant User as User (YAML)
    participant Launch as forge launch CLI
    participant App as agent_rl.py
    participant Prov as Provisioner<br/>(BareMetalLauncher)
    participant Gen as Generator<br/>(vLLM replica)
    participant Agent as AgentActor<br/>(ReTool)
    participant Tool as SandboxActor<br/>(tool_server)
    participant Store as torchstore
    participant Tr as Trainer (FSDP2)

    User->>Launch: forge launch cfg.yaml
    Launch->>App: hydrate LauncherConfig
    App->>Prov: spawn meshes per roles
    par parallel spawn
        Prov->>Gen: spawn on host=generator.host_idx
    and
        Prov->>Tool: spawn on host=tool_server.host_idx<br/>(ToolRegistry mounts python_sandbox)
    and
        Prov->>Store: spawn on storage host
    and
        Prov->>Tr: spawn trainer procs
    end

    loop rollout
        Agent->>Gen: agenerate(prompt)
        Gen-->>Agent: assistant text + tool_call
        Agent->>Tool: execute_tool('python_sandbox', {code})
        Tool-->>Agent: ToolResult(success, output)
        Agent->>Agent: append Turn to Trajectory
    end

    Agent->>App: trajectory complete
    App->>App: RewardPipeline.score(traj)<br/>→ final_reward + step_rewards
    App->>Tr: enqueue Episode (carries step_rewards)
    Tr->>Tr: PPO / GRPO update<br/>(scalar loss today;<br/>per-step loss = A3 follow-up)
    Tr->>Store: push new weights
    Gen->>Store: pull new weights<br/>(collective broadcast)
```

## 8. Boundary rules (what lives where)

```mermaid
flowchart TB
    subgraph F["Forge layer (we own)"]
        FA["YAML schema · roles · registries"]
        FB["RewardPipeline · Trajectory"]
        FC["LauncherConfig bridge"]
        FD["Tool server resolver"]
    end
    subgraph M["Monarch layer (upstream, don't fork)"]
        MA["JobTrait<br/>(SSH/Kubernetes/Slurm/Local)"]
        MB["HostMesh / ProcMesh primitives"]
        MC["attach_to_workers · bootstrap"]
        MD["RDMABuffer · HiXL adapter"]
    end
    subgraph L["Hardware / drivers"]
        LA["NPU / GPU"]
        LB["HCCL / NCCL / HiXL transports"]
    end

    F -- "maps roles → Monarch primitives" --> M
    M -- "runs on" --> L
```

**Rules of thumb:**

- Anything YAML-visible or algorithm-visible lives in Forge.
- Anything that describes *how to spawn workers on some cluster* lives in Monarch's
  `JobTrait`; Forge's `BareMetalLauncher` is a thin adapter for pre-provisioned SSH
  workers.
- Anything hardware-specific (HiXL / HCCL / NPU quirks) lives behind Monarch's transport
  layer; Forge never imports `torch_npu` directly.

## 9. What's intentionally *not* yet in the diagram

Tracked in [`agentic_rl_architecture.md`](agentic_rl_architecture.md)'s phase plan;
listed here so readers don't mistake the omission for a gap in understanding.

| Piece                                    | Why deferred                                                                |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| General-purpose placement scheduler      | No concrete heterogeneous cluster driving it yet.                           |
| Transport abstraction (R3)               | Monarch RPC + HiXL cover every wire we use today.                           |
| `agent_runner` as independent role       | Currently rides on `generator`; separation needs A2-follow.                 |
| PPO/GRPO consuming `step_rewards`        | Data carrier ready (`Episode.step_rewards`); loss change is algorithm-side. |
| Workflows emitting `Trajectory` natively | `Trajectory.from_single_turn` bridge is good enough until needed.           |

## 10. Test coverage map

```mermaid
flowchart LR
    subgraph T1["tests/test_forge_roles.py (24)"]
        T1a["RoleConfig coercion"]
        T1b["Bidirectional bridge"]
        T1c["tool_server resolver"]
        T1d["Tool registry"]
    end
    subgraph T2["tests/test_forge_trajectory_reward.py (19)"]
        T2a["Trajectory roundtrip"]
        T2b["Scope registration"]
        T2c["RewardPipeline aggregate"]
        T2d["Built-in process rewards"]
    end

    Layers1["R1 · R2-narrow · A2-full"] --> T1
    Layers2["A3"] --> T2
```

Total: **43 passed, 0 lint errors.**
