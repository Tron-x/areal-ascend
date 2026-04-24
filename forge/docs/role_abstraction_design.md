# Role Abstraction — Design Note

**Status**: partially shipped in R1.5 (a/b/c). The aspirational schema in §2 below
is the long-term north star; §2b records what actually landed.

**Context**: forge today can place `trainer` / `generator` / `storage` on specific hosts
via `launcher.meshes.<name>.host_idx`. That's enough for the 2-node GRPO smoke and the
"dedicated PS" variant. It's not enough for the broader vision: an RL **scheduling
primitive** where arbitrary roles (Reward Model, Experience Buffer, Data Pipeline,
future components) run on arbitrary (possibly heterogeneous) hardware, each
independently scalable, connected by **explicit transports**.

This doc captures the design we agreed on. The R1.5 shipped subset is documented in
§2b; the aspirational schema in §2 stays as the direction of travel.

## R1.5 delta (what actually shipped, 2026-04)

The delivered schema is intentionally flatter than the original §2 sketch. Three
hand-offs, one per sub-phase:

* **R1.5a** — introduce `RoleConfig { devices, hardware, colocate, host_idx, extras }`
  as a peer of the legacy `meshes` map; `__post_init__` keeps them in sync (either
  one can be authored). Renamed `weight_sync.storage_mesh` → `storage_role`; legacy
  name still works with a `DeprecationWarning`.
* **R1.5b** — `ForgeActor.launch` defaults `hosts=1` when a remote launcher is active,
  so trainer / reward actors go through `get_host_mesh(name)` instead of implicit
  `this_host()`. This decouples driver placement from any specific actor role.
* **R1.5c** — add `launcher.pool: [PoolHost]` as the infrastructure-owned cluster
  pool (hosts + port + device count + optional `role: driver` tag). A greedy
  scheduler inside `LauncherConfig.__post_init__` binds `roles` (device counts,
  colocate constraints) to `pool` entries, auto-populating `workers` and
  `meshes.<name>.host_idx`. `forge launch` prefers `launcher.pool` over
  `--hostfile` when both are authored; in pool-only mode, it materializes a temp
  hostfile from the pool so the downstream bash fleet / ssh_job fleet keep working
  unchanged.

### §2b Shipped YAML schema (R1.5c)

```yaml
launcher:
  type: bare_metal

  # INFRASTRUCTURE-OWNED: what hardware exists.
  pool:
    - host: 192.168.0.26
      port: 22222
      hardware: npu
      n_devices: 8
    - host: 192.168.0.23
      port: 22222
      hardware: npu
      n_devices: 8
      role: driver      # optional; defaults to pool[0]

  # ALGORITHM-OWNED: how this experiment wants to use the cluster.
  roles:
    trainer:
      devices: 8              # accelerator card count
      hardware: npu
    generator:
      devices: 1
      hardware: npu
    storage:
      devices: 4
      hardware: npu
      colocate: trainer       # pin to trainer's host
    reward:
      devices: 0              # CPU-only
      hardware: npu
      colocate: trainer

  weight_sync:
    storage_role: storage     # must match a key in roles[]
    # ... rest unchanged
```

Parallelism strategy (FSDP dp/tp, vLLM TP) stays in the workload config -- it's
deliberately NOT part of `roles[]`, since algorithm authors want to swap TP sizes
independently of resource allocation.

### Legacy escape hatches still work

Every old YAML keeps working during the migration window:

* `bare_metal.workers` + `meshes.<name>.host_idx` — hand-authored placement.
  `LauncherConfig.__post_init__` leaves both intact and the scheduler is a no-op
  when `pool` is empty.
* `weight_sync.storage_mesh` — alias for `storage_role`; emits
  `DeprecationWarning`.
* `--hostfile` in `forge launch` — when no `launcher.pool` exists in the YAML,
  `forge launch` falls back to the legacy hostfile path with the same
  "second non-empty line = driver" heuristic.

______________________________________________________________________

## 1. Why "role" and not "mesh"

Today: `launcher.meshes.<name>.host_idx: N` — a dict from mesh names to integer indices
into `bare_metal.workers`. Assumes:

1. All hosts are interchangeable (same hardware).
1. Every "mesh" is a simple host pin — no colocation constraints, no resource
   requirements, no replica count.
1. A fixed set of names (`trainer` / `generator` / `storage`) baked into `grpo.py`.

None of these hold in the target architecture:

1. Heterogeneous clusters mix NPU / GPU / CPU-heavy nodes.
1. Some roles must be colocated (storage with trainer for HCCS); others must be
   anti-colocated (RM off the trainer host so it doesn't compete for NPU 0).
1. New roles (RM, replay buffer, rollout-only workers, ...) should be declarable in
   YAML, not hardcoded.

So `meshes` → `roles`. A **Role** is: *"I want K processes of this shape running on
nodes that match these hardware / placement constraints"*.

______________________________________________________________________

## 2. Target YAML schema

```yaml
# roles: top-level block (was launcher.meshes).
roles:
  trainer:
    hardware:
      type: npu                   # one of: npu | gpu | cpu
      model: 910b                 # optional: specific chip family
      min_memory_gb: 64           # per-device minimum
    placement:
      node_selector: {group: train}   # match nodes with label group=train
      count: 1                        # number of NODES (not procs)
    procs: 4
    gpus_per_proc: 1
    role_type: training_backend   # enum; drives actor-spawn behavior

  generator:
    hardware:
      type: gpu                   # can differ from trainer!
      model: h100
      min_memory_gb: 80
    placement:
      node_selector: {group: infer}
      count: 2                    # two independent replicas
    procs: 1
    gpus_per_proc: 4              # TP=4 inside one proc
    role_type: inference_engine

  reward_model:                   # NEW — not declarable today
    hardware: {type: gpu, model: a100, min_memory_gb: 40}
    placement:
      node_selector: {group: rm}
    procs: 1
    gpus_per_proc: 1
    role_type: reward_model

  experience_buffer:              # NEW — not declarable today
    hardware: {type: cpu, min_memory_gb: 512}
    placement:
      node_selector: {group: store}
    procs: 2
    role_type: replay_buffer

  storage:                        # today's torchstore volumes
    hardware: {type: any, requires_rdma: true}
    placement:
      colocate_with: trainer      # pin to same hosts as trainer role
    procs: 4
    role_type: weight_storage

# Transport declarations pair-wise between roles. Each entry picks
# the right mode + backend for that pair's workload.
transports:
  - name: weight_push
    from: trainer
    to: storage
    mode: one_sided_rdma
    backend: hixl

  - name: weight_pull
    from: storage
    to: generator
    mode: collective_broadcast
    backend: hccl
    config: {shard_src: 0}

  - name: rm_inference
    from: generator
    to: reward_model
    mode: rpc
    backend: monarch

  - name: replay_append
    from: generator
    to: experience_buffer
    mode: rpc
    backend: monarch

  - name: replay_sample
    from: trainer
    to: experience_buffer
    mode: rpc_streaming
    backend: monarch

# Node profile: static description of each cluster member.  Lives in
# a separate file (node_profile.yaml) and gets merged by the
# launcher.  Hand-authored today; K8s discovery later.
# (not in the launcher YAML above; see node_profile.yaml schema in §4)
```

## 3. Contracts

### Role contract (driver side)

Every Role has a `role_type` from an enum (extensible via registry — same pattern as
rewards). The driver looks up `role_type` to decide:

- What actor class to spawn (`TrainerActor`, `GeneratorActor`, `RewardModelActor`, ...).
- What `ProcessConfig` to request from the Provisioner.
- Which transports reference this role.

New role types register with `@register_role("name")` — same shape as the reward
registry.

### Transport contract

A Transport describes **how two roles exchange data**. Today's weight sync is the
archetype (push leg + pull leg + transport choice). Generalizing:

```python
class Transport(Protocol):
    name: str
    from_role: str
    to_role: str
    mode: Literal["rpc", "rpc_streaming", "one_sided_rdma",
                  "collective_broadcast", "collective_scatter"]
    backend: str      # hixl | hccl | monarch | nccl | ...
    config: dict      # backend-specific

    async def initialize(self, from_actor, to_actor, cluster): ...
    async def shutdown(self): ...
```

Concrete implementations match `mode × backend`:

- `("one_sided_rdma", "hixl")` → today's torchstore-over-HiXL
- `("collective_broadcast", "hccl")` → today's `CollectiveBroadcastBackend`
- `("rpc", "monarch")` → plain Monarch endpoint call (trivial)
- `("rpc_streaming", "monarch")` → streaming iterator over Monarch RPC
- `("collective_scatter", "nccl")` → future, for fine-grained weight sharding

Transports are **constructed by name lookup** at initialization, similar to weight-sync
backends today.

### Placement contract (Provisioner side)

Provisioner receives a list of Roles + node_profiles. It must:

1. For each role, find the set of nodes that satisfy `hardware` +
   `placement.node_selector`.
1. Honor `colocate_with` / `anti_colocate` constraints.
1. Reserve the requested `count` of nodes for the role.
1. Allocate `procs × gpus_per_proc` devices per node from the matched node's free-device
   pool.

No global optimizer needed at first — a greedy pass is fine for the sizes we target (≤
32 nodes). Upgrade to LP / Hungarian matching only when we have a cluster that needs it.

## 4. Node profile (new file)

`forge/configs/node_profile.yaml` (or similar, hand-authored):

```yaml
nodes:
  monarch1:
    labels: {group: train, region: cluster-a}
    hardware:
      devices:
        - {type: npu, model: 910b, count: 8, memory_gb: 64}
      cpu_cores: 128
      memory_gb: 512
      rdma: true

  monarch2:
    labels: {group: infer, region: cluster-a}
    hardware:
      devices:
        - {type: npu, model: 910b, count: 8, memory_gb: 64}
      cpu_cores: 128
      memory_gb: 512
      rdma: true

  rm_host_1:
    labels: {group: rm, region: cluster-b}
    hardware:
      devices:
        - {type: gpu, model: a100, count: 1, memory_gb: 40}
      cpu_cores: 64
      memory_gb: 256
      rdma: false
```

Per-cluster; reusable across multiple experiments on the same cluster.

## 5. Backward compat

`launcher.meshes.*.host_idx` keeps working for one release:

- `LauncherConfig.__post_init__` detects legacy `meshes` block and translates to an
  implicit `roles:` block using a homogeneous default node profile (every worker in
  `bare_metal.workers` gets a synthetic label). All existing smoke runs keep producing
  identical output.
- New YAMLs use top-level `roles:` + `transports:` and skip `launcher.meshes`.

## 6. What becomes trivial after this lands

Problems that are hard today but fall out for free:

- **RM on a separate GPU node**: add one `roles.reward_model` entry.
- **Replay buffer in CPU-heavy box**: add `roles.experience_buffer`, reference it from
  two transports. Trainer and generator use the replay buffer without caring where it
  runs.
- **Heterogeneous training** (ground-truth gen on A100, policy on 910b): two roles with
  different `hardware.type`, a transport between them.
- **Dedicated PS cluster** (N storage nodes): change `roles.storage` `count: 1` →
  `count: N`, the rest of the system doesn't need to know.
- **Async RL**: storage transport's `mode` set to `async_pub_sub`; trainer doesn't block
  on pull.

## 7. Implementation order (proposed — not committed)

Three phases, each independently shippable:

### Phase R1: Role block parses, legacy meshes still work

- Add `@dataclass class RoleConfig` / `RoleHardwareConfig` / `RolePlacementConfig`.
- `LauncherConfig.roles: dict[str, RoleConfig]`.
- `LauncherConfig.__post_init__` translates legacy `meshes` dict to `roles`.
- No behavior change; every existing YAML continues to produce the same spawn pattern.

### Phase R2: Placement scheduler honors hardware + node_selector

- `forge/configs/node_profile.yaml` schema.
- `BareMetalLauncher` reads node_profile, matches role hardware requirements, fills in
  host indices.
- Legacy YAMLs (no node_profile) assume all-homogeneous and behave identically.
- **First real new capability**: `roles.reward_model` can sit on a different hardware
  profile from `trainer`.

### Phase R3: Transport abstraction

- Collapse `WeightSyncBackend` registrations into the `Transport` registry.
  `torchstore_multi_vol` and `collective_broadcast` become transport mode × backend
  combinations.
- Add new transport types as needed (rpc, rpc_streaming).
- Rewrite `grpo.py` weight-sync path to use transport-by-name.
- Ship RM / Replay Buffer roles with transports between them.

Each phase is ≈ 3-5 days of focused work. Total ≈ 2 weeks of architecture work to land
all three.

## 8. What we are NOT doing in this arc

- K8s / Slurm launcher implementations (separate workstream).
- Dynamic scaling (adding/removing replicas mid-run). Phase R1-R3 is static-YAML-driven.
- Fault tolerance / role restart. Out of scope; once roles exist, a supervisor can be
  layered on top.
- Cross-cluster / federated training. Out of scope.

These are all reachable from the Role abstraction if we ever want them; they're not
blockers for it.

## 9. Next action

Agreement needed before code changes:

- [ ] Phase R1 goes first (Role block + legacy translation). **Yes.**
- [ ] Phase R2 goes second once there's a concrete new role needing non-trivial hardware
  (likely RM).
- [ ] Phase R3 happens when there's a concrete new transport that doesn't fit the
  current weight-sync abstraction.

Rule of thumb: **don't build an abstraction until the second user of it appears.** Today
we have one weight-sync-like pair (trainer ↔ storage ↔ generator). That's one user.
Phase R3 waits for the second.

The one thing we can do today that's risk-free is the Role parse work (R1) — no new
behavior, just a schema translation that unblocks everything else.
