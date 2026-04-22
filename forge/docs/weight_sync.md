# Weight Sync Architecture (Forge / HiXL)

Living doc for how weight synchronization works in forge's 2-node GRPO stack today, how
we got here, and what the next hard problem is. Read top-down for orientation; the
per-section headings are stable anchors for future cross-linking.

## 1. What this stack does

Every training step a `TrainerActor` (TorchTitan FSDP2, 4 NPUs on one host) produces
updated model weights. Those weights have to land inside the `Generator`'s vLLM engine
(1 NPU on the other host) before the next rollout can use them. "Weight sync" is this
push+pull cycle:

```
 trainer_rank_0                                    worker (vLLM)
 TorchTitan FSDP2 ---- torchstore volume ---->    inplace copy into
 gather state_dict        (RDMABuffer)             model params via
                                                   HiXL RoCE
```

Two physical transports are in play:

- **HiXL HCCS** — Huawei on-host 2 MiB-aligned NPU DMA (same-host npu→npu).
- **HiXL RoCE** — 200 Gb/s per-NPU RDMA over Ethernet (cross-host).

Everything else — torchstore, Monarch, the WeightSyncService, the launcher — is
orchestration around these two primitives.

## 2. Layered architecture

```
┌─────────────────────── control plane (Monarch RPC over TCP) ──────┐
│                                                                    │
│   Driver proc ── WeightSyncService ── backend plugin               │
│       │             │                     │                        │
│       │             │                     ▼                        │
│       │             │              MultiVolTorchstoreBackend       │
│       │             │                     │                        │
│   BareMetalLauncher + ProcMesh lifecycle │                        │
│       │                                   │                        │
│       ├── spawn trainer procs (FSDP)     │                        │
│       ├── spawn generator procs (vLLM)   │                        │
│       └── spawn storage procs (torchstore volumes)                │
│                                                                    │
└──────────────────────┬─────────────────────────────────────────────┘
                       │                  (one-sided RDMA, no Python)
┌──────────────────────┴──── data plane (HiXL HCCS / RoCE) ─────────┐
│                                                                    │
│   trainer rank 0 flat_buffer -->  storage vol 0 RDMABuffer         │
│                                    (colocated or on other host)   │
│                                          │                         │
│                                          ▼                         │
│           worker flat_buffer  <--  ts.get(key, inplace=flat)       │
│                  │                                                 │
│                  ▼                                                 │
│           direct:  param.data.copy_(view)                          │
│           fused:   model.load_weights([(hf_name, view), ...])      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Why the planes are split**: control-plane RPC carries only KB-sized metadata
(versions, plans, keys, status) and can take the Python / TCP hit. Data-plane moves GB
of tensor bytes and must never visit Python — RDMA one-sided direct from source NPU to
destination NPU is the only acceptable path.

## 2b. Rank taxonomy — which "rank" at which layer

Four independent `rank` spaces are in play at once. Mixing them up is the single biggest
source of confusion when reading logs; the probe in commit `3b7196a5` was to
disambiguate them on our running system.

| rank space                 | where it lives                                                             | how to read it                                           | example (trainer rank 2)                                              |
| -------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------- |
| **Monarch mesh rank**      | per ProcMesh; always 0..mesh_size-1                                        | `context().actor_instance.rank.rank`                     | 2                                                                     |
| **torch.distributed rank** | set by TitanTrainer's SPMD setup into `RANK` / `LOCAL_RANK` env            | `dist.get_rank()` / `os.environ["LOCAL_RANK"]`           | 2                                                                     |
| **HCCL comm rank**         | per HCCL communicator; FSDP gather and each HiXL put use *different* comms | inside HCCL log lines, `rank[<N>]`, `deviceLogicId[<M>]` | FSDP: 2; HiXL put: 0 (it's a 2-rank comm, client=rank0, server=rank1) |
| **physical NPU id**        | hardware; fixed per chip                                                   | `torch.npu.current_device()` after `set_device`          | 2 (for trainer rank 2)                                                |

Each mesh's internal rank restarts from 0:

```
trainer mesh   (4 proc)  → ranks 0, 1, 2, 3
generator mesh (1 proc)  → rank 0
storage mesh   (4 proc)  → ranks 0, 1, 2, 3
```

Monarch has no notion of a "global rank" across meshes — if you need one you build it
yourself.

### Mapping mesh rank → physical NPU

Decided per-mesh, at proc startup, via either the bootstrap (storage) or TitanTrainer's
SPMD setup (trainer):

| mesh                | bootstrap / setup did                                                       | physical NPU                     |
| ------------------- | --------------------------------------------------------------------------- | -------------------------------- |
| trainer             | no mask; `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` + `torch.npu.set_device(rank)` | `rank` on monarch1 (driver host) |
| storage (colocated) | `_storage_bootstrap_factory(npu_base=4)` masks proc to one NPU              | `4 + rank` on monarch1           |
| storage (dedicated) | same bootstrap, different host                                              | `4 + rank` on monarch2           |
| generator           | vLLM's own setup, single proc                                               | NPU 0 on monarch2                |

**Why trainer doesn't use `ASCEND_RT_VISIBLE_DEVICES` masking while storage does**:

- *trainer*: FSDP's 4-way HCCL comm needs every rank to see each peer's `device_ip` and
  resolve HCCS topology; masking each proc to a single NPU breaks that discovery. So
  trainer procs see all 4 NPUs and pick their own via `set_device(rank)`.
- *storage*: each volume proc is isolated — it does not HCCL with its sibling volumes.
  The torchstore transport layer currently hardcodes `npu:0` as the staging pool device,
  which only works if each proc has been masked so that "npu:0" is the one physical NPU
  you want (`4+i` here).

### The HCCL-per-put twist

Every `ts.put` / `ts.get` under `MonarchRDMATransportBuffer` builds a fresh **2-rank
HCCL comm** on the fly (client=rank 0, server=rank 1). The rank numbers inside these
ephemeral comms are local to each comm and unrelated to the Monarch mesh rank. So
trainer mesh rank 2, doing a `ts.put`, shows up in HCCL logs as "rank\[0\],
deviceLogicId\[2\]" inside an `[hixl]192.168.0.23:X_192.168.0.23:Y` comm. None of this
conflicts with its FSDP HCCL rank, which is 2 in a different comm.

Concretely, a trainer proc simultaneously participates in:

1. Monarch `trainer` ProcMesh (rank=2, `_host_id`=UUID-of-monarch1)
1. FSDP 4-way HCCL comm (rank=2, `deviceLogicId`=2)
1. HiXL put-side 2-rank comm (rank=0, `deviceLogicId`=2)
1. Physical NPU 2 (`torch.npu.current_device()`=2)

When a log line prints a rank, ALWAYS check which comm / mesh it belongs to before
drawing any conclusion.

## 3. The flat-buffer fast path

Our measured fast path for a ping-pong update is driven by a single RDMA transfer on
each end:

1. Trainer rank 0 does FSDP `state_dict_for_sync()` (the full-gather collective);
   materialises `DTensor.full_tensor()` for every parameter; translates TorchTitan names
   → HF names via `StateDictAdapter.to_hf()`.
1. Rank 0 packs every HF tensor into one 2 MiB-aligned NPU flat buffer
   (`alloc_aligned_tensor`). The per-param layout
   `[(name, shape, dtype, offset, nbytes), ...]` is cached as the *plan*.
1. Rank 0 issues **one** `ts.put(key, flat)`. torchstore's `MonarchRDMATransportBuffer`
   wraps the byte view in a `RDMABuffer`; because the source is already pool-aligned,
   the staging-pool detour (`pool.alloc + staged.copy_`) is skipped and HiXL RDMA reads
   straight from `flat` to the storage volume's pool slot.
1. Worker allocates a matching 2 MiB-aligned flat buffer; issues one
   `ts.get(key, inplace_tensor=flat)`. Storage volume HiXL-writes the whole thing across
   RoCE in a single `TransferSync`.
1. Worker walks the plan, splits each (name, offset, nbytes) into two buckets:
   - **direct**: name present in `model.named_parameters()` → `param.data.copy_(view)`
     (NPU→NPU DMA).
   - **fused**: name not present (Qwen3 `gate_proj`/`up_proj` etc. collapsed to
     `gate_up_proj` by vLLM) → accumulate into a list and hand to one bulk
     `model.load_weights(pairs)` call so vLLM's WeightsMapper concatenates in one pass.
1. Worker retains a ref to `flat` until the next pull so HiXL's registration isn't torn
   down mid-unpack (fused params keep views into it).

### Why not per-parameter ts.put / ts.get

Torchstore's `MonarchRDMATransportBuffer` inherits `supports_batch_gets = False` from
the base class, so even `ts.get_batch({311_keys})` falls through to a serial
`for request in requests: _get_requests([request])` loop inside the library. For
Qwen3-0.6B that means 311 × ~4 ms/key of HiXL `register_mem` + handshake overhead ->
~1.2 s/sync floor. Flat-buffer collapses all 311 keys into 1 handshake + 1 register + 1
TransferSync; the transfer itself then runs at wire speed.

## 4. Measured performance (Qwen3-0.6B, batch=8, max_new_tokens=128)

End-to-end weight sync wall time, steady state (step 2+, after HCCL/HiXL connections
warm up):

| generation                          | weight_sync | worker pull | effective BW |
| ----------------------------------- | ----------: | ----------: | -----------: |
| initial CPU round-trip              |        14 s |          -- |    0.10 GB/s |
| WeightSyncService + inplace direct  |       4.4 s |       2.9 s |    0.48 GB/s |
| asyncio.gather concurrent per-param |       2.8 s |      1.26 s |    1.11 GB/s |
| **flat-buffer fast path (current)** |   **0.4 s** |  **0.11 s** | **~13 GB/s** |

**Rate compared to hardware ceiling**: the standalone
`forge/scripts/test_weight_sync_2node.py` harness hits ~17 GB/s on the same fabric with
pure `alloc_aligned_tensor + RDMABuffer`. The remaining ~25% gap vs our in-training path
is torchstore's handshake + RDMABuffer lifecycle (one `ts.put` / `ts.get` pair still
wraps around ~5 Monarch RPCs per cycle for state metadata). Not on the critical path —
rollout dominates by 150x.

### Leg choice (HCCS vs RoCE) is a wash at this size

Two YAML layouts tested side-by-side with everything else constant:

| layout       | put leg              | pull leg            | steady pull | workers_load |
| ------------ | -------------------- | ------------------- | ----------: | -----------: |
| colocated PS | trainer→storage HCCS | storage→worker RoCE |      0.11 s |       0.10 s |
| dedicated PS | trainer→storage RoCE | storage→worker HCCS |      0.11 s |       0.10 s |

At 1.4 GiB the bottleneck is torchstore/HiXL fixed setup, not the bytes-in-flight leg.
Larger models (Qwen3-7B = ~15 GiB) should start to show an asymmetry and that's when the
knob becomes interesting. Flipping layout is one line in
`forge/configs/launcher_bare_metal_2node[_dedicated_ps].yaml`:

```yaml
meshes:
  storage: {host_idx: 1}   # colocated with trainer
  # storage: {host_idx: 0} # dedicated on the other host
```

## 5. Extension points

### 5.1 Pluggable backends (`forge/engines/weight_sync/backends/`)

`create_backend(name, **kwargs)` returns a `WeightSyncBackend` implementing `push` /
`pull` / `initialize` / `shutdown`. Today:

- `torchstore_multi_vol` (default) — N storage volumes, flat-buffer path, colocated or
  dedicated.
- `p2p_rdma`, `dedicated_ps`, `areal_xccl` — placeholder classes with design-sketch
  docstrings; not implemented.

To add a backend: register a loader in `forge/engines/weight_sync/backends/__init__.py`
and implement the `WeightSyncBackend` protocol (`forge/engines/weight_sync/service.py`).

### 5.2 Storage mesh injection (commit `031e326a`)

`MultiVolTorchstoreBackend.__init__(storage_mesh: ProcMesh | None = None, ...)`. When
`storage_mesh` is passed in, the backend skips its internal spawn and just runs
`ts.initialize(mesh=...)` on the supplied `ProcMesh`. This is what makes colocated vs
dedicated PS a YAML toggle: the driver (`grpo.py::_spawn_storage_mesh`) picks the host
mesh based on `FORGE_STORAGE_HOST_MESH`, spawns N storage procs there, and hands the
mesh to the backend.

### 5.3 Launcher YAML (commit `317e5cc5`)

`forge/configs/launcher_bare_metal_2node.yaml` is the canonical topology description:

```yaml
launcher:
  type: bare_metal
  bare_metal:
    workers: [tcp://.../22222, ...]   # order matches hostfile
    master_addr: ...
    worker_port: 22222
  meshes:
    trainer:   {host_idx: 1}   # documentation; trainer uses this_host()
    generator: {host_idx: 0}
    storage:   {host_idx: 1}   # flip to 0 for dedicated PS
  weight_sync:
    backend: torchstore_multi_vol
    storage_mesh: storage
    pool_mb: 6144
    storage_npu_base: null       # null = auto = train_world_size
```

Precedence: CLI > env > YAML > dataclass default. `run_multinode.sh` injects a
hostfile-derived `--bare-metal-workers` at CLI level (so dynamic hostfiles work without
YAML edits), but everything else funnels through the YAML once `--forge-config` is set
(either explicitly or via the `FORGE_CONFIG` env var).

### 5.4 Optional shard-parallel publish (commit `1b767b66`)

Opt-in via `FORGE_SHARD_PUBLISH=1`. Scaffolding is ready:

- Trainer path: every rank packs + puts a HIXL_BLOCK-aligned byte-shard of the flat
  buffer under `{key}.shard_{rank}`.
- Backend caches `shard_ranges` in `_push_flat` and forwards them in `_pull_flat`.
- Worker path: `asyncio.gather(ts.get(shard_key_i, inplace=flat[start:end]))` — N RoCE
  links pulling in parallel.

**Status: not working end-to-end yet.** Blocks on a torchstore
MonarchRDMATransportBuffer limitation (see §7 — Follow-ups below). Keep the switch OFF
until that's patched.

## 6. Known-pain ledger

Each item below was a multi-hour debug session; the commit hash cited is the one that
either fixed or documented the trap.

| #   | symptom                                                                     | root cause                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | mitigation                                                                                                                                                                                                                                             | commit               |
| --- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| 1   | 14 s weight sync, 0.10 GB/s, tensors on CPU                                 | state_dict → `.cpu()` → pickle → RPC → state_dict on inference side                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Workers ts.get(inplace=param.data); no Python state_dict over the wire                                                                                                                                                                                 | `aef73317`           |
| 2   | 0.48 GB/s even after inplace                                                | 311 serial ts.get calls — torchstore `supports_batch_gets=False` defeats `ts.get_batch`                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `asyncio.gather(*[ts.get(...)])`                                                                                                                                                                                                                       | `1b53d107`           |
| 3   | 1.1 GB/s still capped                                                       | pool staging: 2 NPU-NPU copies per ts.get                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | flat-buffer with `alloc_aligned_tensor` + 1 key; pool staging skipped because flat is already pool-aligned                                                                                                                                             | `b18c72ef`           |
| 4   | small-tail param `hixl_transfer_write ret=503900`                           | vLLM's param.data is not 2 MiB aligned                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | single flat buffer, not per-param                                                                                                                                                                                                                      | `b18c72ef`           |
| 5   | NPU 0 OOM at weight sync step                                               | pool_mb=8192 sat on the edge of NPU 0's free space after rollout                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | default pool_mb=6144 in YAML                                                                                                                                                                                                                           | `28caa080`           |
| 6   | YAML pool_mb=6144 ignored                                                   | `run_multinode.sh` eagerly exported the env var with its own default, winning precedence                                                                                                                                                                                                                                                                                                                                                                                                                                                      | conditional export: only forward if caller actually set it                                                                                                                                                                                             | `317e5cc5`           |
| 7   | reward > 0 but model not updating (!!)                                      | worker pull_weights_flat returned success=False, backend treated num_keys as success                                                                                                                                                                                                                                                                                                                                                                                                                                                          | raise on empty plan; any-worker-fail → num_keys=0                                                                                                                                                                                                      | `28caa080`           |
| 8   | OOM on generator NPU 0, 613 MiB free                                        | YAML `generator.host_idx: 1` landed generator on the trainer host (NPU 0 already busy)                                                                                                                                                                                                                                                                                                                                                                                                                                                        | flip host_idx to match workers-array indexing (hostfile order)                                                                                                                                                                                         | `28caa080`           |
| 9   | Same physical host got multiple GpuManagers, NPUs overlap                   | launcher returned a fresh slice object for each new mesh name, Provisioner stamped unique `_host_id`                                                                                                                                                                                                                                                                                                                                                                                                                                          | `BareMetalLauncher._slice_by_worker_idx` cache                                                                                                                                                                                                         | `317e5cc5`           |
| 10  | `worker_manager.sh: MonarchRDMATransportBuffer.allocate: command not found` | backticks in rst/md-style comment command-substituted inside an unquoted heredoc                                                                                                                                                                                                                                                                                                                                                                                                                                                              | strip backticks from shell-embedded comments                                                                                                                                                                                                           | `2abe4313`           |
| 11  | shard-publish: `hccl_comm.cc:132 errNo 0x0000000005000007`                  | **Updated after LOCAL_RANK probe**: trainer procs are NOT `ASCEND_RT_VISIBLE_DEVICES`-masked -- they see all 4 NPUs and pick one via `torch.npu.set_device(rank)`. The real cause is every trainer proc inheriting `TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:0`, so every rank's torchstore staging pool lives on NPU 0 and every RDMABuffer registers NPU 0 memory; HCCL rankTable ends up with `device_id:0` for every rank's local endpoint. Plus HiXL-side rankTable dependency makes multi-client HCCL init fail at the cluster layer. | (a) env-per-rank fix in `publish_weights_flat`: `TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:{rank}` when shard-publish is on; (b) HiXL team removing rankTable dependency end-of-month. Both required for `FORGE_SHARD_PUBLISH=1` to actually succeed. | `1b767b66` + pending |

Non-obvious invariants the next hacker will hit if not careful:

- `run_multinode.sh` reads the **second** line of `hostfile.txt` as the driver (so
  `this_host()` resolves there), but `--bare-metal-workers` is built in hostfile
  *order*. `host_idx` is an index into the workers array, not into hostfile lines.
  Getting this wrong costs a whole debugging session (see trap #8).
- `TrainerActor.options(...)` does **not** pass `hosts=1`, so the trainer goes through
  `this_host()` rather than the launcher. YAML's `trainer.host_idx` is
  documentation-only today.
- Generator/worker `print()` does **not** propagate to driver stdout. For debugging,
  make the backend log the per-worker `PullResult.per_worker` messages from the driver
  side (see `_pull_flat` in `torchstore_multi_vol.py`). Trap #7's root cause was silent
  pull failures that looked like success at the driver.
- `TORCHSTORE_MONARCH_RDMA_POOL_MB` must agree between the trainer proc
  (driver-inherited) and storage/worker procs (worker_manager.sh heredoc).
  `run_multinode.sh` now re-exports the value so all procs land on the same number;
  don't let this regress.

## 7. Follow-ups (ranked)

### 7.1 Shard-parallel publish — two independent fixes needed

Updated diagnosis after running the LOCAL_RANK probe (see trap #11 for the full
rewrite). The original hypothesis — "torchstore bakes NPU 0 + NPU base into rankTable
and trainer procs are masked" — was wrong. Actually: trainer procs see all NPUs and use
`torch.npu.set_device(rank)`; the rankTable collision comes from
`TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:0` inherited by every trainer proc, and from
HiXL's choice of the rankTable-based HCCL init API (as opposed to root-info).

**Fix (a) — forge side, env-per-rank pool device.** Landed in
`TrainerActor.publish_weights_flat`: when `FORGE_SHARD_PUBLISH=1`, each trainer proc
monkey-patches `torchstore.transport.monarch_rdma._STORAGE_DEVICE = f"npu:{rank}"` (and
nulls `_GLOBAL_POOL`) before building its flat buffer, so every rank's torchstore
staging pool lives on its own NPU rather than NPU 0. (Note: we rewrite the module
constant rather than the env var because torchstore caches
`_STORAGE_DEVICE = os.environ.get(...)` at import time, and the module is imported long
before this endpoint runs.) Correct and necessary fix — but validated on 2026-04-22 to
be insufficient by itself.

**Fix (b) — HiXL / CANN side, drop the rankTable dependency.** HiXL currently uses
`HcclCommInitClusterInfoMemConfig(rankTable_json, ...)` which requires a rankTable
matching physical device ids. HCCL itself has a second init path
(`HcclCommInitRootInfo`) that negotiates peers at runtime — the same path
`torch.distributed(backend="hccl")` uses for FSDP's 4-rank all-gather, which already
works. The HiXL team is scheduled to switch HiXL's internal comm init to root-info by
**end of the month**.

**2026-04-22 iterative drill-down** — the original 2-fix plan turned out to split into
**three** sub-bugs. We've fixed two, one remains.

### Bug A' — trainer-side rankTable local device_id (FIXED)

Found the actual code path by reading HiXL + monarch source: `HixlManagerActor::new` ->
`monarch_rdma::backend::hixl::manager_actor::resolve_device_id(hint)`, which returns
`hint` if non-negative and otherwise reads env `MONARCH_NPU_DEVICE` (default 0). Storage
bootstrap sets `MONARCH_NPU_DEVICE=0` per-proc (correct under ASCEND_RT_VISIBLE_DEVICES
masking), but **trainer procs never set it**. Every trainer rank's HiXL init went
through with logical device 0, and `aclrtGetPhyDevIdByLogicDevId(0, ...)` returned NPU 0
regardless of which trainer rank was running — hence the "every trainer rank's rankTable
local endpoint is NPU 0" symptom.

Fix landed in `TrainerActor.publish_weights_flat` (shard-publish branch): set
`os.environ["MONARCH_NPU_DEVICE"] = str(rank)` before the first `ts.put` (i.e. before
`HixlManagerActor::init` fires). Trainer procs are not
`ASCEND_RT_VISIBLE_DEVICES`-masked, so logical device id equals the trainer rank. The
staging-pool monkey-patch (`_STORAGE_DEVICE = f"npu:{rank}"`) also still required.

Verified with `FORGE_SHARD_PUBLISH=1` smoke 2026-04-22: rankTable local endpoint now
correctly carries NPU 0/1/2/3 and matching `device_ip`s:

```
rank 0 local: {"device_id":"0", "device_ip":"29.191.188.114", "deviceLogicId":0}
rank 1 local: {"device_id":"1", "device_ip":"29.191.181.122", "deviceLogicId":1}
rank 2 local: {"device_id":"2", "device_ip":"29.191.84.209",  "deviceLogicId":2}
rank 3 local: {"device_id":"3", "device_ip":"29.191.87.140",  "deviceLogicId":3}
```

### Bug B — storage-side bootstrap is silently skipped; all vols collapse to NPU 4

Even with Bug A' fixed, every trainer rank's rankTable **peer half** reads identically
to NPU 4:

```
{"device_id":"4", "device_ip":"29.191.56.199", "rank_id":"1"}
```

**LocalRankStrategy ruled out.** Instrumented `torchstore.client.put_batch` 2026-04-22
and confirmed routing is per-rank-correct:

```
[torchstore-dbg] client RANK='0' -> volume_id='0'
[torchstore-dbg] client RANK='1' -> volume_id='1'
[torchstore-dbg] client RANK='2' -> volume_id='2'
[torchstore-dbg] client RANK='3' -> volume_id='3'
```

Four trainer ranks each send their `ts.put` to a distinct storage volume. The collapse
happens *below* torchstore's strategy.

**Smoking gun — `_storage_bootstrap_factory` doesn't actually run**. We added
marker-file writes inside the bootstrap closure and re-ran the smoke; **no marker files
appeared** on either host, for any of the 4 storage vol procs. The driver side still
prints `[WeightSync] spawned 4 storage volumes on host mesh 'storage' (NPU range 4..7)`
so the spawn succeeded, but the per-proc bootstrap callback that was supposed to

- set `ASCEND_RT_VISIBLE_DEVICES=npu_base+local_rank` per-proc,
- set `MONARCH_NPU_DEVICE=0` per-proc,
- call `torch.npu.set_device(0)`,

never fires. With none of this happening on the storage side, all 4 storage vol procs
inherit the parent env, `resolve_device_id` falls back to 0 for every vol,
`hixl_init_engine(dev=0, ...)` runs in each, and the ACL proc-default device settles on
the same NPU across all four. Result: every vol advertises the same physical NPU in its
HiXL engine handshake, and every trainer rank's rankTable peer side reads the same.

**Why bootstrap is silently skipped**: Monarch's `ProcMesh.spawn_procs(bootstrap=...)`
path has known stability issues. torchforge works around it by using an `EnvSetter`
actor instead (see `torchforge/src/forge/controller/provisioner.py::EnvSetter`,
docstring: "This replaces the old bootstrap approach to avoid Monarch's SetupActor mesh
failures on shutdown. ... we will move back to bootstrap once it's fixed"). Our
`_storage_bootstrap_factory` sits on the same bootstrap path, so on our monarch build
the closure is effectively dead code.

**Fix direction**: migrate storage spawn from `bootstrap=...` to the `EnvSetter`-style
pattern (same as torchforge GPU path):

1. Spawn storage procs with no bootstrap.
1. Spawn a tiny setter actor (`EnvSetter`) on the storage ProcMesh.
1. Call its `set_env(env_vars)` endpoint; each proc writes `ASCEND_RT_VISIBLE_DEVICES` /
   `MONARCH_NPU_DEVICE` / etc. into its own env.
1. *Then* proceed with `ts.initialize`.

~40 LOC in `grpo.py::_spawn_storage_mesh` + `torchstore_multi_vol.py`. Independent of
the HiXL end-of-month rankTable removal. Once it lands, together with the Bug A' fix
already deployed on the trainer side, the `FORGE_SHARD_PUBLISH=1` path should actually
run end-to-end on our setup.

### Bug C — HiXL rankTable dependency itself

Originally framed as "the fix". Now, with Bug A' fixed and the HiXL C++ inspected, we
know the rankTable mechanism **works correctly** for non-colliding (local_device,
peer_device) pairs — it just requires the caller to supply accurate device_ids on both
ends. Bug B is the remaining caller-side work; the HiXL end-of-month rankTable removal
is still welcome (simpler semantics, no caller-side device bookkeeping) but is no longer
on the critical path for enabling shard publish.

**Smoke plan when fix (b) lands**:

1. Update torchstore / monarch against the new CANN + HiXL libs.
1. Flip `FORGE_SHARD_PUBLISH=1`, run the 3-step ping-pong smoke.
1. Expect `push_flat`'s `put_s_max` to drop roughly 4x vs rank-0-only (ideal:
   `total_bytes / (world_size × single_nic_bw)`). `pull_flat` gains a smaller speedup
   from N volumes answering in parallel.
1. Watch for second-order effects: four 6 GiB staging pools (one per trainer NPU) may
   tighten NPU memory vs the rank-0-only layout; possibly need to drop `pool_mb` for the
   shard path.

**Also related, not blocking**: once shard-publish runs, **B1.2** (per-rank
`DTensor.to_local()` packing instead of every rank gathering the full HF state_dict)
gives the 4x peak-memory reduction. Separate change in `state_dict_for_sync` +
`publish_weights_flat`; depends on the shard-publish plumbing being functional first.

### 7.2 HiXL CreateChannel 103901 at pool_mb=4096

Surfaced in a Step 2 experiment. Only reproducible at a specific pool size and not on
the default path. Low priority but a tripwire for anyone trying to push pool_mb low for
bigger models on tighter NPUs.

### 7.3 Architecture / production-grade extensions

Enumerated in the original "infra design" discussion — not started yet, in rough
dependency order:

- **`ps_world` and backend for a cross-host PS cluster.** YAML already has the hook.
- **Version & session manager.** `Version` as immutable snapshot, `Session` as
  per-consumer subscription with staleness bound. Opens the door to async RL /
  sliding-window retention.
- **`DirectMonarchBackend`.** No torchstore; trainer registers a `RDMABuffer`, handle is
  RPC'd to worker, worker reads directly. Experiment to measure whether torchstore is
  net-positive for our specific 1-producer-N-consumer shape.
- **`HoldingActorMesh`.** Thin in-house holder if torchstore's abstraction tax starts to
  bite for more use cases than weight sync (e.g. MoE expert rotation).

## 8. Commit trail

Forward reading order, with the one-liner for each:

```
aef73317 feat(forge): WeightSyncService with pluggable backends
1b53d107 perf(forge): parallel ts.get + bulk load_weights in pull_weights
b18c72ef perf(forge): flat-buffer fast path for weight sync (~35x total)
031e326a refactor(forge): hoist storage-mesh spawn out of the backend
2abe4313 feat(forge): explicit mesh placement + pool-mb env knob
317e5cc5 feat(forge): YAML-driven launcher + fix slice-per-worker cache
28caa080 fix(forge): YAML host_idx flipped + surface silent weight-sync failures
cdbed36a feat(forge): add dedicated-PS launcher variant + workers-order note
1b767b66 feat(forge): scaffold shard-parallel publish behind FORGE_SHARD_PUBLISH
```

The first three are the weight-sync performance ladder (14 s → 0.4 s). The next six are
the topology-abstraction series (storage spawn out of backend → YAML driving →
dedicated-PS variant → shard-parallel scaffolding). Read them in order to reconstruct
the design history.
