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
- **HCCL over RoCE** — collective comm (broadcast / allreduce / ...) via
  `torch.distributed` (added 2026-04-22 for the collective-broadcast pull backend,
  §7.4). Uses the same RoCE NICs as HiXL but with HCCL's own transport and rankTable
  semantics.

Everything else — torchstore, Monarch, the WeightSyncService, the launcher — is
orchestration around these three primitives.

**Backend options today** (`FORGE_WEIGHT_SYNC_BACKEND=...`):

- `torchstore_multi_vol` (default) — every inference TP worker independently does
  `ts.get` against N storage vols. Works great at TP=1; scales connection count as
  `TP × num_vols` and breaks at TP>1 with HiXL `CreateChannel 103901` (see §7.4).
- `collective_broadcast` — opt-in for TP>1. Trainer's HiXL put is unchanged, but pull
  goes through a `1 + TP` HCCL broadcast sourced from storage vol 0 and fanned out to
  every TP worker. See §7.4 for the full design, debug trail, and measured numbers.

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
`forge/configs/clusters/2node_{colocated,dedicated_ps}.yaml`:

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

`forge/configs/clusters/2node_colocated.yaml` is the canonical topology description:

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

### Bug B — storage-side bootstrap is silently skipped (FIXED)

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

**Fix landed 2026-04-22** (commit `eca54172`). Migrated storage spawn from
`bootstrap=...` to an `EnvSetter`-style pattern (same shape as torchforge's GPU path):

1. Spawn storage procs with no bootstrap (`grpo.py::_spawn_storage_mesh`).
1. Spawn a `StorageEnvSetter` actor on the storage ProcMesh
   (`forge/engines/weight_sync/_env_setter.py`).
1. Call its `setup(npu_base, pool_mb)` endpoint; each proc writes
   `ASCEND_RT_VISIBLE_DEVICES` / `MONARCH_NPU_DEVICE` / HiXL / HCCL / torchstore env
   vars into its own env **before** importing `torch_npu`, then
   `torch.npu.set_device(0)` on the masked NPU.
1. The setter returns a per-rank status dict (`masked_device_count`,
   `masked_current_device`, `physical_npu`) so the driver can *observe* that masking
   actually took effect — the evidence we were missing under the silent-bootstrap
   regime.
1. *Then* proceed with `ts.initialize`.

Verified with `FORGE_SHARD_PUBLISH=1` smoke (Qwen3-0.6B, 1.4 GiB state_dict, 3-step
ping-pong):

```
env setup confirmed: [
  {local_rank=0, physical_npu=4, masked_device_count=1, masked_current_device=0},
  {local_rank=1, physical_npu=5, masked_device_count=1, masked_current_device=0},
  {local_rank=2, physical_npu=6, masked_device_count=1, masked_current_device=0},
  {local_rank=3, physical_npu=7, masked_device_count=1, masked_current_device=0},
]

push_flat(v3): ranks_seen=[0,1,2,3],
               shard_ranges=[(0,0,375M), (1,375M,751M), (2,751M,1126M), (3,1126M,1504M)],
               put_s_max=0.03
pull_flat(v3): num_keys=311, pull_s=0.11, workers_load_s=0.10
Weight sync: 0.4 s
```

Each storage vol is correctly pinned to its own physical NPU; trainer rankTable peer
halves carry NPU 4/5/6/7 (one per comm); HCCL inits succeed concurrently on all 4 ranks;
push_flat puts proceed in parallel across 4 NICs. Steady `put_s_max=0.03s` vs
rank-0-only's `0.07s` — the 4 NIC parallelism kicked in as designed.

Total weight sync stays at 0.4s because **pull is still single NIC** (generator runs
TP=1, one worker proc, one NIC). Unlocking 4 NIC on the pull side is a separate piece of
work (generator TP>=2 with per-worker shard-fetch), tracked as §7.4 below.

One subtlety that almost reverted the win: the first revision of the trainer-side shard
fix nulled `_GLOBAL_POOL` at the top of *every* `publish_weights_flat` call to force
pool re-init on the right NPU. That worked on v1, but on v2 the pool teardown ran while
the previous cycle's storage-side handshake still held references to our RDMABuffer
registrations — leading to `hixl_transfer_read ret=503900` on server-side
`handle_put_request`. The fix is a `self._shard_env_set` guard so the pool gets
rewritten exactly once per trainer proc. See the trap-warning comment inline in
`publish_weights_flat`.

### Bug C — HiXL rankTable dependency itself

Originally framed as "the fix". With Bug A' and Bug B both fixed (and HiXL C++
inspected), we now know the rankTable mechanism **works correctly** for non-colliding
(local_device, peer_device) pairs — it just requires the caller to supply accurate
device_ids on both ends. Our 2-node setup runs the full shard-publish path today on the
existing HiXL rankTable API; the HiXL end-of-month rankTable removal is still welcome
(simpler semantics, no caller-side device bookkeeping) but is **no longer on the
critical path**.

**Follow-ups with rankTable removal**:

1. Update torchstore / monarch against the new CANN + HiXL libs when available.
1. Simplify `HixlManagerActor::new` path — no more `resolve_device_id` lookup, no more
   `MONARCH_NPU_DEVICE` env var, no more `_shard_env_set` guard in trainer. The current
   complexity is caller-side device bookkeeping, which becomes unneeded.
1. Re-run the 3-step ping-pong smoke to confirm the steady-state numbers don't regress.

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

### 7.4 Generator TP>1: CollectiveBroadcastBackend (landed)

**Status**: **SOLVED 2026-04-22**. Landed as
`forge/engines/weight_sync/backends/collective_broadcast.py` +
`torchstore/storage_volume.py` bcast endpoints. Flip
`FORGE_WEIGHT_SYNC_BACKEND=collective_broadcast` to use it.

#### 7.4.1 What the bug was

With `allocation_mode: vllm:d1p1t4+d4p1t1` the `Generator.options` refactor correctly
hands 4 NPUs to the driver proc, the vLLM executor spawns the 4 `vllm_workers`, HCCL
builds the TP group — and then `pull_flat(v1)` fails on the first weight sync with
`hixl_connect(...) failed: ret=103901 (CreateChannel)`.

Root cause: the original pull path calls `ts.get` **inside every
`WorkerWrapper.pull_weights_flat`**. At TP=1 that's 1 client × N storage vols = N HiXL
channels (fine). At TP=4 it's 4 × N = 16 concurrent HiXL client channels per step, which
exhausts HiXL's per-proc client resources. Not a HiXL "infra limit" bug — a correctness
gap in forge's pull-side scaling.

#### 7.4.2 Design: storage-as-reflector

Business decision tree we walked through before landing the code:

1. **Stay with torchforge-style shared-mem staging?** No — our volumes are already
   NPU-resident and we have RoCE RDMA to the peer host. torchforge's `SharedTensor` adds
   a device→host→device double-copy we don't need.
1. **AReaL-style direct trainer↔inference NCCL group?** No — forces a single process
   group spanning two independent Monarch meshes, and loses torchstore's async / version
   / fault-tolerance story.
1. **Storage-as-reflector.** Keep trainer's existing HiXL `ts.put` into storage volumes
   (device-resident buffer, no host hop). Replace per-worker `ts.get` with a single
   `dist.broadcast` sourced from storage vol 0 and fanned out to every TP worker over
   HCCL.

Connection count becomes `1 + TP` (handled by HCCL QP aggregation) independent of TP
degree. Storage continues to offer the async / version semantics torchstore always did —
trainers push on their own schedule, storage holds the tensor, bcast reflects it out on
demand.

#### 7.4.3 Three MVPs that validated the transport story

Before writing the backend, three standalone smoke scripts isolated the transport
properties we needed:

| MVP                                                | What it proved                                                        | Measured                                         |
| -------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------ |
| `forge/scripts/test_cross_mesh_bcast.py`           | Cross-Monarch-mesh HCCL group rendezvous works                        | 10.2 GB/s (2-rank, 1.5 GB payload)               |
| `forge/scripts/test_cross_mesh_bcast_1toN.py`      | 1-to-N HCCL bcast (1 src + TP=4 dst) fans out correctly               | per-link 8 GB/s, aggregate 32 GB/s               |
| `forge/scripts/test_cross_mesh_bcast_with_hixl.py` | HiXL put and HCCL bcast coexist in the same storage proc concurrently | put 20 GB/s, bcast agg 32 GB/s (no interference) |

Those three scripts remain the ground-truth harness for anyone poking at the collective
path without the full GRPO machinery in the way.

#### 7.4.4 Interface

Added to torchstore (strictly additive, doesn't affect existing volumes that never call
it):

```python
class StorageVolume:
    @endpoint
    async def init_bcast_group(master_addr, master_port, world_size,
                               rank, backend="hccl",
                               group_name="torchstore_bcast_reflector",
                               timeout_s=180): ...

    @endpoint
    async def bcast_tensor(key: str): ...

    @endpoint
    async def shutdown_bcast_group(): ...
```

Added to forge:

```python
class WorkerWrapper:
    @endpoint
    def init_bcast_group(master_addr, master_port, world_size, rank,
                         backend="hccl",
                         group_name="torchstore_bcast_reflector",
                         timeout_s=180): ...

    @endpoint
    def recv_and_load_flat(version, plan, total_bytes, src_rank=0): ...

    @endpoint
    def shutdown_bcast_group(): ...

class Generator:
    @endpoint
    async def get_worker_mesh(): ...  # exposes vllm_workers ActorMesh
```

#### 7.4.5 Control flow per weight-sync cycle

```
initialize  (once):
  trainer side: unchanged (MultiVol-style HiXL put path)
  backend:      spawn StorageVolumes, controller.init.call(...)
  backend:      rendezvous storage_vol[0] + all TP workers into one
                HCCL group world=1+TP; one probe all_reduce to eagerly
                build the HCCL communicator

push(v):    trainer_actor.publish_weights_flat(version=v, key=...)
            -> rank 0 packs full state_dict into aligned flat
            -> ts.put lands in storage vol 0 (HiXL)

pull(v):    asyncio.gather(
              storage_vol[0].bcast_tensor(key),           # HCCL bcast src
              vllm_workers.recv_and_load_flat(plan, ...)  # fanout recv + load
            )
```

#### 7.4.6 Debug trail (8 traps the diff captured)

Listed here so the next person touching this doesn't have to rediscover them:

1. **`ts.initialize` double-init.** After manually spawning StorageVolumes and calling
   `controller.init`, re-running `ts.initialize` raises "TorchStore is already
   initialized". `controller.init` alone is sufficient for `ts.put/ts.get` clients.
1. **Monarch `Extent` API.** No `.items()`; use `list(extent)` for labels and
   `extent[label]` for size.
1. **Worker mesh is 2D.** vLLM TP workers live on `{hosts: 1, procs: tp}` not a
   single-dim mesh; slice all non-`procs` dims to 0.
1. **`StorageVolume.get_id()` tuple order.** Returns `(volume_id, hostname)`, not
   `(hostname, volume_id)`. Read `vol_info[1]` for the hostname the HCCL TCPStore should
   rendezvous on.
1. **`LOCAL_RANK` doesn't reach workers as a device hint.** Every vLLM worker's
   `torch.npu.current_device()` is 0 until an explicit
   `torch.npu.set_device(LOCAL_RANK)` is called. Without this, all TP workers report
   "(host, device 0)" to HCCL's topology ranktable and init fails with "rank num\[K\] !=
   rank list size\[M\]".
1. **`dist.get_rank(pg)` on torch_npu.** The wrapper hits a `_get_default_group()` check
   even with a custom PG handle. Volumes always take rank 0 by contract here, so
   hard-code `src=0` in the bcast call and in the return payload.
1. **PrefixStore scoping.** `init_custom_process_group` installs a
   `PrefixStore(group_name, store)`; the plain `dist.init_process_group` does not.
   Storage side MUST use the same `init_custom_process_group` + matching `group_name` as
   the worker side, otherwise storage rank 0 writes `hcclUniqueId` under a different
   prefix than workers read from.
1. **TP-shard vs full-tensor shape mismatch.** vLLM's TP=K rank holds params sliced to
   `1/K` along the TP axis. Direct `param.data.copy_(full_view)` blows up with
   "\[shard\] must match \[full\]". Gate direct-copy on
   `tuple(target.shape) == tuple(shape)`; mismatched entries get passed to
   `model.load_weights`, which knows how to slice.

#### 7.4.7 Measured

End-to-end 2026-04-22 with Qwen3-0.6B, `vllm:d1p1t4+d4p1t1`,
`FORGE_WEIGHT_SYNC_BACKEND=collective_broadcast`, 3-step ping-pong:

| step | Weight sync | notes                                  |
| ---: | ----------: | -------------------------------------- |
|    0 |       2.0 s | warmup (HCCL group build, first bcast) |
|    1 |       0.5 s | steady                                 |
|    2 |       0.4 s | steady                                 |

No `103901`, no TP>1 stall, no pull-side connection explosion. TP=4 generator sustains
the same steady-state sync budget as TP=1 did on the multi-vol backend.

#### 7.4.8 MVP limitations and follow-ups

Tracked for the next iteration (not shipped here):

- **No shard publish.** `FORGE_SHARD_PUBLISH=1` is rejected when this backend is
  selected — the bcast source needs the full flat in one place. Multi-vol bcast (one
  HCCL group per storage vol, workers join all N groups and concatenate) is the natural
  extension but adds `N`× group lifecycle and re-sharding complexity.
- **Single bcast source.** Always vol 0 (configurable via `FORGE_BCAST_SRC_VOL_IDX`).
  For very large models the single-NIC egress from vol 0 may become the bottleneck; see
  shard publish point above.
- **Inference side uses `load_weights` for TP-sliced params.** The direct-copy fast path
  still applies to params that match the full shape (embedding tables on the TP axis,
  norm weights, etc.). For TP-sliced params we pay a Python-level dispatch into vLLM's
  `load_weights`. No evidence that this is a bottleneck at today's model sizes.
- **Group persistence across versions.** Built once in `initialize` and reused. If a TP
  worker crashes and respawns the group becomes stale; graceful teardown + re-init via
  `shutdown_bcast_group` is implemented but not yet wired into the replica-supervisor
  path.

#### 7.4.9 Workaround still valid: keep TP=1

If you don't need generator TP>1, the default `MultiVolTorchstoreBackend` stays the
shipped path and keeps the measured 0.4 s / 32 GB/s (agg across N vols) numbers. This
backend is an opt-in for TP>1 topologies.

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
3b7196a5 fix(forge): per-rank pool device + rewritten shard-publish diagnosis
757a47b4 docs(forge): add rank taxonomy section to weight_sync.md
42347917 chore(forge): confirm fix (a) alone doesn't unlock shard publish
79e33b63 fix(forge): unlock trainer-side rankTable via MONARCH_NPU_DEVICE
7c110415 docs(forge): identify Bug B root cause -- bootstrap silently skipped
eca54172 feat(forge): unlock 4x NIC shard publish via EnvSetter actor
514235c5 docs(forge): mark Bug B solved, Bug C no longer blocking
d6c19ec3 refactor(forge): drive mesh topology entirely from allocation_mode
a809c1a9 feat(forge): wire gen_tp/gen_pp fallback + document pull-side TP>1 gap
c18756cb feat(forge): MVP Step 1 for storage-as-reflector -- cross-mesh HCCL bcast
9e2fa38d feat(forge): MVP Step 2 for storage-as-reflector -- 1-to-N HCCL bcast
70b6a862 feat(forge): MVP Step 3 for storage-as-reflector -- HiXL + HCCL bcast coexist
e0752fd2 feat(forge): CollectiveBroadcastBackend -- TP>1 pull via storage-as-reflector
```

Reading order (pairs with the section layout of this doc):

1. **Performance ladder** (14 s → 0.4 s): `aef73317 → 1b53d107 → b18c72ef`.
1. **Topology abstraction** (storage spawn out of backend → YAML driving → dedicated-PS
   variant → shard-parallel scaffolding):
   `031e326a → 2abe4313 → 317e5cc5 → 28caa080 → cdbed36a → 1b767b66`.
1. **Shard-publish unlock** (§7.1 Bug A' + Bug B):
   `3b7196a5 → 757a47b4 → 42347917 → 79e33b63 → 7c110415 → eca54172 → 514235c5`.
1. **Allocation-mode-driven topology** (§7.2–7.3): `d6c19ec3 → a809c1a9`.
1. **CollectiveBroadcastBackend** (§7.4, this latest arc): three MVP smokes
   `c18756cb → 9e2fa38d → 70b6a862` then the backend itself `e0752fd2`.
