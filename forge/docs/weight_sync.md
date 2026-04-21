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

| #   | symptom                                                                     | root cause                                                                                                                                        | mitigation                                                                                                 | commit     |
| --- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------- |
| 1   | 14 s weight sync, 0.10 GB/s, tensors on CPU                                 | state_dict → `.cpu()` → pickle → RPC → state_dict on inference side                                                                               | Workers ts.get(inplace=param.data); no Python state_dict over the wire                                     | `aef73317` |
| 2   | 0.48 GB/s even after inplace                                                | 311 serial ts.get calls — torchstore `supports_batch_gets=False` defeats `ts.get_batch`                                                           | `asyncio.gather(*[ts.get(...)])`                                                                           | `1b53d107` |
| 3   | 1.1 GB/s still capped                                                       | pool staging: 2 NPU-NPU copies per ts.get                                                                                                         | flat-buffer with `alloc_aligned_tensor` + 1 key; pool staging skipped because flat is already pool-aligned | `b18c72ef` |
| 4   | small-tail param `hixl_transfer_write ret=503900`                           | vLLM's param.data is not 2 MiB aligned                                                                                                            | single flat buffer, not per-param                                                                          | `b18c72ef` |
| 5   | NPU 0 OOM at weight sync step                                               | pool_mb=8192 sat on the edge of NPU 0's free space after rollout                                                                                  | default pool_mb=6144 in YAML                                                                               | `28caa080` |
| 6   | YAML pool_mb=6144 ignored                                                   | `run_multinode.sh` eagerly exported the env var with its own default, winning precedence                                                          | conditional export: only forward if caller actually set it                                                 | `317e5cc5` |
| 7   | reward > 0 but model not updating (!!)                                      | worker pull_weights_flat returned success=False, backend treated num_keys as success                                                              | raise on empty plan; any-worker-fail → num_keys=0                                                          | `28caa080` |
| 8   | OOM on generator NPU 0, 613 MiB free                                        | YAML `generator.host_idx: 1` landed generator on the trainer host (NPU 0 already busy)                                                            | flip host_idx to match workers-array indexing (hostfile order)                                             | `28caa080` |
| 9   | Same physical host got multiple GpuManagers, NPUs overlap                   | launcher returned a fresh slice object for each new mesh name, Provisioner stamped unique `_host_id`                                              | `BareMetalLauncher._slice_by_worker_idx` cache                                                             | `317e5cc5` |
| 10  | `worker_manager.sh: MonarchRDMATransportBuffer.allocate: command not found` | backticks in rst/md-style comment command-substituted inside an unquoted heredoc                                                                  | strip backticks from shell-embedded comments                                                               | `2abe4313` |
| 11  | shard-publish: `hccl_comm.cc:132 errNo 0x0000000005000007`                  | torchstore MonarchRDMATransportBuffer bakes `(client NPU 0, server NPU base)` into HCCL rankTable at buffer creation; ranks 1..N-1 fail HCCL init | gated behind `FORGE_SHARD_PUBLISH=1`, default off; needs torchstore patch                                  | `1b767b66` |

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

### 7.1 Patch torchstore's MonarchRDMATransportBuffer rankTable

**Problem.** `MonarchRDMATransportBuffer._pre_put_hook` builds the HCCL rankTable for
`HcclCommInitClusterInfoMemConfig` from the module-level constants
`TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE` (client-side) and `storage_npu_base`
(server-side). All trainer ranks therefore write a rankTable with
`{client NPU: 0, server NPU: storage_npu_base}` — fine when only rank 0 puts, broken
when ranks 1..N-1 also put because HCCL rejects the mismatched device id.

**Shape of the fix.** Let the caller supply its actual
`(client_device_id, server_device_id)` per put. This is visible to torchstore because
`ts.put` has access to the calling rank's `LOCAL_RANK` and the strategy-resolved target
volume's server rank. Plumb those two through to the `HCCL_INIT_ROOT_INFO_CONFIG` call.

**Blocks.** Unlocking shard-parallel publish (4x NIC utilisation on the put side +
per-rank peak-memory halving once B1.2 lands per- rank `DTensor.to_local()` packing).

**Effort estimate.** Medium. Needs reading `torchstore/transport/monarch_rdma.py`
end-to-end plus the Rust- binding rankTable construction in monarch's `_rdma` crate.

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
