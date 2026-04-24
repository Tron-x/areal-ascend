# Forge Quickstart (2-node NPU GRPO)

Fastest path from a clean checkout to a running 2-node GRPO training job on Huawei
Ascend NPU + HiXL RoCE. Shows **the authoritative entry point**
(`python -m forge launch`) — everything else in the repo is a backward-compat wrapper
around it.

## Prerequisites

- Two Ascend 910B nodes with CANN ≥ 9.0, each with ≥ 4 visible NPUs.
- RoCE 200 Gb/s between the nodes (one port per NPU card).
- SSH key-based login from the driver host to every worker (including itself). Default
  SSH port 36000; override with `--ssh-port`.
- Forge installed in the same conda env on every node (`monarch_ascend` by convention;
  see `forge/README.md`).
- `/root/AReaL` checked out to the same commit on every node.

## Step 1: Describe the cluster

Point the launcher at a hostfile listing your nodes:

```txt
# forge/configs/hostfile.txt
192.168.0.26 slots=8
192.168.0.23 slots=8
```

The **second non-empty line is the driver host** (where `forge.apps.grpo` runs). Workers
connect back to it.

## Step 2: Pick or customize a launcher YAML

AReaL uses a **two-layer YAML** split (see
`forge/cli/presets.py`): infra/ops owns a *cluster preset* under
`forge/configs/clusters/`, and the algo author sets `launcher_preset:`
+ per-role device counts in their experiment YAML.

Built-in presets:

* `forge/configs/clusters/2node_colocated.yaml` — 2-node NPU setup, storage on
  the trainer host.
* `forge/configs/clusters/2node_dedicated_ps.yaml` — 2-node NPU setup, storage
  on the generator host (A/B topology).

An experiment YAML references a preset with one line:

```yaml
# examples/math/gsm8k_grpo_npu.yaml  (excerpt)
launcher_preset: 2node_colocated

roles:
  trainer:   {devices: 4}
  generator: {devices: 4}
  storage:   {devices: 4}
  reward:    {devices: 0}
```

The preset file itself holds the infra details (pool, colocate, weight-sync
fabric):

```yaml
# forge/configs/clusters/2node_colocated.yaml  (excerpt)
launcher:
  type: bare_metal
  launcher_impl: ssh_job

  pool:
    - {host: 192.168.0.26, port: 22222, n_devices: 8}
    - {host: 192.168.0.23, port: 22222, n_devices: 8, role: driver}

  roles:
    trainer:   {devices: 4, hardware: npu}
    generator: {devices: 4, hardware: npu}
    storage:   {devices: 4, hardware: npu, colocate: trainer}
    reward:    {devices: 0, hardware: npu, colocate: trainer}

  weight_sync:
    method: torchstore            # use the WeightSyncService path
    backend: torchstore_multi_vol # TP=1 default; swap to collective_broadcast for TP>1
    storage_mesh: storage
    pool_mb: 6144
    # ... see the YAML itself for the full annotated schema
```

Every field has a sane default and an extensive comment in the YAML. See
`forge/docs/env_to_yaml_mapping.md` for the complete mapping from env vars to YAML
fields and the `yaml > env > default` precedence rule.

## Step 3: Launch

One Python command — no env vars needed:

```bash
python -m forge launch examples/math/gsm8k_grpo_npu.yaml \
    --steps 3 \
    --backend titan --model-name qwen3 --model-flavor 0.6B \
    --model /root/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \
    -- \
    allocation_mode=vllm:d1p1t1+d4p1t1 \
    gconfig.max_new_tokens=128  gconfig.n_samples=2 \
    rollout.consumer_batch_size=8  rollout.max_concurrent_rollouts=8
```

The positional arg is the **algorithm YAML** — `forge launch`
auto-detects the `launcher_preset:` key, resolves the preset, merges
the algo's role-device overrides, and writes a composed launcher YAML
to `/tmp/forge_composed_*.yaml` before starting workers.  Pass a raw
preset (`forge/configs/clusters/*.yaml`) for legacy / bring-up flows.

What happens:

1. **Pre-flight**: hostfile existence, per-host SSH reachability, YAML parseability.
   Fails fast with a concrete error list.
1. **Worker start**: `worker_manager.sh start` spawns Monarch workers on every host in
   the hostfile.
1. **Training**: SSHs to the driver host, activates the conda env, exports NPU/HiXL
   runtime env, runs `python -m forge.apps.grpo`.
1. **Cleanup (guaranteed)**: on success, failure, or Ctrl+C, all Monarch workers are
   stopped remotely. Use `--keep-workers` to disable cleanup during iterative
   development.

Expected output on success (Qwen3-0.6B, 3 steps, TP=1):

```
 forge launch: multi-node GRPO
   launcher config : /tmp/forge_composed_*.yaml
   algo config     : examples/math/gsm8k_grpo_npu.yaml
   driver host     : 192.168.0.23
   workers         : tcp://192.168.0.26:22222,tcp://192.168.0.23:22222
   steps           : 3
 ... (worker start, training, auto-cleanup) ...
 [Train-Engine] Weight sync: 3.1s   (warmup)
 [Train-Engine] Weight sync: 0.4s   (steady)
 [Train-Engine] Weight sync: 0.4s
 forge launch: training OK (401s)
```

The remote driver's full training log lives at `/tmp/forge_multinode.log` on the driver
host.

## Common variations

### TP > 1 on the generator

Flip two YAML fields (or CLI-override them):

```yaml
launcher:
  weight_sync:
    backend: collective_broadcast   # change backend
```

```bash
python -m forge launch ... -- allocation_mode=vllm:d1p1t4+d4p1t1
```

Pull path now goes through one HCCL broadcast (`1 + TP` ranks) instead of per-worker
`ts.get`. Measured steady-state 0.4s/sync at TP=4. See `forge/docs/weight_sync.md §7.4`
for the full design.

### 4-NIC parallel publish (trainer-side 4× put bandwidth)

```yaml
launcher:
  weight_sync:
    shard_publish: true
```

Incompatible with `backend: collective_broadcast` (collective needs the full flat tensor
on one vol). Aggregate throughput ~32 GB/s on our 2-node RoCE setup.

### Dedicated-PS topology (storage on the non-trainer host)

Flip the preset reference in your algo YAML:

```yaml
launcher_preset: 2node_dedicated_ps
```

Same role-device fields; the preset flips `storage.colocate` from
`trainer` to `generator` and sets `storage_npu_base: 4`.

### Backwards compat: existing shell scripts still work

```bash
# Legacy shell entry point is now a thin wrapper around `forge launch`.
# Existing CI / muscle-memory scripts keep working verbatim.
bash forge/scripts/run_multinode.sh --hostfile forge/configs/hostfile.txt \
    --steps 3 --backend titan --model-name qwen3 --model-flavor 0.6B \
    --model /path/to/model
```

### Env-var overrides (soft-deprecated, still honored)

Every user-visible field in `launcher.weight_sync.*` has a corresponding legacy env var.
Set the env to override the YAML for one-off experiments; you'll see a single-line
deprecation warning to stderr pointing at the canonical YAML field:

```bash
FORGE_SHARD_PUBLISH=1 python -m forge launch ...
# [forge deprecation] env var FORGE_SHARD_PUBLISH='1' is still honored
# as an override, but the canonical location is the YAML field
# `launcher.weight_sync.shard_publish`. Move the setting into YAML;
# env support will be removed in the next major release. Silence with
# FORGE_SUPPRESS_DEPRECATION=1.
```

Full env-to-YAML catalog: `forge/docs/env_to_yaml_mapping.md`.

## Troubleshooting

| Symptom                                     | Where to look                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `forge launch: pre-flight FAILED`           | CLI's own stderr — message tells you exactly which check broke                                               |
| `hixl_connect ... ret=103901` at TP>1       | Known; switch to `backend: collective_broadcast`. See weight_sync.md §7.4                                    |
| `Ranktable_Detect_Failed EI0015`            | HCCL topology mismatch.  See weight_sync.md §7.4.6 for the 8 debug traps                                     |
| Training OK but `Weight sync` lines missing | `method: torchstore` probably not set in YAML — verify it's there, otherwise legacy `nccl` path takes over   |
| Orphan workers after crash                  | `bash forge/scripts/worker_manager.sh stop --hostfile ...` (forge launch usually handles this automatically) |

For more, see:

- `forge/docs/weight_sync.md` — complete weight-sync architecture + known-pain ledger
  (21 commits' worth of lessons captured).
- `forge/docs/env_to_yaml_mapping.md` — every env var, its YAML field, and the design
  decisions locking the schema.
- `forge/README.md` — repo-level orientation.
