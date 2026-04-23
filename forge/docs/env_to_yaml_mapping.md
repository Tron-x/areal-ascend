# Env Var → YAML Mapping (Draft for Unified-Entry)

**Status**: draft for review. Purpose: inventory every `FORGE_*` / `TORCHSTORE_*` /
`MONARCH_*` environment variable forge reads today, then propose the corresponding YAML
field so users only ever need to know one (YAML) instead of three (YAML / env /
defaults).

**Design rule after refactor**:

> **YAML is authoritative. Env is override-only. Defaults are the last resort.** Every
> user-visible env var in this table must have a YAML field of equal precedence — if the
> YAML field is set, the env var is ignored; if the env var is set but YAML is empty,
> the env wins over the default; otherwise default applies.

This explicit resolution order replaces the current ad-hoc mix (some fields prefer env,
some prefer YAML, some read from both and merge).

______________________________________________________________________

## 1. Inventory by category

### 1a. Topology / parallelism (single source of truth: `allocation_mode`)

These ALREADY have YAML fields today after the alloc-mode refactor (`d6c19ec3`). Env
vars only remain as overrides.

| Env var          | Consumer                           | Proposed YAML field                            | Notes                                                                   |
| ---------------- | ---------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------- |
| `FORGE_GEN_TP`   | `grpo.py::ParallelLayout.gen_tp`   | derived from `allocation_mode` (`gen.tp_size`) | Already wired. Env is override-only today. Keep as override.            |
| `FORGE_GEN_PP`   | `grpo.py::ParallelLayout.gen_pp`   | derived from `allocation_mode` (`gen.pp_size`) | Same as above.                                                          |
| `FORGE_PS_WORLD` | `grpo.py::ParallelLayout.ps_world` | `launcher.weight_sync.ps_world`                | **NEW field.** Currently only env. Used by future dedicated-PS backend. |

### 1b. Weight-sync control (backend selection & tuning)

| Env var                                  | Consumer                                         | Proposed YAML field                                                  | Notes                                                                               |
| ---------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `FORGE_WEIGHT_SYNC`                      | `grpo.py::_create_weight_sync` (method dispatch) | `weight_sync.method` (enum: `nccl`/`checkpoint`/`hixl`/`torchstore`) | Currently env-only (default `nccl`). Promote to YAML.                               |
| `FORGE_WEIGHT_SYNC_BACKEND`              | `grpo.py::_create_weight_sync_service`           | `launcher.weight_sync.backend`                                       | **Partially wired today** (YAML → env fallback). Flip to YAML-authoritative.        |
| `FORGE_SHARD_PUBLISH`                    | `trainer.py::publish_weights_flat`               | `weight_sync.shard_publish` (bool)                                   | **Currently env-only.** Promote to YAML; backends that don't support it can assert. |
| `FORGE_BCAST_SRC_VOL_IDX`                | `collective_broadcast.py`                        | `launcher.weight_sync.bcast.src_vol_idx`                             | **Currently env-only.** New backend-specific knob; put under nested block.          |
| `FORGE_BCAST_MASTER_PORT`                | `collective_broadcast.py`                        | `launcher.weight_sync.bcast.master_port`                             | Same.                                                                               |
| `TORCHSTORE_MONARCH_RDMA_POOL_MB`        | `grpo.py`, `_env_setter.py`                      | `launcher.weight_sync.pool_mb`                                       | **Partially wired** (YAML → env fallback). Flip.                                    |
| `TORCHSTORE_MONARCH_RDMA_EAGER_D2H`      | `trainer.py`, `_env_setter.py`                   | `launcher.weight_sync.eager_d2h` (bool)                              | **Currently env-only.** Promote.                                                    |
| `TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE` | `trainer.py`, `worker.py`, `_env_setter.py`      | internal; do NOT expose in YAML                                      | Per-proc computed value; don't let user override from YAML (too error-prone).       |
| `TORCHSTORE_STORAGE_NPU_BASE`            | `grpo.py` & backends                             | `launcher.weight_sync.storage_npu_base`                              | **Partially wired.** Flip.                                                          |

### 1c. Mesh placement

| Env var                          | Consumer                       | Proposed YAML field                                                  | Notes                                                                                                                                                  |
| -------------------------------- | ------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `FORGE_MESH_PLACEMENT`           | `grpo.py::main` (CLI alt)      | `launcher.meshes.<name>.host_idx`                                    | **Already preferred from YAML today** (mesh_placement CLI / env fallback still exists). Keep env for one-offs but document CLI as the proper override. |
| `FORGE_STORAGE_HOST_MESH`        | `grpo.py`                      | `launcher.weight_sync.storage_mesh` (points to a `meshes.<name>`)    | **Partially wired** (YAML → env fallback). Flip.                                                                                                       |
| `FORGE_STORAGE_SPAWN_IN_BACKEND` | `grpo.py` (legacy path toggle) | `launcher.weight_sync.storage_spawn_mode` (enum: `driver`/`backend`) | **Currently env-only.** Legacy migration flag; promote for clarity, default `driver`.                                                                  |

### 1d. Monarch / Ascend runtime (set by bootstrap / EnvSetter)

These are set **inside** actor bootstrap by forge itself, not meant for user tuning.
Audit is for completeness only — **do not add YAML fields** unless specifically needed
to override per-cluster defaults.

| Env var                      | Where set                                         | User-facing?                                                                                               |
| ---------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `MONARCH_NPU_DEVICE`         | `trainer.py`, `_env_setter.py`, bootstrap scripts | **Internal** — set per-proc to correct NPU id, never user config.                                          |
| `MONARCH_HIXL_TRANSPORT`     | bootstrap scripts, `_env_setter.py`               | **Advanced** — always `roce` on our RoCE fabric. Could expose in YAML for other fabrics, but skip for MVP. |
| `ASCEND_RT_VISIBLE_DEVICES`  | `_env_setter.py`, bootstrap scripts               | **Internal** — per-proc NPU masking.                                                                       |
| `HCCL_INTRA_ROCE_ENABLE`     | bootstrap scripts                                 | **Advanced** — RoCE behavior knob. Skip for MVP.                                                           |
| `HCCL_CONNECT_TIMEOUT`       | bootstrap scripts                                 | **Advanced** — timeout knob.                                                                               |
| `HCCL_NPU_SOCKET_PORT_RANGE` | bootstrap scripts                                 | **Advanced** — port-range knob to avoid 16666 HiXL collision.                                              |

### 1e. Launcher / entrypoint flow

These are consumed by `run_multinode.sh` to wire arguments into the driver invocation.
After unified-entry lands, these disappear from the user's view entirely (the
`forge launch` CLI sets them internally).

| Env var        | Role                        | After refactor                                       |
| -------------- | --------------------------- | ---------------------------------------------------- |
| `FORGE_CONFIG` | Which launcher YAML to load | Becomes a positional CLI arg: `forge launch <path>`. |

______________________________________________________________________

## 2. Proposed unified YAML schema

After the refactor, every env in §1 above maps to a position under:

```yaml
# ~/my-grpo-config.yaml
launcher:
  # Section 1c
  type: bare_metal                  # or slurm / k8s / local
  bare_metal:
    workers:
      - tcp://192.168.0.26:22222
      - tcp://192.168.0.23:22222
    master_addr: 192.168.0.26       # auto-detected if empty
    worker_port: 22222

  meshes:
    trainer:   { host_idx: 1 }
    generator: { host_idx: 0 }
    storage:   { host_idx: 1 }

  # Section 1b
  weight_sync:
    method: torchstore              # Section 1b: nccl | checkpoint | hixl | torchstore
    backend: collective_broadcast   # Section 1b: which backend to use when method=torchstore
    shard_publish: false            # Section 1b: 4-NIC parallel trainer put

    # Storage placement (consumed by torchstore_multi_vol / collective_broadcast)
    storage_mesh: storage           # -> meshes.storage
    storage_npu_base: null          # null = auto (train_world_size)
    storage_spawn_mode: driver      # driver | backend

    # Transport tuning
    pool_mb: 8192                   # MonarchRDMA staging pool per vol
    eager_d2h: false                # TORCHSTORE_MONARCH_RDMA_EAGER_D2H

    # Backend-specific: collective_broadcast
    bcast:
      src_vol_idx: 0
      master_port: null             # null = auto-free

    # Backend-specific: dedicated_ps (future)
    ps_world: 0                     # Section 1a

# Standard forge / areal training config unchanged:
allocation_mode: vllm:d1p1t4+d4p1t1
# ... (rest of gsm8k_grpo_npu.yaml)
```

______________________________________________________________________

## 3. Precedence rules (explicit)

For every field listed in §1a–§1c:

```
FINAL_VALUE = yaml_value if yaml_value is not None else (
              env_value if env_value is not None else (
              default_value))
```

**Rule of thumb**: users should never need to set env vars to run forge normally. Env
vars exist only for:

1. Temporary debugging: `FORGE_SHARD_PUBLISH=1 forge launch config.yaml`
1. Per-machine cluster config that can't live in a shared YAML (rare).
1. Compat during the transition window (will be removed after one release cycle).

______________________________________________________________________

## 4. Migration impact (what changes in code)

Files that need edits:

1. `forge/core/types.py` — extend `LauncherConfig` with the new nested fields under
   `weight_sync` (`bcast`, `shard_publish`, etc.).
1. `forge/apps/grpo.py` — replace every `os.environ.get("FORGE_X", ...)` with a single
   helper `_resolve(yaml_value, env_name, default)` and route everything through it.
   Central lookup function means future added fields don't need to repeat the precedence
   logic.
1. `forge/actors/trainer.py` — read `FORGE_SHARD_PUBLISH` via an env that `grpo.py`
   populates based on YAML (or expose it as trainer constructor arg, cleaner but more
   invasive).
1. `forge/engines/weight_sync/backends/*` — same: accept values from constructor, not
   env, for all user-visible knobs.
1. `forge/scripts/run_multinode.sh` — becomes a thin wrapper that forwards argv to
   `forge launch` internally; eventually deleted once `forge launch` is mature.
1. `forge/configs/launcher_bare_metal_2node*.yaml` — update example YAMLs to cover the
   full schema, with comments pointing to this doc.
1. New file: `forge/cli/__init__.py` and `forge/cli/launch.py` (Python CLI) OR
   `forge/scripts/launch.sh` (bash wrapper). Language decision at implementation time
   (my preference: Python for the pre-flight check quality, fall back to bash if
   dependency pressure hits).

**Estimated churn**: ~200 LOC, spread across 6-7 files. Manageable in 1-2 days.

______________________________________________________________________

## 5. Design decisions (resolved)

Four open questions on the draft; decided here so later sections and the implementation
can assume them.

**Q1. `FORGE_WEIGHT_SYNC=nccl` legacy path — retire or keep?**

Decision: **keep, mark legacy in YAML comments.** torchstore + HiXL is the preferred
path but hasn't soaked in production long enough. nccl is our fallback for environments
without HiXL. Retire in a later release (≥6 months) once torchstore is proven at scale.

**Q2. `weight_sync` nested under `launcher:` or promoted to top-level?**

Decision: **keep nested under `launcher:`.** Several fields under `weight_sync`
(`storage_mesh`, `storage_npu_base`) reference `launcher.meshes.<name>`, so they belong
together conceptually. Promotion would force a rewrite of every existing YAML and test
for no user-visible benefit. Revisit only when/if weight sync needs to be shared across
multiple launcher configs.

**Q3. CLI flags `--config` and `--forge-config` — merge?**

Decision: **merge into a single `--config`.** User mental model: one YAML, one command.
Merging two YAML files is trivial (they're just two top-level keys, `allocation_mode` +
`launcher`). Keep `--forge-config` as a deprecated alias — same parse path, emits a
warning, removed next major.

**Q4. Env-var backward compatibility window?**

Decision: **1-release soft deprecation.** Every env var we're replacing still works,
emits a single-line warning that names the YAML field users should migrate to, and is
removed in the next major release. Rationale: our own smoke/repro scripts use env vars
extensively; a hard cut breaks them. Warnings silenced by `FORGE_SUPPRESS_DEPRECATION=1`
for automation.

______________________________________________________________________

## 6. Next steps (after your sign-off on this mapping)

1. Adjust schema based on your feedback (§5 answers).
1. Extend `LauncherConfig` with the new fields.
1. Add the `_resolve()` helper and route ~15 call sites through it.
1. Update two example YAMLs to the new shape.
1. Build `forge launch` CLI.
1. Smoke TP=1 and TP=4 with new entry.
1. Update `quickstart.md`.
