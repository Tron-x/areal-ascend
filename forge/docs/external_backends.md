# External training backends: bare-metal PoC vs production deployment

**Audience**: anyone integrating a third-party training framework (LlamaFactory,
TorchTitan, …) as a forge backend, OR migrating an already-integrated one from
bare-metal PoC to a container/shared-FS production cluster.

> **Read this together with `deployment_modes.md`** — that doc draws the broader
> bare-metal-vs-cluster-manager boundary; this one focuses on what changes per-backend
> when you cross the line.

## The "external repo" integration pattern

LlamaFactory and TorchTitan are **not** vendored into this repo. They live in their own
checkouts on every host (`/root/LlamaFactory`, `/root/torchtitan`), and forge plugs into
them via:

```
+-------------------------------------------------------+
| forge committed code (must be deployment-agnostic):   |
|                                                       |
|   forge/actors/llamafactory_trainer.py                |
|   forge/actors/titan_trainer.py                       |
|   forge/apps/llamafactory_train.py                    |
|   forge/apps/titan_pretrain.py                        |
|   examples/sft/qwen3vl_4b_lf_*.yaml         (algo YAMLs)
|   examples/pretrain/llama3_titan_debug.yaml          |
|                                                       |
|   ^- These files NEVER reference host-specific cache  |
|      paths, conda paths, model snapshots, etc.        |
+-------------------------------------------------------+
            |                            |
            v                            v
+--------------------------+   +--------------------------+
| External repo on disk    |   | Model / dataset cache    |
| (per-host or shared FS): |   | on disk (per-host or NFS):
|                          |   |                          |
|   /root/LlamaFactory     |   |   ~/.cache/modelscope    |
|   /root/torchtitan       |   |   /shared/models/...     |
|                          |   |                          |
| Pinned via the algo      |   | Pinned via lf_config /   |
| YAML's ``cwd`` field --  |   | the upstream repo's      |
| same convention TT uses. |   | training config.         |
+--------------------------+   +--------------------------+
```

The committed YAMLs (`examples/sft/qwen3vl_4b_lf_*.yaml`) *do* hold absolute paths — but
only the same kind of paths `examples/pretrain/llama3_titan_debug.yaml` already pins
(`config: /root/torchtitan/...`, `cwd: /root/torchtitan`). They follow the project-wide
convention `/root/<external_repo>` and move to e.g. `/workspace/<external_repo>` in a
container build by *editing the YAML once*, not by changing forge code.

What the committed YAMLs explicitly do **not** pin:

- Model snapshot paths (`model_name_or_path: Qwen/Qwen3-VL-4B-Instruct` — a logical hub
  ID that resolves via ModelScope/HF cache, not a bare-metal cache path).
- Conda / venv paths.
- DNS / proxy / mirror endpoints.
- Per-host network config.

If you find yourself wanting to commit any of those, push it down to either the external
repo's training config (LF YAML, TT TOML) or to a per-site overlay file the operator
owns.

## Bare-metal PoC bootstrap (today, 2-node NPU pool)

**Goal**: prep a fresh second host (e.g. `192.168.0.23`) so it can participate in a
`forge launch ... llamafactory-train` run alongside the launcher host.

### What lives where

| Artefact                                | Origin host               | Target                                                               | Why                                                                                                                                                                                                                                                                                                 |
| --------------------------------------- | ------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `forge/`, `areal/`                      | launcher (`192.168.0.26`) | every worker, FUSE-mounted at `/root/AReaL_remote/`                  | Source-of-truth pull from the launcher; no per-host drift. Configured by `_SSHJobFleet`.                                                                                                                                                                                                            |
| `examples/`                             | launcher                  | **driver host only** (per `role: driver` in the preset; today `.23`) | The algo-author YAML and any `lf_config` / `accelerate_config` it references are read **once on the driver** and shipped inline (as parsed dicts) to every actor. Workers never open them, so a worker host that lacks `examples/` still trains correctly. (Verified by smoke 2026-04-28 on `.26`.) |
| `LlamaFactory/`, model cache, conda env | each worker host          | each worker host                                                     | NOT mounted by forge today (mount list is `forge/`+`areal/` only). Must be present on every host before `forge launch`.                                                                                                                                                                             |

### One-shot bootstrap (operator runs once per new host)

The four tar-streams below are what we currently do to bring `.23` online. They are
**not committed as a script** — they are infrastructure prep, not application code, and
the production replacement (image bake + NFS) makes them obsolete (see next section).

> **Note on `examples/`** — `forge.apps.llamafactory_train` and the
> `LlamaFactoryTrainerActor.run()` endpoint exchange the LF/accelerate configs as parsed
> `dict` objects, not paths. The driver reads them once on its local FS and ships them
> inline through Monarch. Worker hosts therefore do **not** need `examples/` on disk;
> only the host tagged `role: driver` (in this preset, `.23`) does. The step below
> covers the driver-host-only case.

```bash
# From the launcher host (.26).  Replace HOST/PORT for your target.
HOST=192.168.0.23 ; PORT=36000
SP=/root/miniconda3/envs/monarch_ascend/lib/python3.11/site-packages

# 1. LlamaFactory checkout (+ NPU Conv3D loader patch).
cd /root && tar -czf - --exclude='.git' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='*.egg-info' LlamaFactory \
  | ssh -p $PORT root@$HOST 'cd /root && tar -xzf -'

# 2. trl wheel from the launcher's site-packages
#    (.23 has no PyPI access in our cluster).
cd $SP && tar -cz trl trl-*.dist-info \
  | ssh -p $PORT root@$HOST "cd $SP && tar -xz"

# 3. Add LF source to the worker's PYTHONPATH via a .pth shim
#    (avoids needing a working pip install on .23).
ssh -p $PORT root@$HOST \
    "echo /root/LlamaFactory/src > $SP/llamafactory.pth"

# 4. (DRIVER HOST ONLY -- skip on worker-only hosts.)  The algo
#    YAML + the LF/accelerate YAMLs it references must be readable
#    on whichever host the cluster preset tags ``role: driver``.
#    Other workers don't need ``examples/`` because the driver
#    parses these YAMLs and ships dicts to every actor.
cd /root/AReaL && tar -czf - examples/sft examples/accelerate \
  | ssh -p $PORT root@$HOST 'mkdir -p /root/AReaL && cd /root/AReaL && tar -xzf -'

# 5. Model snapshot (8.3GB Qwen3-VL-4B; required because .23's DNS
#    can't resolve modelscope.cn so first-run download fails).
cd /root/.cache/modelscope/hub/models/Qwen \
  && tar -cf - Qwen3-VL-4B-Instruct \
  | ssh -p $PORT root@$HOST \
        'mkdir -p /root/.cache/modelscope/hub/models/Qwen && cd /root/.cache/modelscope/hub/models/Qwen && tar -xf -'
```

After that, `forge launch examples/sft/qwen3vl_4b_lf_npu_2node.yaml --steps 3` should
green on 2x4 NPUs.

### What's deliberately NOT in the committed YAML

Two things came up during PoC that you might be tempted to bake into
`examples/sft/qwen3vl_4b_lf_npu_2node.yaml` but **must not be**:

1. **Model snapshot path** like `/root/.cache/modelscope/hub/models/...` pinned to
   `model_name_or_path`. Reason: hard-coding a host-side cache path makes the YAML
   un-runnable on any other layout (container image, NFS mount, k8s PVC). Instead, leave
   `lf_config.model_name_or_path` as a logical hub ID and rely on per-host cache layout.
1. **Per-host DNS / proxy workarounds** like `HF_HUB_OFFLINE=1` or custom
   `MODELSCOPE_DOMAIN`. Reason: same — cluster network policy is per-environment, not
   per-algorithm. Set those in the cluster preset (`forge/configs/clusters/*.yaml`)
   profile env block, or in the operator's shell, not in `examples/`.

If you genuinely need to override these per-run (debugging an offline host, A/B-ing two
snapshots, …), do it via:

- a local *unversioned* copy of the algo YAML, OR
- the upstream training config (`lf_config.model_name_or_path`), which IS the right
  place for "which model" decisions.

## Production migration: container + shared FS

This is the path from `deployment_modes.md` rendered for external backends specifically.
None of the bootstrap above survives — every step has a cleaner equivalent:

| Bare-metal step                  | Container/k8s replacement                                                                                                                                                                                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tar `LlamaFactory/` to each host | `pip install llamafactory==<pinned>` (or `git clone` + editable install) baked into the training image at build time.                                                                                                                                          |
| Tar `trl` site-package           | Listed in `requirements.txt` baked into the image.                                                                                                                                                                                                             |
| Add a `.pth` shim for LF source  | None. The pinned wheel is the source of truth.                                                                                                                                                                                                                 |
| Tar `examples/` to driver host   | Repo lives at `/workspace/AReaL` mounted from a PVC (read-only) OR baked into the image. The YAML in `examples/` is the same file, just at `/workspace/AReaL/examples/...`. (Workers don't need it either way — see actor↔driver dict-passing contract above.) |
| Tar 8GB model snapshot           | NFS / Lustre / S3-FUSE mount at `/shared/models/...`, OR `initContainer` that pulls from an internal registry. The mount path replaces ModelScope cache.                                                                                                       |
| Conv3D NPU loader patch          | Either upstream LF fixes it, or we ship a vendored fork (versioned, in our internal registry) — not a runtime tarball.                                                                                                                                         |

What stays the same:

- Every `forge/actors/`, `forge/apps/`, `forge/configs/clusters/*` file.
- The algo YAML *content* — only its absolute paths get rewritten by the operator
  (`/root/LlamaFactory` → `/workspace/LlamaFactory`, etc.). One sed pass per
  environment.
- `LlamaFactoryTrainerActor.run()` signature, the rendezvous logic, the FSDP2 env
  mapping, `use_modelscope` passthrough — all of it.

Concretely, the migration delta on the forge side is:

```diff
- launcher_impl: ssh_job
+ launcher_impl: kubernetes_job        # Monarch KubernetesJob

- pool:
-   - host: 192.168.0.26
-   - host: 192.168.0.23
+ pod_template: forge-trainer.yaml      # k8s PodSpec referencing
+                                       # the prebaked image and PVCs
```

i.e. a cluster-preset replacement, not an algorithm-side rewrite. That's the contract
the bare-metal PoC is upholding: *we paid the tarball-bootstrap tax once on .23 to
validate the design; we did not let any of it leak into the committed code*.

## Adding a new external backend (post-LF)

If you're plugging in a third backend (verl, OpenRLHF, …), follow the LF/TT pattern:

1. Add `forge/actors/<x>_trainer.py` — Monarch SPMDActor wrapping the upstream
   `run_exp()` / `main()`. No host paths.
1. Add `forge/apps/<x>_train.py` — the multi-host driver. Mirror
   `llamafactory_train.py`. No host paths.
1. Wire a new mode in `forge/cli/launch.py::_VALID_MODES` and a matching
   `_run_driver_<x>(...)` helper. Stays generic.
1. Add `examples/.../qwen3vl_4b_<x>_npu.yaml` algo YAML referencing the upstream
   training config by absolute path (same convention TT/LF use). Per-site knobs go in
   the upstream config, NOT the algo YAML.
1. Document the bare-metal bootstrap (which extra repos / wheels / model snapshots each
   host needs) **here**, not in `examples/`.

The litmus test for whether you've done it right: **a container-image-only operator
should be able to run your new mode by editing only the algo YAML's absolute paths and
the cluster preset's pod template** — never by patching `forge/actors/`, `forge/apps/`,
or `forge/cli/`.
