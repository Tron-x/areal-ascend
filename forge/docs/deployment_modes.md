# Deployment modes: dev vs. prod, and where the "ease-of-use" layer lives

**Audience**: new contributors trying to figure out which of the half-dozen
flags in `forge launch` are "core" vs. "bare-metal dev scaffolding",
and anyone asking "do I still need `forge sync` / `launcher_impl=bash`
after we move to k8s?".

## TL;DR

- **Bare-metal dev** (today, 2-node NPU cluster): local editable
  checkouts on each host, SSH-based worker lifecycle, opt-in source
  sync via `forge sync`. This is the path where `--launcher-impl
  bash|ssh_job`, `hostfile.txt`, `forge sync`, and
  `worker_manager.sh` live.
- **Containerized prod** (future, k8s/slurm/etc.): a single immutable
  image is the source of truth; Monarch's `KubernetesJob` /
  `SlurmJob` owns worker lifecycle; no source sync. The bare-metal
  dev tooling is NOT the prod path -- it will be replaced wholesale
  when we adopt a cluster manager.
- Both paths coexist long-term. Bare-metal is not a temporary scaffold
  -- some Ascend / HiXL customer deployments will stay bare-metal for
  the foreseeable future, and dev iteration on containers often still
  wants `forge sync` semantics.

## The architecture boundary

```
+-----------------------------------------------------------+
| Algorithm layer                                           |
|   - workflows (RLVR, multi-turn, react, ...)              |
|   - rewards (gsm8k, geometry3k, ...)                      |
|   - trainer (PPO / GRPO, FSDP2 / Megatron / Archon)       |   <- deployment-agnostic
+-----------------------------------------------------------+
| Monarch scheduling layer                                  |
|   - Actor / HostMesh / ProcMesh / WeightUpdateMeta        |
|   - JobTrait (LoginJob, SSHJob, SlurmJob, KubernetesJob)  |   <- form-factor dispatch
+-----------------------------+-----------------------------+
| Bare-metal dev adapter      | Cluster-manager adapter     |
|   (this repo, today)        |   (future)                  |
|   - launcher_impl: bash     |   - Monarch SlurmJob        |
|   - launcher_impl: ssh_job  |   - Monarch KubernetesJob   |
|   - hostfile.txt            |   - k8s PodSpec / helm      |
|   - forge sync              |   - baked container image   |
|   - worker_manager.sh       |   - PVC / ConfigMap / NFS   |
|   - ForgeSSHJob override    |   - (no remote code push)   |
+-----------------------------+-----------------------------+
```

The line to remember: **algorithm code is deployment-agnostic; the
bottom row is not**. When you move from bare-metal to k8s you swap
the entire bottom-left block for the bottom-right. You do NOT keep
`forge sync` running alongside k8s and slowly refactor it -- it
disappears, replaced by the image pipeline.

## Per-scenario comparison

| Scenario | Who it's for | Code distribution | `forge sync`? | `launcher_impl`? |
|----------|--------------|-------------------|---------------|------------------|
| **Bare-metal dev** (current 2-node NPU) | research iteration | local editable checkout on every host | **YES** -- local edit must propagate | `ssh_job` (preferred) or `bash` (legacy) |
| **Bare-metal prod** (rare) | air-gapped HPC sites, small fleets | pre-installed `/opt/forge/` or NFS mount on every host | only for hotfixes (with `--remote-root`) | `ssh_job` |
| **k8s / slurm + image prod** (future, primary) | large-scale training jobs | `COPY forge/ /workspace` baked into image | **NO** -- image is source of truth | N/A (replaced by `KubernetesJob` / `SlurmJob`) |
| **k8s + editable install dev** | "attach to pod and iterate" | base image + `pip install -e /mnt/forge` on a PVC/hostPath | maybe -- via SSH to pod, or `kubectl cp` | N/A |
| **Shared filesystem** (NFS / Lustre / GPFS) | shared-storage clusters | one `/nfs/forge` mount point | **NO** -- single writer visible to all | `ssh_job` or `SlurmJob` |

Rule of thumb: ask "can two hosts see each other's local filesystem
writes?". Bare-metal = no = sync needed. Image or shared FS = yes =
sync meaningless.

## How to read today's bare-metal dev tooling

Everything under "Bare-metal dev adapter" exists to paper over the
fact that the cluster has no native way to broadcast code changes or
manage process lifecycle. Concretely:

- **`hostfile.txt`**: replaces what a SLURM `sbatch` / k8s `PodSpec`
  would encode (which machines, how many slots).
- **`launcher_impl=ssh_job` -> `ForgeSSHJob`**: replaces what
  `SlurmJob._create()` / `KubernetesJob._create()` will eventually
  provide (remote process spawn + lifecycle + cleanup).
- **`forge sync`**: replaces what a container registry +
  `imagePullPolicy: Always` would provide (code distribution).
- **`worker_manager.sh`**: the even-older `launcher_impl=bash` path,
  kept as a rollback channel until `ssh_job` is battle-tested enough
  to become the default. Will be deleted after one release cycle with
  no regressions reported.

Each of these can be thought of as a "POSIX shim for a missing
cluster manager". None of them are algorithm-side features.

## Migration story

1. **Today (bare-metal dev)**: keep current defaults. `launcher_impl`
   default is `bash` (we just landed the `ssh_job` opt-in); plan is to
   flip to `ssh_job` after one or two more real training rounds, then
   delete the bash fleet after a release.
2. **Next (bare-metal prod polish)**: no-op path-wise -- bare-metal
   prod uses the same tooling as bare-metal dev, minus `forge sync`
   (code comes from a versioned tarball or NFS snapshot, not from a
   researcher's laptop).
3. **Later (containerized dev)**: add optional `--in-pod` / `--pod-
   selector` support to `forge launch` that dispatches through
   Monarch's `KubernetesJob`. `forge sync` stays available but is
   usually NOT what you want -- prefer `kubectl cp` for ad-hoc code
   pushes, or `imagePullPolicy: Always` for clean rebuilds.
4. **Steady state (containerized prod)**: `forge launch` reads the
   launcher from YAML; `LauncherConfig.launcher` is
   `kubernetes | slurm | bare_metal`. Bare-metal-only fields
   (`hostfile`, `launcher_impl`, `cann` preamble) become no-ops or
   validation errors when `launcher != bare_metal`. `forge sync` is
   documented as dev-only.

## FAQ

**Q: If `forge sync` is dev-only, why ship it in the main CLI instead
of a `dev/` helper?**
A: Dev ergonomics is part of the framework's contract. "New user
runs two nodes" is a first-class supported scenario, and making it
ten minutes faster is worth a 500-line module. The alternative is
everyone reinventing `rsync` one-liners that drift over time (which
is the bug this module fixes). It's explicitly opt-in via `--sync`
so prod pipelines that don't want it simply don't pass the flag.

**Q: When do I hit "image is the source of truth"?**
A: The moment you `docker push registry/forge:v1.2.3` and the cluster
pulls it. Any code inside a running container after that point is
either (a) the image's baked-in version, or (b) a manual mutation
that gets wiped on restart. In either case, running `forge sync` is
at best redundant, at worst a silent drift source (your sync would
overwrite the image and the next pod restart would revert your edit).
Don't mix modes.

**Q: We're on k8s but I still want editable iteration.**
A: Two good patterns and one bad pattern:
- *Good*: mount a PVC at `/workspace/forge` with a pre-committed
  checkout, `pip install -e .` at entrypoint. Editors push via
  `kubectl cp` or a sidecar sync. No image rebuild.
- *Good*: dev pod runs `pip install git+https://...@branch` on every
  start. Slower but reproducible.
- *Bad*: `kubectl exec -it pod -- vim`. Tempting, doesn't survive a
  restart, silently drifts from what others see.

**Q: How does this interact with `WeightSyncBlock` / `torchstore`?**
A: Completely separately. `WeightSyncBlock` covers *model weight*
transfer between trainer and generator engines (that's a data path,
network-bound, happens every training step). `forge sync` covers
*source code* distribution to hosts (that's a dev path, disk-bound,
happens at most once per edit). They share "distribute something
across nodes" as a concept but have nothing in common
implementation-wise and should not be conflated in reviews.

## Related docs

| File | Focus |
|------|-------|
| `forge/docs/launcher_impl.md` | `launcher_impl: bash vs ssh_job` switch + `forge sync` companion; details the bare-metal dev adapter's two sub-paths |
| `forge/docs/architecture_current.md` | What's wired today (components, message flow) |
| `forge/docs/weight_sync.md` | `WeightSyncBlock` + `torchstore` backend; data-path doc, not to be confused with source sync |
| `forge/docs/role_abstraction_design.md` | Phase R1/R2 schema for placing roles on hosts; relevant to both dev and prod paths |
