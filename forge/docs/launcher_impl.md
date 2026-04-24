# Bare-metal worker lifecycle: `launcher_impl` switch

**Audience**: anyone running multi-node `forge launch` on the 2-node
NPU cluster (or a future equivalent).

**Scope**: this doc covers the **bare-metal dev** adapter only --
`launcher_impl`, `ForgeSSHJob`, `worker_manager.sh`, and the companion
`forge sync` subcommand. For the bigger picture (why this adapter
exists, how it maps to a future `KubernetesJob` / `SlurmJob` path,
and when you should stop reading this doc and use an image instead),
see `forge/docs/deployment_modes.md`. Short version: **everything on
this page is bare-metal-dev-only scaffolding; containerized prod
doesn't use any of it**.

## TL;DR

```yaml
# forge/configs/clusters/2node_colocated.yaml
launcher:
  launcher: bare_metal
  launcher_impl: ssh_job   # or "bash" (default, legacy)
```

- `bash` (default): `forge launch` shells out to `forge/scripts/worker_manager.sh`
  to start/stop `run_worker_loop_forever` on every host. Battle-tested
  path; unchanged from the pre-refactor flow.
- `ssh_job`: `forge launch` uses Monarch's native `SSHJob` (via our
  `ForgeSSHJob` subclass) to manage the worker fleet. Same SSH
  commands, same remote binary, but lifecycle is Python-native so
  status/stop/restart wire up cleanly with Monarch's `JobTrait`.

CLI override: `forge launch --launcher-impl ssh_job cfg.yaml ...`
takes precedence over whatever is in YAML.

## Why a new path exists

Summary of the architecture debt `ssh_job` pays off:

1. **`worker_manager.sh` is a re-implementation of `SSHJob`**. Monarch
   already ships `LocalJob` / `SSHJob` / `SlurmJob` / `KubernetesJob`
   as siblings of `JobTrait`. Our bash script does exactly what
   `SSHJob` does (`ssh host "python -c run_worker_loop_forever(...)"`)
   but with 300 lines of shell and no PID tracking.
2. **Unified lifecycle API**. Switching to `SSHJob` means the
   `BareMetalLauncher` we ship today and the `SlurmJob` we'd need for
   a bigger cluster tomorrow share the same `apply()` / `_state()` /
   `_kill()` surface. Launcher swap becomes a one-line config change.
3. **Reliable PID tracking**. `SSHJob._host_to_pid` tracks the local
   ssh client PID per host; `forge status` / `forge stop` (future
   follow-ups) can read that dict directly instead of having to
   `pgrep` residuals out of remote hosts.
4. **Faster startup**. POC measured 4-12s per host for the ssh_job
   path vs the bash path's 60s TCP-probe ceiling, because we can
   start attaching as soon as `apply()` returns.

## Two concrete Monarch quirks we work around

Both are documented inline in `forge/provisioner_ssh.py` but worth
surfacing here because they're feedback candidates for the Monarch
upstream.

### 1. `_create()` does NOT transition the job to "running"

`SSHJob._create(client_script=None)` starts the remote workers BUT
leaves `self._status = "not_running"`. A subsequent `_state()` then
trips `if not self._pids_active(): raise RuntimeError("lost
connection")` because `_pids_active` short-circuits on
`self.active == False`.

**Fix**: always call the public `apply()` method, which wraps
`_create` + the status transition. `ForgeSSHJob` exposes only
`apply()` and `_kill()` to callers, not `_create()`.

### 2. `_kill()` orphans remote workers on bare-metal SSH

Vanilla `SSHJob._kill()` only SIGKILLs the local ssh client PIDs.
Because `_start_host` uses `ssh -n` (no PTY allocation), sshd on the
remote side sees the TCP connection drop but has no controlling
terminal to forward SIGHUP through. The remote `python -c
"run_worker_loop_forever(...)"` keeps running as an orphan.

**Fix**: `ForgeSSHJob._kill` calls `super()._kill()` first (cleans up
local ssh clients), then opens a best-effort `ssh host 'pgrep -f
"python.*[r]un_worker_loop_forever" | xargs kill -9'` against each
tracked host. The `[r]un_...` bracket trick is critical: without it
the pkill's own invoking shell matches the pattern and self-terminates
before reaping the actual worker.

## Migration plan

Inside the bare-metal dev adapter (the `bash` -> `ssh_job` transition):

1. **Today** (landed): `ssh_job` is opt-in via YAML or CLI. Default
   stays `bash` so CI / muscle memory keep working.
2. **After a few clean runs** with `ssh_job`: flip the default in
   `forge/core/types.py` to `launcher_impl = "ssh_job"`.
3. **After a release cycle**: delete `forge/scripts/worker_manager.sh`
   and the `_BashFleet` class; remove the switch entirely.

Step 3 should only happen when there's NO remaining user with
`launcher_impl: bash` in pinned configs. Keep the bash path in
read-only mode until then.

Outside the adapter (the bare-metal-dev -> containerized-prod
transition): this page stops applying entirely. When
`LauncherConfig.launcher` flips from `bare_metal` to `kubernetes` /
`slurm`, Monarch dispatches via `KubernetesJob` / `SlurmJob`, and
`launcher_impl`, `hostfile`, `worker_manager.sh`, `ForgeSSHJob`, and
`forge sync` all become no-ops or validation errors. See
`deployment_modes.md` for that migration.

## Related code

| File | Purpose |
|------|---------|
| `forge/provisioner_ssh.py` | `ForgeSSHJob` class -- subclass of `SSHJob` with CANN/HCCL/HiXL env preamble + remote-cleanup override |
| `forge/cli/launch.py` | `_WorkerFleet` / `_BashFleet` / `_SSHJobFleet` plus the CLI switch |
| `forge/core/types.py` | `LauncherConfig.launcher_impl` field |
| `forge/scripts/poc_ssh_job.py` | Standalone POC harness; runs all 3 invariants against a live cluster |
| `tests/test_forge_ssh_job.py` | 19 unit tests covering preamble, `_start_host`, `_kill`, YAML reader |

## Companion: `forge sync` (code sync to all hosts)

Bare-metal multi-node assumes the repository checkout under
`FORGE_ROOT` is byte-identical on every host. Before we had a first-
class sync command, two full `forge launch` reruns were lost to
driver-side `ConfigKeyError: Key 'reward' not in 'GRPOConfig'` --
literally because the driver's `areal/api/cli_args.py` lagged the
local checkout by one schema change.

### CLI

```bash
# Push default paths (forge/, areal/) to every host in the hostfile.
python -m forge sync --hostfile forge/configs/hostfile.txt --ssh-port 36000

# Explicit path set (defaults replaced, not extended).
python -m forge sync \
    --hostfile forge/configs/hostfile.txt \
    --ssh-port 36000 \
    --path forge --path areal --path examples

# Dry-run prints the tar | ssh argv per host without executing.
python -m forge sync --hostfile ... --dry-run
```

### `forge launch --sync`

The same helper is wired into `forge launch` as an opt-in step that
runs AFTER pre-flight checks and BEFORE worker startup:

```bash
python -m forge launch examples/math/gsm8k_grpo_npu.yaml \
    --launcher-impl ssh_job --sync
```

`--sync-path` implies `--sync` and replaces the default path set
(pass `--sync-path forge --sync-path areal --sync-path examples` to
keep the built-ins plus `examples/`).

### Why `tar | ssh` instead of `rsync`?

`rsync` is the obvious choice for delta transfer but requires the
binary on BOTH sides. Our NPU container images frequently ship
minimal and lack it (we hit this exact wall in the same session that
motivated this module). `tar | ssh` only needs `tar` + `ssh`, which
are guaranteed. The trade-off is that we ship the full path set every
call (no delta), which is acceptable -- relevant source is tens of
MB and sync is a launch prelude, not a hot loop.

### Safety rails

- Default path set is narrow (`forge`, `areal`). Everything else is
  opt-in via `--path`. No surprise `scp` of a 40 GB checkpoint dir.
- Default excludes (`__pycache__`, `*.pyc`, `.git`, `.venv`,
  `node_modules`, ...) filter out the usual junk.
- We NEVER delete remote-only files; the command is strictly
  additive / overwriting. If rsync `--delete` semantics are needed,
  open a follow-up.

### When NOT to use `forge sync`

`forge sync` makes sense precisely in the bare-metal dev scenario
where every host holds an independent local checkout. If any of the
following are true, you should not be running it:

- You're on k8s / SLURM with a baked container image. The image is
  the source of truth; `forge sync` would at best be redundant, at
  worst silently overwrite the image and drift back on the next pod
  restart.
- You're on a shared filesystem (NFS / Lustre / GPFS) where
  `/root/AReaL` is the same mount on every host. One writer, all
  readers see it; sync is meaningless.
- You want rsync's `--delete` semantics (removing stale remote
  files). `forge sync` is strictly additive by design.

See `forge/docs/deployment_modes.md` for the full per-scenario
breakdown.

### Related code

| File | Purpose |
|------|---------|
| `forge/cli/sync.py` | `sync_paths_to_hosts()` helper + `forge sync` CLI |
| `forge/cli/launch.py` | `--sync` / `--sync-path` flags (wires into `sync_paths_to_hosts`) |
| `tests/test_forge_sync.py` | 25 unit tests for argv composition, parallelism, error aggregation |
