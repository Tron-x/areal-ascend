# areal.weight_sync — collective weight sync component

A standalone, framework-neutral weight-synchronisation component shared
across AReaL, Forge, and external trainers (ms-swift, TRL, verl, ...).

This package solves one specific problem: **broadcast trainer weights
into a vLLM rollout server every step, on multi-host NPU (and CUDA),
without depending on vllm-ascend's single-host `HcclCommInitRootInfo`
path**.  We do that by going through PyTorch's `torch.distributed` C10D
layer (`init_custom_process_group`), which dispatches to the right
multi-host HCCL APIs on Ascend and to NCCL on CUDA.

It is a refactored extraction of code originally living under
`areal.engine.core.distributed` and `areal.engine.vllm_ext`.  The legacy
import paths still re-export the same symbols, so existing AReaL users
do not need to change anything.

## Layout

```
areal/weight_sync/
├── __init__.py             # public API
├── distributed.py          # init_custom_process_group / patch_dist_group_timeout
└── vllm_ext/
    ├── __init__.py
    ├── client.py           # WeightSyncClient (trainer-side, HTTP + dist.broadcast)
    ├── server_router.py    # FastAPI router exposing /areal_* endpoints
    └── worker_extension.py # vLLM `--worker_extension_cls` target
```

Three components, one matched protocol:

| Side               | Component                                            | Drives                                                   |
|--------------------|------------------------------------------------------|----------------------------------------------------------|
| vLLM workers       | `VLLMWorkerExtension`                                | `init_update_weight_group`, `update_weight_xccl`, ...    |
| vLLM API server    | `server_router.router` (FastAPI APIRouter)           | `/areal_init_weights_update_group`, `/areal_update_weights_xccl`, ... |
| Trainer rank 0     | `WeightSyncClient`                                   | `init_communicator()`, `push_weights_xccl()`, `close()`  |

The trainer side is *new* in this extraction (previously this logic was
spread across `areal.engine.fsdp_engine`, `areal.engine.vllm_remote`,
and `areal.infra.RolloutController`).  `WeightSyncClient` is a
self-contained reimplementation that uses only `requests` for HTTP and
raw `torch.distributed.broadcast` -- no AReaL control plane, no Monarch
actors required.

## Quick reference (AReaL itself)

AReaL's `FSDPEngine`/`MegatronEngine` keep using `init_custom_process_group`
internally; nothing changes for AReaL operators.  Forge users go
through the `areal_xccl` strategy in
`forge/engines/weight_sync/backends/areal_xccl.py`, which wires this
component into Forge's `WeightSyncBackend` Protocol.

## ms-swift integration recipe (NPU, multi-host)

### Why this exists

ms-swift ≥ `vllm_mode=server` ships with its own `vllm_client` whose
`init_communicator` path uses vllm-ascend's `PyHcclCommunicator`, which
internally calls the **single-host-only** `HcclCommInitRootInfo` API.
That works on a single box with `vllm_data_parallel_size=1`, but fails
with `HCCL error: resource unavailable` the moment trainer and rollout
live on different hosts -- which is the typical Forge / Monarch
deployment.

This component restores cross-host weight sync by replacing both ends
with `torch.distributed`-based HCCL/NCCL, exactly the path AReaL has
been validating in production.

### Step 1 — start the rollout server with the AReaL extension

`SwiftRolloutDeploy` (`swift rollout` CLI) launches vLLM under the
hood; we just need to (a) inject our `VLLMWorkerExtension` into every
vLLM worker and (b) mount our FastAPI router onto its app.

```python
# launch_rollout.py
from areal.weight_sync.vllm_ext import server_router as ws_router

# Make every worker expose init_update_weight_group / update_weight_xccl
# as collective_rpc methods.
import sys
sys.argv += [
    "--worker_extension_cls",
    "areal.weight_sync.vllm_ext.worker_extension.VLLMWorkerExtension",
]

# Mount /areal_* endpoints.
from swift.pipelines import rollout_main
from swift.pipelines.infer.rollout import SwiftRolloutDeploy

_orig_get_app = SwiftRolloutDeploy._get_app  # or whatever the hook is

def _patched_get_app(self, *args, **kwargs):
    app = _orig_get_app(self, *args, **kwargs)
    app.include_router(ws_router.router)
    return app

SwiftRolloutDeploy._get_app = _patched_get_app

if __name__ == "__main__":
    rollout_main()
```

Run:

```bash
HCCL_IF_IP=192.168.0.23 \
HCCL_SOCKET_IFNAME=enp67s0f5 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
python launch_rollout.py \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --vllm_tensor_parallel_size 4 \
    --host 0.0.0.0 --port 8000
```

### Step 2 — replace ms-swift's vLLM client on the trainer side

ms-swift's GRPO trainer (inherited from `trl.GRPOTrainer`) instantiates
a `VLLMClient` in `_init_vllm_client` and calls
`vllm_client.init_communicator()` once at startup, then
`vllm_client.update_named_param(name, tensor)` every step.  Replace
both with a thin adapter on top of `WeightSyncClient`:

```python
# patch_swift_client.py
import torch
from areal.weight_sync import WeightSyncClient

class _WSClientAdapter:
    """Drop-in for trl.extras.vllm_client.VLLMClient (only the methods
    ms-swift's GRPOTrainer actually calls)."""

    def __init__(self, server_url: str, *, group_name: str = "swift_grpo"):
        self._inner = WeightSyncClient(server_url, group_name=group_name)
        self._pending: list[tuple[str, torch.Tensor]] = []

    def init_communicator(self, master_addr, master_port, world_size, **_):
        # ms-swift's VLLMClient passes vLLM-side world_size already.
        self._inner.init_communicator(
            master_addr=master_addr,
            master_port=master_port,
            vllm_world_size=world_size,
        )

    def update_named_param(self, name: str, tensor: torch.Tensor):
        # Buffer; ms-swift calls this in a loop and we want to
        # amortise the HTTP round-trips.
        self._pending.append((name, tensor))

    def update_model_params(self):
        if not self._pending:
            return
        self._inner.push_weights_xccl(self._pending, chunk_mb=512)
        self._pending.clear()

    def close_communicator(self):
        self._inner.close()

# Patch ms-swift's lookup so its trainer picks up our adapter.
import trl.extras.vllm_client as _vc
_vc.VLLMClient = _WSClientAdapter
```

Run the trainer with this patch loaded before `swift rlhf`:

```bash
PYTHONPATH=$(pwd) \
HCCL_IF_IP=192.168.0.26 \
HCCL_SOCKET_IFNAME=enp67s0f5 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 -m \
    swift.cli.main rlhf \
    --vllm_mode server \
    --vllm_server_base_url http://192.168.0.23:8000 \
    --rlhf_type grpo \
    ...                # rest of the swift GRPO args
```

### Verifying

`init_communicator()` should log:

```
WeightSyncClient: init_communicator: server=http://... group=swift_grpo
                                     backend=hccl world=5
                                     init_method=tcp://192.168.0.26:<port>
WeightSyncClient: init_communicator complete: group=swift_grpo
```

The vLLM server should log (per worker):

```
vllm_worker_extension: weight update group `swift_grpo` joined
                       (rank=N/5, backend=hccl)
```

If you see `HCCL error: resource unavailable` again, double-check that
`server_router.router` is actually mounted (`curl http://server:8000/areal_pause_generation`
should return JSON, not 404) and that
`--worker_extension_cls=areal.weight_sync.vllm_ext.worker_extension.VLLMWorkerExtension`
made it into vLLM's argv.

## Backward compatibility shims

The following legacy paths re-export the symbols we moved.  They are
preserved so that `areal/`, `forge/`, third-party scripts, and vLLM's
`--worker-extension-cls=areal.engine.vllm_ext.vllm_worker_extension.VLLMWorkerExtension`
keep working unchanged:

| Old path                                                   | New canonical path                                |
|------------------------------------------------------------|---------------------------------------------------|
| `areal.engine.core.distributed`                            | `areal.weight_sync.distributed`                   |
| `areal.engine.vllm_ext.vllm_worker_extension`              | `areal.weight_sync.vllm_ext.worker_extension`     |
| `areal.engine.vllm_ext.areal_vllm_server`                  | `areal.weight_sync.vllm_ext.server_router`        |

Both `python -m areal.engine.vllm_ext.areal_vllm_server` and
`python -m areal.weight_sync.vllm_ext.server_router` start the same
forked vLLM server CLI.

## Roadmap

- [x] Phase 1 — extract collective primitives into `areal/weight_sync/`
- [x] Phase 2 — Forge `WeightSyncBackend` adapter (`areal_xccl`)
- [x] Phase 3 — standalone trainer-side `WeightSyncClient` for ms-swift / TRL
- [ ] Phase 4 — sibling `areal.weight_sync.torchstore` for one-sided
      (TorchStore-based) async weight sync; same package, separate
      subpackage so users can pick collective vs one-sided per
      deployment without touching the rest of their training code.
