#!/usr/bin/env python3
"""Cross-host torchstore put/get test over Monarch + HiXL RoCE.

Validates that a trainer actor on host A can push / pull NPU tensors
through a torchstore StorageVolume hosted on host B, entirely over
HiXL RoCE (no Gloo / TCP fallback for the bulk data path).

Prerequisites
-------------
1. ``forge/scripts/worker_manager.sh start`` has been run so that the
   two monarch workers are listening on tcp://<HOST>:22222, and each
   worker process was started with:

     MONARCH_HIXL_TRANSPORT=roce
     HCCL_INTRA_ROCE_ENABLE=1
     HCCL_CONNECT_TIMEOUT=120
     TORCHSTORE_RDMA_ENABLED=1          (default, unset by worker_manager.sh)
     TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0  (keep tensors on NPU)

2. /etc/hosts has the hostnames of both workers resolvable on both
   sides (``worker_manager.sh hosts-setup`` does this).

Usage
-----
    python /root/AReaL/forge/scripts/test_torchstore_2node.py \
        --workers tcp://192.168.0.26:22222,tcp://192.168.0.23:22222 \
        --size-mb 64

Exit code 0 = everything round-tripped correctly.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import time

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _npu_bootstrap(dev_id: int):
    """Pin a worker proc to one NPU and make sure HiXL RoCE is on.

    These env vars are already exported by worker_manager.sh before
    python starts, but we re-set them in the spawned proc in case the
    test is run against workers launched differently.
    """

    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ["MONARCH_NPU_DEVICE"] = "0"
        os.environ["MONARCH_HIXL_TRANSPORT"] = "roce"
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        os.environ["TORCHSTORE_MONARCH_RDMA_EAGER_D2H"] = "0"
        # LocalRankStrategy requires LOCAL_RANK/RANK; single storage volume, so
        # rank 0 is fine. If this test ever spawns more procs per host, we need
        # per-proc rank assignment here.
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


async def _run(workers: list[str], size_mb: int, iters: int) -> int:
    import torch
    import torch_npu  # noqa: F401
    import torchstore as ts
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers
    from monarch.actor import Actor, endpoint
    from torchstore.strategy import LocalRankStrategy
    from torchstore.transport import TransportType
    from torchstore.transport.monarch_rdma import monarch_rdma_transport_available

    configure(default_transport=ChannelTransport.TcpWithHostname)

    if len(workers) != 2:
        print(f"ERROR: need exactly 2 workers, got {len(workers)}", flush=True)
        return 2

    size_bytes = size_mb * 1024 * 1024
    assert size_bytes % (2 * 1024 * 1024) == 0, "size must be 2 MB aligned"

    print(f"[driver] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized
    print(f"[driver] attached to {hosts.size()} hosts", flush=True)

    host_trainer = hosts.slice(hosts=0)  # host A: client / producer
    host_storage = hosts.slice(hosts=1)  # host B: torchstore volume

    trainer_mesh = host_trainer.spawn_procs(
        per_host={"npus": 1}, bootstrap=_npu_bootstrap(0)
    )
    storage_mesh = host_storage.spawn_procs(
        per_host={"npus": 1}, bootstrap=_npu_bootstrap(0)
    )

    class TrainerActor(Actor):
        """Actor on host A that drives ts.put / ts.get."""

        def __init__(self) -> None:
            pass

        @endpoint
        async def whoami(self) -> dict:
            return {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "hixl_transport": os.environ.get("MONARCH_HIXL_TRANSPORT", ""),
                "eager_d2h": os.environ.get(
                    "TORCHSTORE_MONARCH_RDMA_EAGER_D2H", "(default=1)"
                ),
                "rdma_enabled": os.environ.get(
                    "TORCHSTORE_RDMA_ENABLED", "(default=1)"
                ),
                "rdma_backend_available": monarch_rdma_transport_available(),
            }

        @endpoint
        async def init_store(self, storage_mesh_handle) -> None:
            await ts.initialize(
                num_storage_volumes=1,
                strategy=LocalRankStrategy(TransportType.MonarchRDMA),
                mesh=storage_mesh_handle,
            )

        @endpoint
        async def put_tensor(self, key: str, shape: tuple, fill: float) -> float:
            """Allocate NPU tensor, put via RDMA, return its sum."""
            from monarch._src.rdma.xdma import alloc_aligned_tensor

            t, _raw = alloc_aligned_tensor(shape, dtype=torch.float32, device="npu:0")
            t.fill_(fill)
            torch.npu.synchronize()
            await ts.put(key, t)
            return float(t.sum().cpu().item())

        @endpoint
        async def get_into(self, key: str, shape: tuple) -> dict:
            """Allocate NPU tensor, pull via RDMA, return checksum + bandwidth."""
            from monarch._src.rdma.xdma import alloc_aligned_tensor

            dst, _raw = alloc_aligned_tensor(shape, dtype=torch.float32, device="npu:0")
            dst.zero_()
            torch.npu.synchronize()

            bytes_xferred = dst.numel() * dst.element_size()

            t0 = time.time()
            got = await ts.get(key, dst)
            torch.npu.synchronize()
            elapsed = time.time() - t0

            return {
                "sum": float(got.sum().cpu().item()),
                "elapsed_s": elapsed,
                "bytes": bytes_xferred,
                "bw_gbps": bytes_xferred / elapsed / (1024**3),
                "device": str(got.device),
            }

        @endpoint
        async def pull_bandwidth(self, key: str, shape: tuple, iters: int) -> dict:
            """Repeatedly ts.get to measure steady-state bandwidth."""
            from monarch._src.rdma.xdma import alloc_aligned_tensor

            dst, _raw = alloc_aligned_tensor(shape, dtype=torch.float32, device="npu:0")
            dst.zero_()
            torch.npu.synchronize()
            bytes_xferred = dst.numel() * dst.element_size()

            # warmup
            await ts.get(key, dst)
            torch.npu.synchronize()

            t0 = time.time()
            for _ in range(iters):
                await ts.get(key, dst)
            torch.npu.synchronize()
            elapsed = (time.time() - t0) / iters

            return {
                "avg_s": elapsed,
                "bytes": bytes_xferred,
                "bw_gbps": bytes_xferred / elapsed / (1024**3),
            }

        @endpoint
        async def shutdown_store(self) -> None:
            await ts.shutdown()

    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    print("[driver] spawned trainer on host A", flush=True)

    info = await trainer.whoami.call_one()
    print(f"[driver] trainer = {info}", flush=True)
    if not info["rdma_backend_available"]:
        print(
            "[driver] FATAL: monarch RDMA transport not available on trainer",
            flush=True,
        )
        return 3

    # Storage volume will be spawned by ts.initialize on storage_mesh.
    await trainer.init_store.call_one(storage_mesh)
    print("[driver] ts.initialize done (storage on host B)", flush=True)

    n = size_bytes // 4
    shape = (n,)
    key = "rl/weight/layer0.weight"
    fill = 3.1415

    put_sum = await trainer.put_tensor.call_one(key, shape, fill)
    print(f"[driver] put  sum={put_sum:.3f} (expected {fill * n:.3f})", flush=True)

    result = await trainer.get_into.call_one(key, shape)
    print(
        f"[driver] get  sum={result['sum']:.3f} bw={result['bw_gbps']:.2f} GB/s "
        f"elapsed={result['elapsed_s'] * 1e3:.2f} ms on {result['device']}",
        flush=True,
    )

    expected_sum = fill * n
    # float32 loses precision when summing hundreds of millions of identical values;
    # compare at the put-side sum, which also round-trips through fp32 accumulation.
    if abs(result["sum"] - put_sum) > 1e-3 * abs(put_sum):
        print(
            f"[driver] FAIL: sum mismatch, got {result['sum']} "
            f"(put observed {put_sum}, ideal {expected_sum})",
            flush=True,
        )
        await trainer.shutdown_store.call_one()
        return 1

    if iters > 1:
        bw = await trainer.pull_bandwidth.call_one(key, shape, iters)
        print(
            f"[driver] steady-state {iters}x get: avg={bw['avg_s'] * 1e3:.2f} ms "
            f"bw={bw['bw_gbps']:.2f} GB/s ({size_mb} MB)",
            flush=True,
        )

    await trainer.shutdown_store.call_one()
    print("[driver] OK", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workers",
        required=True,
        help="Comma-separated monarch worker addresses "
        "(e.g. tcp://192.168.0.26:22222,tcp://192.168.0.23:22222)",
    )
    parser.add_argument("--size-mb", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    return asyncio.run(_run(workers, args.size_mb, args.iters))


if __name__ == "__main__":
    sys.exit(main())
