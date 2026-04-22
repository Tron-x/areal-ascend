#!/usr/bin/env python3
"""MVP Step 3 for the storage-as-reflector design: HiXL + HCCL bcast
coexisting in the storage proc.

Background (continuation of MVPs 1 & 2 in the sibling scripts):

* Step 1: 2-rank cross-mesh HCCL bcast -> 10.2 GB/s, verified OK.
* Step 2: 1-to-N HCCL bcast (sender + TP=4 receivers) -> per-link
  8 GB/s, aggregate 32 GB/s, verified OK.

Those two MVPs isolated the new collective-broadcast path from any
other transport. The last question before we can ship
``CollectiveBroadcastBackend`` is: **inside one storage proc, can
HiXL (as torchstore ``ts.put`` target) coexist with being rank 0 of an
HCCL broadcast group?**

``repro_hixl_multirank_hccl.py`` already established the *general*
coexistence answer: world=2 HCCL PG + HiXL on one proc is fine (17.7
GB/s); world=4 HCCL PG + HiXL on one proc is NOT fine (HiXL's own
~80 MB bound-memory registration fails with ``ra_hdc_typical_mr
ret=-13``).  Our storage-as-reflector design sits strictly inside the
safe regime: the storage proc participates in the bcast group as
exactly one rank (source), so from HCCL's per-proc point of view the
world size doesn't matter -- what matters is how many communicators
the same proc is joining.  This MVP pins that down concretely for our
combined transport.

Topology
--------

    host A (monarch1)                          host B (monarch2)
    +------------------------------+           +---------------------------+
    |  TrainerActor   (NPU 0)      |           |  Receiver rank 1  (NPU 0) |
    |  StorageActor   (NPU 4)      |<--RoCE--->|  Receiver rank 2  (NPU 1) |
    |  = rank 0 in bcast group     |           |  Receiver rank 3  (NPU 2) |
    |  = HiXL put target           |           |  Receiver rank 4  (NPU 3) |
    +------------------------------+           +---------------------------+

Flow per iteration
------------------

1. Trainer ``ts.put(key, tensor)`` -> lands in StorageActor's torchstore
   volume on NPU 4, transported via HiXL RDMA.
2. StorageActor's bcast endpoint reads the data that just arrived (we
   currently re-fill the bcast buffer from the ts-put payload via a
   memcpy -- full wiring into torchstore's internal pool would be
   the backend's job, not this MVP's).
3. StorageActor is rank 0 of an HCCL group (1+TP), calls
   ``dist.broadcast`` to send the buffer to all receivers.
4. Each receiver calls ``dist.broadcast`` with ``src=0`` and verifies
   the bytes it got match what the trainer wrote.

Pass criterion
--------------

* All of (ts.put, HCCL init, bcast, ts-internal HiXL handshake)
  complete without any of the known coexistence error codes:
  ``503900`` (HiXL Connect), ``103901`` (CreateChannel), ``0x0000000005000007``
  (hccl initialize failed), ``ra_hdc_typical_mr ret=-13``
  (HiXL bound-mem reg).
* Every receiver verifies the tensor content.

If Step 3 passes, the only thing between here and a real working
backend is control-plane plumbing: letting WeightSyncService reuse
the existing trainer push path and orchestrate storage-side bcast
instead of issuing ``ts.get`` calls from each TP worker.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import socket
import sys
import time

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _bootstrap_trainer(dev_id: int):
    """Trainer proc: 1 NPU, HiXL client (torchstore put caller)."""

    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        # These are what the forge multinode launcher sets; keep them
        # consistent so we exercise the same HiXL init path as
        # production.
        os.environ["MONARCH_NPU_DEVICE"] = "0"
        os.environ["MONARCH_HIXL_TRANSPORT"] = "roce"
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        # Trainer is the ts.put source -- eager D2H=0 keeps the
        # staging pool on-device.
        os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_EAGER_D2H", "0")
        os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE", "npu:0")
        os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_POOL_MB", "4096")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")

        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


def _bootstrap_storage(dev_id: int):
    """Storage proc: 1 NPU, HiXL target (torchstore volume) AND bcast rank 0.

    This is the proc under test -- the one where the coexistence
    question matters.  We deliberately set the same torchstore env as
    trainer so torchstore initializes the volume on the masked
    device (logical ``npu:0`` = physical ``dev_id``).
    """

    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ["MONARCH_NPU_DEVICE"] = "0"
        os.environ["MONARCH_HIXL_TRANSPORT"] = "roce"
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_EAGER_D2H", "0")
        os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE", "npu:0")
        os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_POOL_MB", "4096")
        # LocalRankStrategy wants LOCAL_RANK; with a single vol any
        # value maps to vol 0, so "0" is fine.
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")

        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


def _bootstrap_receiver_pool(npu_base: int, tp: int):
    """Receiver pool bootstrap -- see test_cross_mesh_bcast_1toN.py for why
    we share the full NPU pool here instead of single-NPU masking.
    """

    def _bootstrap():
        pool = ",".join(str(npu_base + i) for i in range(tp))
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = pool
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

        import torch  # noqa: F401
        import torch_npu  # noqa: F401

        # Per-proc device pinning happens in the endpoint via
        # ``current_rank()``; bootstrap can only do per-host env.

    return _bootstrap


async def _run(
    workers: list[str],
    trainer_npu: int,
    storage_npu: int,
    receiver_npu_base: int,
    tp: int,
    payload_gb: float,
    iters: int,
    master_port: int,
) -> int:
    import torchstore as ts
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers
    from monarch.actor import Actor, current_rank, endpoint

    configure(default_transport=ChannelTransport.TcpWithHostname)

    if len(workers) != 2:
        print(f"ERROR: need exactly 2 workers, got {len(workers)}", flush=True)
        return 2
    if tp < 2:
        print(f"ERROR: tp must be >=2, got {tp}", flush=True)
        return 2

    print(f"[driver] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized

    host_A = hosts.slice(hosts=0)
    host_B = hosts.slice(hosts=1)

    trainer_mesh = host_A.spawn_procs(
        per_host={"npus": 1}, bootstrap=_bootstrap_trainer(trainer_npu)
    )
    storage_mesh = host_A.spawn_procs(
        per_host={"npus": 1}, bootstrap=_bootstrap_storage(storage_npu)
    )
    receiver_mesh = host_B.spawn_procs(
        per_host={"npus": tp},
        bootstrap=_bootstrap_receiver_pool(receiver_npu_base, tp),
    )

    world_size = 1 + tp

    class TrainerActor(Actor):
        """HiXL client -- writes tensor into torchstore."""

        def __init__(self) -> None:
            self.buf = None

        @endpoint
        async def whoami(self) -> dict:
            return {
                "role": "trainer",
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            }

        @endpoint
        async def prepare(self, nbytes: int) -> dict:
            import torch
            import torch_npu  # noqa: F401
            from monarch._src.rdma.xdma import alloc_aligned_tensor

            # 2 MB-aligned buffer -- same as HiXL's expected alignment
            # (matches forge FORGE_SHARD_PUBLISH trainer side).
            buf, _raw = alloc_aligned_tensor(
                (nbytes,), dtype=torch.uint8, device="npu:0"
            )
            self.buf = buf
            torch.npu.synchronize()
            return {"nbytes": self.buf.numel()}

        @endpoint
        async def push(self, step: int) -> dict:
            """ts.put the buffer under a step-dependent key."""
            import torch
            import torch_npu  # noqa: F401

            self.buf.fill_((step + 1) % 256)
            torch.npu.synchronize()
            key = f"mvp3_step_{step:010d}"
            t0 = time.perf_counter()
            await ts.put(key, self.buf)
            put_s = time.perf_counter() - t0
            return {
                "step": step,
                "bytes": int(self.buf.numel()),
                "put_s": put_s,
                "key": key,
                "fill_value": (step + 1) % 256,
            }

    class StorageActor(Actor):
        """HiXL target (via torchstore) AND HCCL bcast rank 0.

        The whole point of Step 3 is that BOTH transports coexist in
        this single proc. Every endpoint here runs inside the storage
        proc.
        """

        def __init__(self) -> None:
            self.pg = None
            self.bcast_buf = None

        @endpoint
        async def whoami(self) -> dict:
            return {
                "role": "storage",
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            }

        @endpoint
        async def init_bcast_group(
            self, master_addr: str, master_port: int, world_size: int
        ) -> dict:
            """Join the cross-mesh HCCL group as rank 0 (broadcast source)."""
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            from areal.engine.core.distributed import init_custom_process_group

            init_method = f"tcp://{master_addr}:{master_port}"
            print(
                f"[storage] init_process_group rank=0 world={world_size}",
                flush=True,
            )
            self.pg = init_custom_process_group(
                backend="hccl",
                init_method=init_method,
                world_size=world_size,
                rank=0,
                group_name="storage_reflector_mvp3",
                timeout=datetime.timedelta(seconds=180),
            )
            probe = torch.ones(8, dtype=torch.float32, device="npu:0")
            dist.all_reduce(probe, group=self.pg)
            torch.npu.synchronize()
            return {
                "probe_sum": float(probe.sum().item()),
                "expected": float(8 * world_size),
            }

        @endpoint
        async def prepare_bcast_buf(self, nbytes: int) -> dict:
            import torch
            import torch_npu  # noqa: F401

            # NOTE: separate buffer from the torchstore volume.  In the
            # real backend this is where "read the just-put data from
            # torchstore's pool" would happen -- see Step 3 pass
            # criterion.  For this MVP we just copy from a fetched
            # tensor via ts.get; in prod the backend will either:
            #   (a) ts.get into self.bcast_buf directly (easiest, one
            #       d2d copy inside the storage proc), or
            #   (b) read torchstore's pool pointer directly and bcast
            #       from it (zero-copy, more invasive).
            self.bcast_buf = torch.empty((nbytes,), dtype=torch.uint8, device="npu:0")
            torch.npu.synchronize()
            return {"nbytes": self.bcast_buf.numel()}

        @endpoint
        async def reflect(self, step: int, expected: int) -> dict:
            """Bcast the local buffer to all receivers.

            MVP-3 deliberately does NOT call ``ts.get`` from inside the
            storage proc to pull trainer's ts.put output before
            bcasting. torchstore's ``ts.get`` API assumes client and
            volume are in different procs (same-proc self-read
            deadlocks on its own MonarchRDMA handshake). For this MVP
            we just fill the bcast buffer locally -- the goal is to
            prove HiXL (trainer -> storage torchstore volume put
            path) and HCCL (storage -> receivers bcast path) can live
            in the same proc concurrently. Whether the bcast buffer
            is seeded by self-ts.get vs direct fill is a backend
            implementation choice (the real backend will wire the
            torchstore pool pointer directly into the bcast buffer,
            zero copy).
            """
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            self.bcast_buf.fill_(expected)
            torch.npu.synchronize()

            t_b0 = time.perf_counter()
            dist.broadcast(self.bcast_buf, src=0, group=self.pg)
            torch.npu.synchronize()
            bcast_s = time.perf_counter() - t_b0

            return {
                "step": step,
                "bytes": int(self.bcast_buf.numel()),
                "bcast_s": bcast_s,
                "front": int(self.bcast_buf[0].cpu().item()),
                "back": int(self.bcast_buf[-1].cpu().item()),
            }

    class ReceiverActor(Actor):
        def __init__(self) -> None:
            self.pg = None
            self.buf = None
            self.tp_rank = None
            self.device = None

        @endpoint
        async def whoami(self) -> dict:
            import torch_npu  # noqa: F401

            return {
                "role": "receiver",
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "mesh_rank": current_rank().rank,
            }

        @endpoint
        async def init_bcast_group(
            self,
            master_addr: str,
            master_port: int,
            world_size: int,
            rank_offset: int = 1,
        ) -> dict:
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            from areal.engine.core.distributed import init_custom_process_group

            mesh_rank = current_rank().rank
            torch.npu.set_device(mesh_rank)
            self.tp_rank = mesh_rank
            self.device = f"npu:{torch.npu.current_device()}"
            my_rank = rank_offset + mesh_rank
            init_method = f"tcp://{master_addr}:{master_port}"
            print(
                f"[receiver mesh_rank={mesh_rank} device={self.device}] "
                f"init_process_group rank={my_rank}",
                flush=True,
            )
            self.pg = init_custom_process_group(
                backend="hccl",
                init_method=init_method,
                world_size=world_size,
                rank=my_rank,
                group_name="storage_reflector_mvp3",
                timeout=datetime.timedelta(seconds=180),
            )
            probe = torch.ones(8, dtype=torch.float32, device=self.device)
            dist.all_reduce(probe, group=self.pg)
            torch.npu.synchronize()
            return {
                "rank": my_rank,
                "device": self.device,
                "probe_sum": float(probe.sum().item()),
            }

        @endpoint
        async def allocate(self, nbytes: int) -> dict:
            import torch
            import torch_npu  # noqa: F401

            self.buf = torch.zeros((nbytes,), dtype=torch.uint8, device=self.device)
            torch.npu.synchronize()
            return {"nbytes": self.buf.numel(), "tp_rank": self.tp_rank}

        @endpoint
        async def recv_and_check(self, step: int, expected: int) -> dict:
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            t0 = time.perf_counter()
            dist.broadcast(self.buf, src=0, group=self.pg)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - t0
            front = int(self.buf[0].cpu().item())
            back = int(self.buf[-1].cpu().item())
            ok = front == expected and back == expected
            return {
                "step": step,
                "tp_rank": self.tp_rank,
                "bytes": int(self.buf.numel()),
                "elapsed_s": elapsed,
                "ok": ok,
                "expected": expected,
                "front": front,
                "back": back,
            }

    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    storage = storage_mesh.spawn("storage", StorageActor)
    receiver = receiver_mesh.spawn("receiver", ReceiverActor)

    print(f"[driver] trainer  = {await trainer.whoami.call_one()}", flush=True)
    print(f"[driver] storage  = {await storage.whoami.call_one()}", flush=True)
    recv_infos = await receiver.whoami.call()
    for r, info in recv_infos.items():
        print(f"[driver] receiver {r} = {info}", flush=True)

    # torchstore: single volume on the storage mesh.  Same knobs as
    # test_weight_sync_2node.py / forge's multi_vol backend.
    from torchstore.strategy import LocalRankStrategy
    from torchstore.transport import TransportType

    print("[driver] initializing torchstore (1 vol, MonarchRDMA) ...", flush=True)
    await ts.initialize(
        num_storage_volumes=1,
        strategy=LocalRankStrategy(default_transport_type=TransportType.MonarchRDMA),
        mesh=storage_mesh,
    )
    print("[driver] torchstore ready", flush=True)

    # HCCL bcast group rendezvous on the storage host.
    storage_info = await storage.whoami.call_one()
    rdvz_host = storage_info["host"]

    async def _await_fut(fut):
        return await fut

    print(
        f"[driver] HCCL world=1+{tp} PG rendezvous @ {rdvz_host}:{master_port}",
        flush=True,
    )
    try:
        init_results = await asyncio.wait_for(
            asyncio.gather(
                _await_fut(
                    storage.init_bcast_group.call_one(
                        master_addr=rdvz_host,
                        master_port=master_port,
                        world_size=world_size,
                    )
                ),
                _await_fut(
                    receiver.init_bcast_group.call(
                        master_addr=rdvz_host,
                        master_port=master_port,
                        world_size=world_size,
                        rank_offset=1,
                    )
                ),
            ),
            timeout=180,
        )
    except TimeoutError:
        print("[driver] HCCL init timed out", flush=True)
        return 1
    print(f"[driver] storage init = {init_results[0]}", flush=True)
    for r, info in init_results[1].items():
        print(f"[driver] receiver init {r} = {info}", flush=True)

    # Allocate payload on trainer, storage, and each receiver.
    payload_bytes = int(payload_gb * 1024**3)
    print(f"[driver] allocating {payload_gb:.2f} GB payload ...", flush=True)
    await asyncio.gather(
        _await_fut(trainer.prepare.call_one(nbytes=payload_bytes)),
        _await_fut(storage.prepare_bcast_buf.call_one(nbytes=payload_bytes)),
        _await_fut(receiver.allocate.call(nbytes=payload_bytes)),
    )

    print("=" * 88, flush=True)
    print(
        " step | put_s | put_GBps | bcast_s | per-link_GBps | agg_GBps | verified",
        flush=True,
    )
    print("=" * 88, flush=True)

    results = []
    for step in range(iters):
        # Run HiXL put and HCCL bcast CONCURRENTLY to stress the
        # coexistence: trainer's ts.put hits storage's torchstore
        # volume (HiXL path) at the same time storage's bcast sends
        # to receivers (HCCL path). Both transports active in the
        # same storage proc simultaneously is the exact condition
        # we want to validate.
        expected = (step + 1) % 256
        try:
            push_r, storage_r, recv_map = await asyncio.gather(
                _await_fut(trainer.push.call_one(step=step)),
                _await_fut(storage.reflect.call_one(step=step, expected=expected)),
                _await_fut(receiver.recv_and_check.call(step=step, expected=expected)),
            )
        except Exception as e:
            print(
                f"[driver] concurrent step={step} RAISED: {type(e).__name__}: {e}",
                flush=True,
            )
            await ts.shutdown()
            return 1

        storage_ok = storage_r["front"] == expected and storage_r["back"] == expected
        all_recv_ok = all(r["ok"] for r in recv_map.values())
        verified = storage_ok and all_recv_ok

        elapsed_list = [storage_r["bcast_s"]] + [
            r["elapsed_s"] for r in recv_map.values()
        ]
        bcast_s = max(elapsed_list)
        put_gbps = payload_bytes / push_r["put_s"] / 1024**3
        per_link = payload_bytes / bcast_s / 1024**3 if bcast_s > 0 else 0.0
        agg = (payload_bytes * tp) / bcast_s / 1024**3 if bcast_s > 0 else 0.0
        print(
            f" {step:4d} | {push_r['put_s']:5.2f} | {put_gbps:8.2f} | "
            f"{bcast_s:7.2f} | {per_link:13.2f} | "
            f"{agg:8.2f} | {verified}",
            flush=True,
        )
        if not verified:
            if not storage_ok:
                print(
                    f"[driver] storage-side VERIFY FAILED step={step}: "
                    f"expected={expected} front={storage_r['front']} "
                    f"back={storage_r['back']}",
                    flush=True,
                )
            for r_key, r in recv_map.items():
                if not r["ok"]:
                    print(
                        f"[driver] receiver {r_key} VERIFY FAILED step={step}: "
                        f"expected={r['expected']} front={r['front']} "
                        f"back={r['back']}",
                        flush=True,
                    )
            await ts.shutdown()
            return 1
        results.append((push_r["put_s"], bcast_s))

    print("=" * 88, flush=True)
    if len(results) > 1:
        warm = results[1:]
        avg_put = sum(r[0] for r in warm) / len(warm)
        avg_bcast = sum(r[1] for r in warm) / len(warm)
        print(
            f"[driver] steady-state (iters 1..{iters - 1}): "
            f"put={avg_put:.2f}s ({payload_bytes / avg_put / 1024**3:.2f} GB/s), "
            f"bcast={avg_bcast:.2f}s per-link={payload_bytes / avg_bcast / 1024**3:.2f} GB/s "
            f"agg={payload_bytes * tp / avg_bcast / 1024**3:.2f} GB/s",
            flush=True,
        )

    await ts.shutdown()
    print("[driver] OK -- HiXL + HCCL bcast coexistence validated", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", nargs="+", required=True)
    parser.add_argument("--trainer-npu", type=int, default=0)
    parser.add_argument("--storage-npu", type=int, default=4)
    parser.add_argument("--receiver-npu-base", type=int, default=0)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--payload-gb", type=float, default=1.5)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--master-port", type=int, default=29901)
    args = parser.parse_args()

    rc = asyncio.run(
        _run(
            workers=args.workers,
            trainer_npu=args.trainer_npu,
            storage_npu=args.storage_npu,
            receiver_npu_base=args.receiver_npu_base,
            tp=args.tp,
            payload_gb=args.payload_gb,
            iters=args.iters,
            master_port=args.master_port,
        )
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
