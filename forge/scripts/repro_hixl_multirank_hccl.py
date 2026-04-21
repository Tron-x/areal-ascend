#!/usr/bin/env python3
"""Minimum reproducer: HiXL ``ra_hdc_typical_mr ret(-13)`` when coexisting
with a multi-rank HCCL process group on the same host.

Findings that motivate this repro (captured from AReaL/forge smoke test
investigation, Apr 2026):

* HiXL alone, cross-node, RoCE, Qwen3 0.6B (1.5 GB flat buffer):
  **17 GB/s steady-state ✅**  -- ``test_weight_sync_2node.py`` with a
  single-proc trainer.
* HiXL + one cross-node HCCL PG (``world_size=2``, one rank per host)
  coexisting in the trainer proc:
  **17.7 GB/s steady-state ✅**  -- same test with ``TEST_HCCL_PREWARM=1``.
* HiXL + multi-rank HCCL PG (``world_size=4``, all on the trainer host,
  FSDP-style) coexisting in the trainer procs:
  **fails at trainer-rank-0 -> storage HiXL Connect ❌**.

  The error chain matches HiXL's official troubleshoot guide
  ``hixl-troubleshoot/references/guides.md`` section "场景二":

      [ERROR] HCCP ra_hdc_typical_mr ret(-13) phyId(0)
      [ERROR] HCCL hrtRaRegGlobalMr errNo[0x0000000005000013] ra reg global mr fail.
              return[128102], size[~80 MB], access[7]
      [ERROR] HCCL LocalRdmaRmaBuffer::Init call trace: hcclRet -> 19
      [ERROR] HCCL hccl_one_sided_service.cc:826 [RegisterBoundMems] hcclRet -> 19
      [ERROR] HCCL hccl_one_sided_service.cc:635 [PrepareFullMesh] hcclRet -> 19
      [ERROR] GE  hixl_impl.cc:251 Connect remote engine:...  503900

  That is: HiXL's internal ``LocalRdmaRmaBuffer`` (its own ~80 MB bound-mem
  book-keeping buffer, *not* the user's staging pool -- our
  ``HcclCommBindMem`` of the user pool already succeeded at this point)
  fails to register at the RA HDC driver layer with ``ret=-13`` when a
  same-host multi-rank HCCL PG is already live on NPUs 0-3.

Topology of this repro
----------------------

    host A (trainer)                           host B (generator+storage)
    +-----------------------------+            +---------------------------+
    | TrainerRank0 (NPU 0)  ------+--- RoCE ---+--> StorageVolume (NPU 7) |
    | TrainerRank1 (NPU 1)  (idle HiXL, active FSDP)                      |
    | TrainerRank2 (NPU 2)  (idle HiXL, active FSDP)   GeneratorRank      |
    | TrainerRank3 (NPU 3)  (idle HiXL, active FSDP)   (idle HiXL)        |
    +-----------------------------+            +---------------------------+
             ^                                           (NPU 0, read path
             |                                            not exercised)
      FSDP-style HCCL PG
      (world=4 all-reduce)

Expected output
---------------

Success case (``--trainer-world 1``):

    [driver] step 0: push=... pull=...   <- completes, prints bandwidth

Failure case (``--trainer-world 4``, this repro's default):

    [ERROR] HCCP(...) ra_hdc_typical_mr ret(-13) phyId(0)
    [ERROR] ... hixl_connect(<remote engine>): 503900
    [driver] push failed: ...
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


def _bootstrap_single_npu(dev_id: int, trainer_rank: int | None = None):
    """Bootstrap for a single-proc / single-NPU actor.

    ``dev_id`` masks the physical NPU to local index 0 in the proc.
    ``trainer_rank`` -- when set, makes this proc the FSDP rank ``trainer_rank``
    for HCCL PG init purposes (sets ``LOCAL_RANK`` / ``RANK`` accordingly);
    leave ``None`` for non-trainer procs (generator, storage).
    """

    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ["MONARCH_NPU_DEVICE"] = "0"
        os.environ["MONARCH_HIXL_TRANSPORT"] = "roce"
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        os.environ["TORCHSTORE_MONARCH_RDMA_EAGER_D2H"] = "0"
        os.environ["TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE"] = "npu:0"
        os.environ.setdefault("TORCHSTORE_MONARCH_RDMA_POOL_MB", "4096")
        # torchstore LocalRankStrategy reads LOCAL_RANK to pick a volume.
        # With ``num_storage_volumes=1`` any value routes to volume 0, so we
        # just need the var to exist.
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


async def _run(
    workers: list[str],
    trainer_world: int,
    payload_gb: float,
    iters: int,
    num_tensors: int,
) -> int:
    import torch
    import torch_npu  # noqa: F401
    import torchstore as ts
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers
    from monarch.actor import Actor, endpoint

    configure(default_transport=ChannelTransport.TcpWithHostname)

    if len(workers) != 2:
        print(f"ERROR: need exactly 2 workers, got {len(workers)}", flush=True)
        return 2
    if trainer_world < 1 or trainer_world > 8:
        print(f"ERROR: --trainer-world must be 1..8, got {trainer_world}", flush=True)
        return 2

    print(f"[driver] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized

    host_trainer = hosts.slice(hosts=0)  # host A
    host_generator = hosts.slice(hosts=1)  # host B

    # Trainer: ``trainer_world`` procs on host A.  Each is a separate proc
    # mesh with its own NPU-specific bootstrap so we can force rank i onto
    # physical NPU i.  (Monarch's ``per_host={"npus": N}`` *actor-style*
    # spawn does not auto-set ``ASCEND_RT_VISIBLE_DEVICES`` / ``LOCAL_RANK``
    # -- that's only done by mesh_controller for SPMD jobs.)
    trainer_meshes = []
    for r in range(trainer_world):
        m = host_trainer.spawn_procs(
            per_host={"npus": 1},
            bootstrap=_bootstrap_single_npu(dev_id=r),
        )
        trainer_meshes.append(m)
    # Generator & storage: one proc each on host B, NPU 0 and NPU 7.
    generator_mesh = host_generator.spawn_procs(
        per_host={"npus": 1}, bootstrap=_bootstrap_single_npu(0)
    )
    storage_mesh = host_generator.spawn_procs(
        per_host={"npus": 1},
        bootstrap=_bootstrap_single_npu(int(os.environ.get("REPRO_STORAGE_NPU", "7"))),
    )

    class TrainerActor(Actor):
        def __init__(self) -> None:
            self.buf = None
            self.nbytes = 0

        @endpoint
        async def whoami(self) -> dict:
            return {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "rank": os.environ.get("RANK"),
                "world_size": os.environ.get("WORLD_SIZE"),
            }

        @endpoint
        async def fsdp_prewarm(
            self, master_addr: str, master_port: int, rank: int, world_size: int
        ) -> dict:
            """Stand-in for FSDP's HCCL PG init: build a world_size-rank PG
            on the trainer host, run one all_reduce to force communicator
            creation, and leave it alive so subsequent HiXL Connect has to
            coexist with it in the same proc."""
            import torch.distributed as dist

            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)
            os.environ["RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = str(rank)  # all on same host
            os.environ["WORLD_SIZE"] = str(world_size)

            if not dist.is_initialized():
                dist.init_process_group(
                    backend="hccl",
                    init_method=f"tcp://{master_addr}:{master_port}",
                    rank=rank,
                    world_size=world_size,
                    timeout=datetime.timedelta(seconds=120),
                )

            probe = torch.ones(8, dtype=torch.float32, device="npu:0") * (rank + 1)
            dist.all_reduce(probe)
            torch.npu.synchronize()
            return {
                "rank": rank,
                "world_size": world_size,
                "probe_sum": float(probe.sum().item()),
            }

        @endpoint
        async def prepare_payload(self, nbytes: int, num_tensors: int = 1) -> dict:
            """Rank 0 only: allocate ``num_tensors`` independent NPU tensors.

            ``num_tensors=1`` -> single 2 MB-aligned pool-allocated flat buffer
            (same path as test_weight_sync_2node.py + standalone smoke).

            ``num_tensors=N>1`` -> N independent ``torch.empty(..., device="npu:0")``
            tensors, **not pool-aligned**, mimicking what forge sees after
            FSDP's ``state_dict_for_sync()`` gathers per-parameter tensors
            from sharded state.  Each ts.put on such a tensor forces
            torchstore's transport layer to set up its own HiXL RDMA
            buffer for the source address -- which is what seems to break
            in forge but not in the pool-aligned case.
            """
            block = 2 * 1024 * 1024
            if nbytes % block != 0:
                nbytes += block - (nbytes % block)

            self.num_tensors = num_tensors
            self.nbytes = nbytes

            if num_tensors == 1:
                from monarch._src.rdma.xdma import alloc_aligned_tensor

                buf, _raw = alloc_aligned_tensor(
                    (nbytes,), dtype=torch.uint8, device="npu:0"
                )
                buf.fill_(7)
                self.buf = buf
                self.tensors = None
            else:
                per = nbytes // num_tensors
                self.buf = None
                self.tensors = []
                for _ in range(num_tensors):
                    t = torch.empty((per,), dtype=torch.uint8, device="npu:0")
                    t.fill_(7)
                    self.tensors.append(t)
            torch.npu.synchronize()
            return {"nbytes": nbytes, "num_tensors": num_tensors}

        @endpoint
        async def push(self, version: int) -> dict:
            """Rank 0 only: ``ts.put`` every tensor prepared in
            ``prepare_payload`` under a distinct key.
            """
            t0 = time.perf_counter()
            if self.tensors is not None:
                # Independent-tensor path (mimics forge).
                for idx, t in enumerate(self.tensors):
                    key = f"repro_ver_{version % 2:010d}.tensor{idx:04d}"
                    await ts.put(key, t)
                nchunks = len(self.tensors)
            else:
                # Single flat buffer path.
                key = f"repro_ver_{version % 2:010d}.flat"
                await ts.put(key, self.buf)
                nchunks = 1
            put_s = time.perf_counter() - t0
            return {"bytes": self.nbytes, "put_s": put_s, "num_chunks": nchunks}

    class GeneratorActor(Actor):
        def __init__(self) -> None:
            self.buf = None
            self.nbytes = 0

        @endpoint
        async def whoami(self) -> dict:
            return {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            }

        @endpoint
        async def allocate(self, nbytes: int, num_tensors: int = 1) -> dict:
            self.num_tensors = num_tensors
            self.nbytes = nbytes
            if num_tensors == 1:
                from monarch._src.rdma.xdma import alloc_aligned_tensor

                buf, _raw = alloc_aligned_tensor(
                    (nbytes,), dtype=torch.uint8, device="npu:0"
                )
                buf.zero_()
                self.buf = buf
                self.tensors = None
            else:
                per = nbytes // num_tensors
                self.buf = None
                self.tensors = [
                    torch.empty((per,), dtype=torch.uint8, device="npu:0").zero_()
                    for _ in range(num_tensors)
                ]
            torch.npu.synchronize()
            return {"nbytes": nbytes, "num_tensors": num_tensors}

        @endpoint
        async def pull(self, version: int) -> dict:
            t0 = time.perf_counter()
            if self.tensors is not None:
                for idx, t in enumerate(self.tensors):
                    key = f"repro_ver_{version % 2:010d}.tensor{idx:04d}"
                    await ts.get(key, inplace_tensor=t)
                nchunks = len(self.tensors)
            else:
                key = f"repro_ver_{version % 2:010d}.flat"
                await ts.get(key, inplace_tensor=self.buf)
                nchunks = 1
            torch.npu.synchronize()
            pull_s = time.perf_counter() - t0
            return {"bytes": self.nbytes, "pull_s": pull_s, "num_chunks": nchunks}

    trainers = [
        m.spawn(f"trainer_r{r}", TrainerActor) for r, m in enumerate(trainer_meshes)
    ]
    generator = generator_mesh.spawn("generator", GeneratorActor)

    # Print who's where.
    for r, t in enumerate(trainers):
        info = await t.whoami.call_one()
        print(f"[driver] trainer rank {r} = {info}", flush=True)
    info_g = await generator.whoami.call_one()
    print(f"[driver] generator        = {info_g}", flush=True)

    # ---- FSDP-style HCCL PG prewarm on the trainer host --------------------
    if trainer_world > 1:
        # Rendezvous via the trainer host's hostname.  rank 0 listens, 1..N-1
        # connect.
        rank0_info = await trainers[0].whoami.call_one()
        rdvz_host = rank0_info["host"]
        rdvz_port = int(os.environ.get("REPRO_FSDP_PORT", "29580"))
        print(
            f"[driver] FSDP-style HCCL PG prewarm @ {rdvz_host}:{rdvz_port} "
            f"(world={trainer_world}) ...",
            flush=True,
        )

        async def _prewarm_rank(rank: int) -> dict:
            return await trainers[rank].fsdp_prewarm.call_one(
                master_addr=rdvz_host,
                master_port=rdvz_port,
                rank=rank,
                world_size=trainer_world,
            )

        prewarm_results = await asyncio.gather(
            *[_prewarm_rank(r) for r in range(trainer_world)]
        )
        print(
            f"[driver] HCCL PG alive on {trainer_world} ranks:",
            [r["probe_sum"] for r in prewarm_results],
            flush=True,
        )

    # ---- Torchstore init (same as test_weight_sync_2node.py) --------------
    from torchstore.strategy import LocalRankStrategy
    from torchstore.transport import TransportType

    print(
        "[driver] Initializing torchstore (LocalRankStrategy + MonarchRDMA) ...",
        flush=True,
    )
    await ts.initialize(
        num_storage_volumes=1,
        strategy=LocalRankStrategy(default_transport_type=TransportType.MonarchRDMA),
        mesh=storage_mesh,
    )
    print("[driver] torchstore ready", flush=True)

    # ---- Allocate payload on rank 0 and on the generator ------------------
    payload_bytes = int(payload_gb * 1024**3)
    print(
        f"[driver] Allocating {payload_gb:.2f} GB payload on trainer rank 0 "
        f"and on the generator ...",
        flush=True,
    )
    rank0 = trainers[0]
    prep = await rank0.prepare_payload.call_one(
        nbytes=payload_bytes, num_tensors=num_tensors
    )
    actual_bytes = prep["nbytes"]
    await generator.allocate.call_one(nbytes=actual_bytes, num_tensors=num_tensors)
    print(f"[driver] Ready ({actual_bytes / 1024**3:.3f} GB, 2 MB-aligned)", flush=True)

    # ---- Drive push/pull iterations ---------------------------------------
    print("=" * 72, flush=True)
    print(" step | push_s | push_GBps | pull_s | pull_GBps", flush=True)
    print("=" * 72, flush=True)
    results = []
    for step in range(iters):
        try:
            push_r = await rank0.push.call_one(version=step)
        except Exception as e:
            print(
                f"[driver] push(step={step}) RAISED: {type(e).__name__}: {e}",
                flush=True,
            )
            await ts.shutdown()
            return 1
        push_s = push_r["put_s"]

        try:
            pull_r = await generator.pull.call_one(version=step)
        except Exception as e:
            print(
                f"[driver] pull(step={step}) RAISED: {type(e).__name__}: {e}",
                flush=True,
            )
            await ts.shutdown()
            return 1
        pull_s = pull_r["pull_s"]

        push_gbps = actual_bytes / push_s / 1024**3 if push_s > 0 else 0.0
        pull_gbps = actual_bytes / pull_s / 1024**3 if pull_s > 0 else 0.0
        print(
            f" {step:4d} | {push_s:6.2f} | {push_gbps:9.2f} | "
            f"{pull_s:6.2f} | {pull_gbps:9.2f}",
            flush=True,
        )
        results.append((push_s, pull_s))

    print("=" * 72, flush=True)
    if len(results) > 1:
        warm = results[1:]
        avg_push = sum(r[0] for r in warm) / len(warm)
        avg_pull = sum(r[1] for r in warm) / len(warm)
        print(
            f"[driver] Steady-state: push={avg_push:.2f}s "
            f"({actual_bytes / avg_push / 1024**3:.2f} GB/s), "
            f"pull={avg_pull:.2f}s "
            f"({actual_bytes / avg_pull / 1024**3:.2f} GB/s)",
            flush=True,
        )
    await ts.shutdown()
    print("[driver] OK", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workers",
        required=True,
        help="Comma-separated Monarch worker addresses: "
        "tcp://<trainer_host>:22222,tcp://<generator_host>:22222",
    )
    parser.add_argument(
        "--trainer-world",
        type=int,
        default=4,
        help="Number of trainer ranks (== NPUs used on trainer host). "
        "Default 4 reproduces the forge GRPO 4-rank FSDP scenario.",
    )
    parser.add_argument(
        "--payload-gb",
        type=float,
        default=1.5,
        help="Size of the fake weight tensor in GiB (default 1.5 -- "
        "about the size of Qwen3-0.6B bf16 weights).",
    )
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument(
        "--num-tensors",
        type=int,
        default=1,
        help="If 1, send the payload as a single 2MB-aligned pool-allocated "
        "flat buffer.  If >1, allocate that many independent "
        "``torch.empty(..., device='npu:0')`` tensors (NOT pool-aligned) "
        "and ``ts.put`` each separately, mimicking forge/torchtitan's "
        "``for name, t in state_dict.items(): ts.put(name, t)`` pattern "
        "(e.g. 311 for Qwen3-0.6B).",
    )
    args = parser.parse_args()

    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    return asyncio.run(
        _run(workers, args.trainer_world, args.payload_gb, args.iters, args.num_tensors)
    )


if __name__ == "__main__":
    sys.exit(main())
