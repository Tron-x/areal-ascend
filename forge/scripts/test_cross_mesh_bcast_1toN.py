#!/usr/bin/env python3
"""MVP Step 2 for the storage-as-reflector design: 1-to-N cross-mesh HCCL bcast.

Builds on test_cross_mesh_bcast.py. Moves from the 2-rank smoke to the
real target topology: one ``sender`` proc (storage role) on host A plus
``N`` ``receiver`` procs (inference TP workers role) on host B, all in a
single ``world_size=1+N`` HCCL group. Sender broadcasts, every receiver
is expected to get the same bytes.

This is the direct next building block for the
``CollectiveBroadcastBackend`` (weight_sync.md §7.4): once this passes,
the only thing left before the backend can be wired up is "trainer side
still HiXL-put to storage at the same time as storage HCCL-bcasts
out". That's a separate coexistence question from "does 1-to-N HCCL
work cross-mesh" and will be a third MVP.

Topology
--------

    host A (monarch1, role=sender)              host B (monarch2, role=receivers)
    +-------------------------+                 +---------------------------+
    |  SenderActor (NPU X)    |<---RoCE--->  ...rcv rank 1..N on NPU 0..N-1
    |    rank=0               |                 |    each on its own NPU    |
    |                         |                 |    all in same PG         |
    +-------------------------+                 +---------------------------+

A few MVP-Step-2 specific choices worth flagging:

* ``per_host={"npus": N}`` on the receiver mesh. This tells Monarch to
  spawn N procs on host B and give each proc one of host B's
  first-N NPUs via ``ASCEND_RT_VISIBLE_DEVICES`` masking. Each proc sees
  exactly one NPU (locally device index 0). Matches the real forge
  inference TP layout; also matches how the coexistence bug shape gets
  stressed (N concurrent procs each trying to participate in the same
  collective).

* No per-proc bootstrap closure. Since the Forge
  ``StorageEnvSetter`` / ``EnvSetter`` pattern is already in tree, we
  intentionally use the plain Monarch ``bootstrap=...`` closure here to
  keep this MVP scaffold independent of Forge-specific env-setter
  plumbing. Bug B (bootstrap silently skipped) does NOT reproduce in
  this environment because the bootstrap closure is minimal and only
  sets env vars that torch_npu happily re-reads.

* The sender is still 1 proc, 1 NPU, rank 0. No change vs MVP Step 1.
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


def _bootstrap_sender(dev_id: int):
    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


def _bootstrap_receiver_npu_pool(npu_base: int, tp: int):
    """Bootstrap for the per-host={npus: N} spawn.

    Important Monarch quirk (also documented in
    ``forge/scripts/repro_hixl_multirank_hccl.py`` line 144): the
    ``per_host={"npus": N}`` spawn path does **not** set ``LOCAL_RANK``
    or ``RANK`` env vars in the child procs.  So this bootstrap can
    only set per-host env vars, not per-proc ones -- per-proc NPU
    selection must happen in the actor endpoints via Monarch's
    ``current_rank()`` API (see ``ReceiverActor.init_group``).

    Why the receiver procs share the full NPU pool (instead of each
    being masked to one NPU): when N procs on the same host each mask
    themselves to a single physical NPU, every proc's
    ``torch.npu.current_device()`` returns 0 (the local index of the
    single visible NPU), so HCCL's ``hcclCommInitRootInfo`` topology
    discovery sees "N ranks on (host, device 0)" and fails with
    ``Ranktable_Detect_Failed(EI0015): rank num[K] is different with
    rank list size[M]``.  Giving every proc the full pool and picking
    device via ``set_device(rank_in_mesh)`` matches FSDP's approach
    in ``fsdp_engine.py`` and keeps each rank on a distinct
    physical NPU.
    """

    def _bootstrap():
        pool = ",".join(str(npu_base + i) for i in range(tp))
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = pool
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

        import torch  # noqa: F401
        import torch_npu  # noqa: F401

        # Do NOT call torch.npu.set_device here -- rank is unknown at
        # bootstrap time.  Endpoints pin the device via current_rank().

    return _bootstrap


async def _run(
    workers: list[str],
    sender_npu: int,
    receiver_npu_base: int,
    tp: int,
    payload_gb: float,
    iters: int,
    master_port: int,
) -> int:
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers
    from monarch.actor import Actor, current_rank, endpoint

    configure(default_transport=ChannelTransport.TcpWithHostname)

    if len(workers) != 2:
        print(f"ERROR: need exactly 2 workers, got {len(workers)}", flush=True)
        return 2
    if tp < 2:
        print(
            f"ERROR: tp must be >=2 for the 1-to-N MVP, got {tp}. "
            "Use test_cross_mesh_bcast.py for the 2-rank smoke.",
            flush=True,
        )
        return 2

    print(f"[driver] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized

    host_sender = hosts.slice(hosts=0)
    host_receiver = hosts.slice(hosts=1)

    sender_mesh = host_sender.spawn_procs(
        per_host={"npus": 1},
        bootstrap=_bootstrap_sender(sender_npu),
    )
    # TP receivers: N procs on host B. All procs see the full NPU pool
    # (``ASCEND_RT_VISIBLE_DEVICES=<base>,<base+1>,...,<base+tp-1>``);
    # each proc then calls ``torch.npu.set_device(local_rank)`` to pick
    # its distinct physical NPU.  See ``_bootstrap_receiver_npu_pool``
    # docstring for why the per-proc single-NPU mask breaks HCCL's
    # topology ranktable discovery.
    receiver_mesh = host_receiver.spawn_procs(
        per_host={"npus": tp},
        bootstrap=_bootstrap_receiver_npu_pool(receiver_npu_base, tp),
    )

    world_size = 1 + tp

    class SenderActor(Actor):
        def __init__(self) -> None:
            self.pg = None
            self.buf = None

        @endpoint
        async def whoami(self) -> dict:
            return {
                "role": "sender",
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            }

        @endpoint
        async def init_group(
            self, master_addr: str, master_port: int, world_size: int
        ) -> dict:
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            from areal.engine.core.distributed import init_custom_process_group

            init_method = f"tcp://{master_addr}:{master_port}"
            print(
                f"[sender] init_process_group rank=0 world={world_size}",
                flush=True,
            )
            self.pg = init_custom_process_group(
                backend="hccl",
                init_method=init_method,
                world_size=world_size,
                rank=0,
                group_name="storage_reflector_mvp_1toN",
                timeout=datetime.timedelta(seconds=180),
            )
            probe = torch.ones(8, dtype=torch.float32, device="npu:0")
            dist.all_reduce(probe, group=self.pg)
            torch.npu.synchronize()
            return {
                "role": "sender",
                "probe_sum": float(probe.sum().item()),
                "expected_probe_sum": float(8 * world_size),
            }

        @endpoint
        async def prepare_payload(self, nbytes: int) -> dict:
            import torch
            import torch_npu  # noqa: F401

            self.buf = torch.empty((nbytes,), dtype=torch.uint8, device="npu:0")
            self.buf.fill_(42)
            torch.npu.synchronize()
            return {"nbytes": self.buf.numel()}

        @endpoint
        async def broadcast(self, step: int) -> dict:
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            # Step-dependent fill so every receiver can distinguish
            # between iterations and spot a stale-buffer regression.
            self.buf.fill_((step + 1) % 256)
            torch.npu.synchronize()

            t0 = time.perf_counter()
            dist.broadcast(self.buf, src=0, group=self.pg)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - t0
            return {
                "step": step,
                "bytes": int(self.buf.numel()),
                "elapsed_s": elapsed,
                "fill_value": (step + 1) % 256,
            }

    class ReceiverActor(Actor):
        def __init__(self) -> None:
            self.pg = None
            self.buf = None
            self.tp_rank = None

        @endpoint
        async def whoami(self) -> dict:
            import torch
            import torch_npu  # noqa: F401

            mesh_rank = current_rank().rank
            return {
                "role": "receiver",
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "mesh_rank": mesh_rank,
                "device_count": torch.npu.device_count(),
                # Before init_group picks the device, current_device
                # is 0 for everyone; this field is informative only.
            }

        @endpoint
        async def init_group(
            self,
            master_addr: str,
            master_port: int,
            world_size: int,
            rank_offset: int = 1,
        ) -> dict:
            """Rank in the HCCL group = rank_offset + Monarch mesh rank.

            Monarch does not propagate ``LOCAL_RANK`` into per-host spawned
            procs, so we read the mesh rank via ``current_rank()`` (Monarch
            API) and use it both for the HCCL rank assignment and for
            ``torch.npu.set_device`` (pinning this proc to its distinct
            physical NPU -- without this every proc would land on NPU 0
            and HCCL's topology detection would blow up with the
            ``rank num[K] != rank list size[M]`` error we saw earlier).
            """
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
                f"init_process_group rank={my_rank} world={world_size}",
                flush=True,
            )
            self.pg = init_custom_process_group(
                backend="hccl",
                init_method=init_method,
                world_size=world_size,
                rank=my_rank,
                group_name="storage_reflector_mvp_1toN",
                timeout=datetime.timedelta(seconds=180),
            )
            probe = torch.ones(8, dtype=torch.float32, device=self.device)
            dist.all_reduce(probe, group=self.pg)
            torch.npu.synchronize()
            return {
                "role": "receiver",
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
        async def recv_and_check(self, step: int) -> dict:
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            t0 = time.perf_counter()
            dist.broadcast(self.buf, src=0, group=self.pg)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - t0
            expected = (step + 1) % 256
            front = int(self.buf[0].cpu().item())
            mid = int(self.buf[self.buf.numel() // 2].cpu().item())
            back = int(self.buf[-1].cpu().item())
            ok = front == expected and mid == expected and back == expected
            return {
                "step": step,
                "tp_rank": self.tp_rank,
                "bytes": int(self.buf.numel()),
                "elapsed_s": elapsed,
                "ok": ok,
                "expected": expected,
                "front": front,
                "mid": mid,
                "back": back,
            }

    sender = sender_mesh.spawn("sender", SenderActor)
    receiver = receiver_mesh.spawn("receiver", ReceiverActor)

    info_s = await sender.whoami.call_one()
    print(f"[driver] sender   = {info_s}", flush=True)
    recv_infos = await receiver.whoami.call()
    for r, info in recv_infos.items():
        print(f"[driver] receiver {r} = {info}", flush=True)

    rdvz_host = info_s["host"]
    print(
        f"[driver] HCCL world=1+{tp} PG rendezvous @ {rdvz_host}:{master_port}",
        flush=True,
    )

    async def _await_fut(fut):
        return await fut

    # Init sender (single rank 0) and all receivers (ranks 1..TP)
    # concurrently. Each receiver proc will pick its own rank from
    # LOCAL_RANK.
    try:
        init_results = await asyncio.wait_for(
            asyncio.gather(
                _await_fut(
                    sender.init_group.call_one(
                        master_addr=rdvz_host,
                        master_port=master_port,
                        world_size=world_size,
                    )
                ),
                _await_fut(
                    receiver.init_group.call(
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
        print(
            f"[driver] init timed out after 180s "
            f"(world={world_size}) -- rendezvous failed",
            flush=True,
        )
        return 1
    sender_info, recv_init_map = init_results
    print(f"[driver] sender init = {sender_info}", flush=True)
    for r, info in recv_init_map.items():
        print(f"[driver] receiver init {r} = {info}", flush=True)

    payload_bytes = int(payload_gb * 1024**3)
    print(
        f"[driver] allocating {payload_gb:.2f} GB payload "
        f"(sender + {tp} receivers) ...",
        flush=True,
    )
    await asyncio.gather(
        _await_fut(sender.prepare_payload.call_one(nbytes=payload_bytes)),
        _await_fut(receiver.allocate.call(nbytes=payload_bytes)),
    )

    print("=" * 80, flush=True)
    print(
        " step | bcast_s | GBps  | total_bytes_moved | aggregate_GBps | all_verified",
        flush=True,
    )
    print("=" * 80, flush=True)

    results = []
    for step in range(iters):
        try:
            s_r, r_map = await asyncio.gather(
                _await_fut(sender.broadcast.call_one(step=step)),
                _await_fut(receiver.recv_and_check.call(step=step)),
            )
        except Exception as e:
            print(
                f"[driver] broadcast(step={step}) raised: {type(e).__name__}: {e}",
                flush=True,
            )
            return 1

        # bcast is collective; take max of all rank elapsed.
        elapsed_list = [s_r["elapsed_s"]] + [r["elapsed_s"] for r in r_map.values()]
        bcast_s = max(elapsed_list)
        per_link_gbps = payload_bytes / bcast_s / 1024**3 if bcast_s > 0 else 0.0
        # Aggregate view: 1-to-N bcast "moves" payload_bytes * N bytes
        # across the NIC layer (one copy to each receiver).  Depending on
        # HCCL's implementation this can be tree-topology so the
        # aggregate number caps at a few x single-NIC, not N x. Still a
        # useful sanity number.
        total_moved = payload_bytes * tp
        aggregate_gbps = total_moved / bcast_s / 1024**3 if bcast_s > 0 else 0.0
        all_ok = all(r["ok"] for r in r_map.values())
        print(
            f" {step:4d} | {bcast_s:7.2f} | {per_link_gbps:5.2f} | "
            f"{total_moved / 1024**3:17.2f} | {aggregate_gbps:14.2f} | "
            f"{all_ok}",
            flush=True,
        )
        if not all_ok:
            for r_key, r in r_map.items():
                if not r["ok"]:
                    print(
                        f"[driver] VERIFY FAILED at {r_key}: "
                        f"expected={r['expected']} front={r['front']} "
                        f"mid={r['mid']} back={r['back']}",
                        flush=True,
                    )
            return 1
        results.append(bcast_s)

    print("=" * 80, flush=True)
    if len(results) > 1:
        warm = results[1:]
        avg = sum(warm) / len(warm)
        per_link = payload_bytes / avg / 1024**3
        agg = (payload_bytes * tp) / avg / 1024**3
        print(
            f"[driver] steady-state (iters 1..{iters - 1}): "
            f"bcast={avg:.2f}s, per-link={per_link:.2f} GB/s, "
            f"aggregate={agg:.2f} GB/s",
            flush=True,
        )
    print("[driver] OK", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", nargs="+", required=True)
    parser.add_argument("--sender-npu", type=int, default=4)
    parser.add_argument(
        "--receiver-npu-base",
        type=int,
        default=0,
        help="Receivers occupy NPUs base..base+tp-1 on host B.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Number of receiver TP ranks on host B (default 4, matches "
        "our target Qwen3/Qwen2.5 TP=4 gen topology).",
    )
    parser.add_argument("--payload-gb", type=float, default=1.5)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--master-port", type=int, default=29721)
    args = parser.parse_args()

    rc = asyncio.run(
        _run(
            workers=args.workers,
            sender_npu=args.sender_npu,
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
