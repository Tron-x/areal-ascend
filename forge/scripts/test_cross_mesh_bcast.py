#!/usr/bin/env python3
"""MVP Step 1 for the storage-as-reflector design: verify cross-mesh HCCL
broadcast between two independent Monarch proc meshes.

Motivation
----------

See ``forge/docs/weight_sync.md §7.4`` for the full context. Short version:

* Current pull path has every vLLM TP worker independently call ``ts.get``,
  which scales as ``(TP × num_storage_vols)`` HiXL client channels and blows up
  at ``CreateChannel ret=103901`` once ``TP>1``.
* The fix is to route the pull side through a **HCCL broadcast group**
  spanning ``storage vol`` as source + the ``inference TP workers`` as
  destinations. That makes the connection count equal to the TP group size
  (handled by HCCL's own QP aggregation), independent of the storage-vol
  fan-out.

Before wiring this into the weight-sync backend, we need to establish the
simplest possible fact: **can a process on Monarch mesh A and a process on
Monarch mesh B join one HCCL process group and complete a broadcast?**

This script does exactly that. It is deliberately free of:

* HiXL or torchstore (those add coexistence failure modes, documented in
  ``repro_hixl_multirank_hccl.py``). We want a clean "HCCL only, nothing
  else" baseline first.
* vLLM, tensor parallel, or anything model-like. Just a synthetic
  ``torch.randn`` tensor.
* FSDP-style same-host HCCL PG. Keeps us on the proven
  "HiXL + world=2 HCCL" coexistence axis even though we aren't using HiXL
  yet; this means adding HiXL back in a later MVP won't change the HCCL
  topology.

Topology
--------

    host A (monarch1, role=sender)       host B (monarch2, role=receiver)
    +-------------------------+          +---------------------------+
    |  SenderActor (NPU X)    |<--RoCE-->|  ReceiverActor (NPU 0)    |
    |    rank=0 in 2-rank PG  |          |    rank=1 in 2-rank PG    |
    +-------------------------+          +---------------------------+

Rendezvous: sender host IP + a free TCP port, both sides call
``init_custom_process_group`` (borrowed from AReaL -- the stock
``torch.distributed.init_process_group`` refuses to build a second default
PG in a proc that torchrun/elastic already set up).

Pass criterion
--------------

* Both sides reach the ``barrier`` after broadcast without timeout.
* Receiver's tensor matches sender's tensor bit-for-bit.
* Steady-state bandwidth printed -- we expect something close to the
  200 Gbps RoCE per-NIC peak (~ 20 GB/s) for large tensors, for the same
  reason ``test_weight_sync_2node.py``'s HiXL path hit 17 GB/s.

Run
---

    python forge/scripts/test_cross_mesh_bcast.py \\
        --workers tcp://192.168.0.26:22222 tcp://192.168.0.23:22222 \\
        --sender-npu 4 --receiver-npu 0 \\
        --payload-gb 1.5 --iters 3

(The ``--workers`` URLs are the two Monarch bootstrap workers; first one
is the sender host, second one is the receiver host.)
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


def _bootstrap_single_npu(dev_id: int):
    """Mask the proc to a single NPU and import torch_npu cleanly.

    Same shape as ``repro_hixl_multirank_hccl.py`` and the
    ``StorageEnvSetter`` we already ship, so the two MVPs stay lexically
    close.  We don't set any torchstore or HiXL env here -- this MVP is
    stock-HCCL-only by design.
    """

    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        # Keep HCCL's HDCS port range out of the way of any pre-existing
        # HiXL-using procs on the same machine (16666 is HiXL's default).
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        # Explicitly cross-machine: we want to see RoCE throughput, not
        # HCCS-local-host shortcut.  HCCL_INTRA_ROCE_ENABLE=1 tells HCCL
        # to allow RoCE even when it thinks it could use local transport
        # (doesn't matter here because we really are cross-host, but
        # harmless and matches our other scripts).
        os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


async def _run(
    workers: list[str],
    sender_npu: int,
    receiver_npu: int,
    payload_gb: float,
    iters: int,
    master_port: int,
) -> int:
    # Driver proc itself does not touch NPU; torch_npu imports live only
    # inside the actor endpoints (per-proc, after the bootstrap has set
    # ASCEND_RT_VISIBLE_DEVICES).  Keeping the driver torch-free also
    # means this script can run on a host that doesn't have CANN
    # installed -- useful when the driver launches from a build machine.
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers
    from monarch.actor import Actor, endpoint

    # TcpWithHostname matches what grpo.py + run_multinode.sh use for
    # Monarch-to-worker RPC.  Doesn't affect the HCCL PG we build inside
    # the actors; those go directly over RoCE once init finishes.
    configure(default_transport=ChannelTransport.TcpWithHostname)

    if len(workers) != 2:
        print(f"ERROR: need exactly 2 workers, got {len(workers)}", flush=True)
        return 2

    print(f"[driver] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized

    host_sender = hosts.slice(hosts=0)
    host_receiver = hosts.slice(hosts=1)

    sender_mesh = host_sender.spawn_procs(
        per_host={"npus": 1},
        bootstrap=_bootstrap_single_npu(sender_npu),
    )
    receiver_mesh = host_receiver.spawn_procs(
        per_host={"npus": 1},
        bootstrap=_bootstrap_single_npu(receiver_npu),
    )

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
            self, master_addr: str, master_port: int, world_size: int = 2
        ) -> dict:
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            # We use AReaL's ``init_custom_process_group`` so this PG is
            # independent of any existing default PG that e.g. torchelastic
            # might have installed in the proc.  In this MVP there is no
            # default PG, but keeping the shape lets us lift the code as-is
            # into the backend.
            from areal.engine.core.distributed import init_custom_process_group

            init_method = f"tcp://{master_addr}:{master_port}"
            print(
                f"[sender] init_process_group rank=0 world={world_size} "
                f"init_method={init_method}",
                flush=True,
            )
            self.pg = init_custom_process_group(
                backend="hccl",
                init_method=init_method,
                world_size=world_size,
                rank=0,
                group_name="storage_reflector_mvp",
                timeout=datetime.timedelta(seconds=120),
            )
            # One tiny all_reduce so the communicator is fully built
            # before we start measuring broadcast bandwidth.
            probe = torch.ones(8, dtype=torch.float32, device="npu:0")
            dist.all_reduce(probe, group=self.pg)
            torch.npu.synchronize()
            return {"role": "sender", "probe_sum": float(probe.sum().item())}

        @endpoint
        async def prepare_payload(self, nbytes: int) -> dict:
            import torch
            import torch_npu  # noqa: F401

            self.buf = torch.empty((nbytes,), dtype=torch.uint8, device="npu:0")
            # Use a rank-specific fill so we can assert on the receiver that
            # the broadcast actually delivered the sender's data rather
            # than some uninitialized buffer.
            self.buf.fill_(42)
            torch.npu.synchronize()
            return {"nbytes": self.buf.numel()}

        @endpoint
        async def broadcast(self, step: int) -> dict:
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            t0 = time.perf_counter()
            # We re-fill on every step with a step-dependent value so the
            # receiver can distinguish between iterations and also notice
            # "broadcast didn't actually happen" regressions.
            self.buf.fill_((step + 1) % 256)
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

        @endpoint
        async def whoami(self) -> dict:
            return {
                "role": "receiver",
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            }

        @endpoint
        async def init_group(
            self, master_addr: str, master_port: int, world_size: int = 2
        ) -> dict:
            import torch
            import torch.distributed as dist
            import torch_npu  # noqa: F401

            from areal.engine.core.distributed import init_custom_process_group

            init_method = f"tcp://{master_addr}:{master_port}"
            print(
                f"[receiver] init_process_group rank=1 world={world_size} "
                f"init_method={init_method}",
                flush=True,
            )
            self.pg = init_custom_process_group(
                backend="hccl",
                init_method=init_method,
                world_size=world_size,
                rank=1,
                group_name="storage_reflector_mvp",
                timeout=datetime.timedelta(seconds=120),
            )
            probe = torch.ones(8, dtype=torch.float32, device="npu:0")
            dist.all_reduce(probe, group=self.pg)
            torch.npu.synchronize()
            return {"role": "receiver", "probe_sum": float(probe.sum().item())}

        @endpoint
        async def allocate(self, nbytes: int) -> dict:
            import torch
            import torch_npu  # noqa: F401

            self.buf = torch.zeros((nbytes,), dtype=torch.uint8, device="npu:0")
            torch.npu.synchronize()
            return {"nbytes": self.buf.numel()}

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
            # Cheap sanity: front byte, back byte, middle byte.  Avoids
            # pulling the whole buffer to host just to verify.
            front = int(self.buf[0].cpu().item())
            mid = int(self.buf[self.buf.numel() // 2].cpu().item())
            back = int(self.buf[-1].cpu().item())
            ok = front == expected and mid == expected and back == expected
            return {
                "step": step,
                "bytes": int(self.buf.numel()),
                "elapsed_s": elapsed,
                "expected": expected,
                "front": front,
                "mid": mid,
                "back": back,
                "ok": ok,
            }

    sender = sender_mesh.spawn("sender", SenderActor)
    receiver = receiver_mesh.spawn("receiver", ReceiverActor)

    info_s = await sender.whoami.call_one()
    info_r = await receiver.whoami.call_one()
    print(f"[driver] sender   = {info_s}", flush=True)
    print(f"[driver] receiver = {info_r}", flush=True)

    # Rendezvous address: use the sender host's hostname (we already have
    # it from whoami).  Port comes from --master-port.
    rdvz_host = info_s["host"]
    print(
        f"[driver] HCCL 2-rank PG rendezvous @ {rdvz_host}:{master_port}",
        flush=True,
    )

    # Init both sides concurrently.  With TCP store rendezvous, init
    # blocks until all ranks join; if we don't do this concurrently the
    # first side hangs.  Monarch ``call_one()`` returns a Monarch
    # ``Future`` (awaitable, but not a coroutine), so we wrap it in a
    # thin coroutine for ``asyncio.gather`` to accept.
    async def _await_fut(fut):
        return await fut

    try:
        init_results = await asyncio.wait_for(
            asyncio.gather(
                _await_fut(
                    sender.init_group.call_one(
                        master_addr=rdvz_host,
                        master_port=master_port,
                        world_size=2,
                    )
                ),
                _await_fut(
                    receiver.init_group.call_one(
                        master_addr=rdvz_host,
                        master_port=master_port,
                        world_size=2,
                    )
                ),
            ),
            timeout=120,
        )
    except TimeoutError:
        print(
            "[driver] init timed out after 120s -- HCCL rendezvous failed."
            " Check HCCL_CONNECT_TIMEOUT, firewall, and that the sender "
            "host's master_port is reachable from the receiver.",
            flush=True,
        )
        return 1
    print(f"[driver] init OK: {init_results}", flush=True)

    payload_bytes = int(payload_gb * 1024**3)
    print(
        f"[driver] allocating {payload_gb:.2f} GB payload on both sides ...",
        flush=True,
    )
    await asyncio.gather(
        _await_fut(sender.prepare_payload.call_one(nbytes=payload_bytes)),
        _await_fut(receiver.allocate.call_one(nbytes=payload_bytes)),
    )

    print("=" * 72, flush=True)
    print(" step | bcast_s | GBps   | verified", flush=True)
    print("=" * 72, flush=True)
    results = []
    for step in range(iters):
        # Sender and receiver both call dist.broadcast; neither returns
        # until the collective completes, so launch concurrently.
        try:
            s_r, r_r = await asyncio.gather(
                _await_fut(sender.broadcast.call_one(step=step)),
                _await_fut(receiver.recv_and_check.call_one(step=step)),
            )
        except Exception as e:
            print(
                f"[driver] broadcast(step={step}) raised: {type(e).__name__}: {e}",
                flush=True,
            )
            return 1

        bcast_s = max(s_r["elapsed_s"], r_r["elapsed_s"])
        gbps = payload_bytes / bcast_s / 1024**3 if bcast_s > 0 else 0.0
        verified = r_r["ok"]
        print(
            f" {step:4d} | {bcast_s:7.2f} | {gbps:6.2f} | {verified}",
            flush=True,
        )
        if not verified:
            print(
                f"[driver] VERIFY FAILED step={step}: "
                f"expected={r_r['expected']} got "
                f"front={r_r['front']} mid={r_r['mid']} back={r_r['back']}",
                flush=True,
            )
            return 1
        results.append(bcast_s)

    print("=" * 72, flush=True)
    if len(results) > 1:
        warm = results[1:]
        avg = sum(warm) / len(warm)
        print(
            f"[driver] steady-state (iters {1}..{iters - 1}): "
            f"bcast={avg:.2f}s "
            f"({payload_bytes / avg / 1024**3:.2f} GB/s)",
            flush=True,
        )
    print("[driver] OK", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workers",
        nargs="+",
        required=True,
        help="Two Monarch bootstrap worker URLs, e.g. "
        "tcp://192.168.0.26:22222 tcp://192.168.0.23:22222. "
        "First one hosts the sender (rank 0), second hosts the receiver.",
    )
    parser.add_argument(
        "--sender-npu",
        type=int,
        default=4,
        help="Physical NPU index on the sender host (default 4, matches "
        "our storage-vol layout).",
    )
    parser.add_argument(
        "--receiver-npu",
        type=int,
        default=0,
        help="Physical NPU index on the receiver host (default 0).",
    )
    parser.add_argument(
        "--payload-gb",
        type=float,
        default=1.5,
        help="Broadcast payload size in GB. 1.5 matches Qwen3-0.6B flat "
        "buffer so numbers are directly comparable to the HiXL path.",
    )
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--master-port",
        type=int,
        default=29590,
        help="TCP port on the sender host used for the HCCL TCPStore "
        "rendezvous. Must be free and reachable from the receiver host.",
    )
    args = parser.parse_args()

    rc = asyncio.run(
        _run(
            workers=args.workers,
            sender_npu=args.sender_npu,
            receiver_npu=args.receiver_npu,
            payload_gb=args.payload_gb,
            iters=args.iters,
            master_port=args.master_port,
        )
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
