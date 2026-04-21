#!/usr/bin/env python3
"""End-to-end 2-node weight-sync smoke test using torchstore + Monarch RDMA (HiXL).

Uses REAL Qwen2.5-1.5B-Instruct weights (~3 GB bf16) loaded from local
safetensors cache. Spawns three actor meshes under a single driver so that
they share one ``get_or_spawn_controller("TorchStore", ...)`` lookup:

* ``trainer_mesh`` (host A, NPU 0) - loads the real safetensors into
  2 MB-aligned NPU buffers, exposes ``push_weights_torchstore(policy_version)``
  which calls ``ts.put_batch`` under ``policy_ver_{0|1}.{name}`` keys.
* ``generator_mesh`` (host B, NPU 0) - pre-allocates matching-shape
  destination buffers and exposes ``update_weights_sync(version, method, payload)``
  which calls ``ts.get(key, inplace_tensor=dst)`` into those buffers.
* ``storage_mesh`` (host B, NPU 1 [*]) - torchstore ``StorageVolume`` lives
  here, NPU-resident + 2 MB-aligned (via our ``TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE``
  patch), so HiXL RoCE can do one-sided put/get without any CPU staging.

[*] The storage volume sits on NPU 1 on host B to avoid any cross-process NPU
contention with the generator's dst buffers on NPU 0. They're still on the
same node so ts.put traffic goes cross-node over RoCE while ts.get from the
generator stays intra-node on HCCS.

Every ``iters`` iteration calls ``TorchstoreWeightSync.push(version=step)``
and prints per-step push/pull time + bandwidth. A sampled checksum is
validated on step 0 to prove the round-trip is actually delivering bytes.
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
    """Bootstrap for a 1-NPU spawned proc.

    These env vars are also set in ``worker_manager.sh`` before ``python``
    starts, but we mirror them here so the test can be run against workers
    launched differently (e.g. by hand for debugging).
    """

    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ["MONARCH_NPU_DEVICE"] = "0"
        os.environ["MONARCH_HIXL_TRANSPORT"] = "roce"
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
        # HiXL + HCCL coexistence: give HCCL a wider socket port pool so
        # HiXL's internal HcclCommPrepare + any FSDP-style HCCL process
        # group can both grab slots in the same proc (default is a single
        # port 16666 → second HcclCommPrepare returns ret=0x13; a narrow
        # 32-slot pool avoids 0x13 but still starves HiXL Connect with
        # ret=503900 once an FSDP PG is already alive on the same NPU).
        # 256 slots is the empirically validated minimum for FSDP + HiXL
        # coexistence on CANN 9.0 / Ascend 910B; see
        # ``monarch/docs/hixl_per_pair_region_retraction.md``.
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60255")
        os.environ["TORCHSTORE_MONARCH_RDMA_EAGER_D2H"] = "0"
        # LocalRankStrategy needs RANK/LOCAL_RANK; one NPU per proc so 0 is fine.
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    return _bootstrap


async def _run(workers: list[str], model_path: str, iters: int) -> int:
    import torch
    import torch_npu  # noqa: F401
    import torchstore as ts
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._src.actor.actor_mesh import configure
    from monarch._src.actor.bootstrap import attach_to_workers
    from monarch.actor import Actor, endpoint

    from forge.core.weight_sync import WeightSyncConfig, WeightSyncMethod
    from forge.engines.weight_sync import create_weight_sync
    from forge.engines.weight_sync.torchstore_sync import get_param_key

    configure(default_transport=ChannelTransport.TcpWithHostname)

    if len(workers) != 2:
        print(f"ERROR: need exactly 2 workers, got {len(workers)}", flush=True)
        return 2

    print(f"[driver] attaching to workers: {workers}", flush=True)
    hosts = attach_to_workers(ca="trust_all_connections", workers=workers)
    await hosts.initialized

    host_trainer = hosts.slice(hosts=0)  # host A
    host_generator = hosts.slice(hosts=1)  # host B

    # NPU device assignment:
    #   trainer  -> host A / NPU 0  (holds a full 3 GB state_dict)
    #   generator-> host B / NPU 0  (pre-allocated 3 GB dst buffers)
    #   storage  -> host B / NPU 1  (torchstore StorageVolume, 3 GB registered)
    # Keeping generator and storage on different NPUs lets the intra-node path
    # go through HCCS without any single NPU holding 6 GB of HiXL-registered
    # memory.
    trainer_mesh = host_trainer.spawn_procs(
        per_host={"npus": 1}, bootstrap=_npu_bootstrap(0)
    )
    generator_mesh = host_generator.spawn_procs(
        per_host={"npus": 1}, bootstrap=_npu_bootstrap(0)
    )
    storage_mesh = host_generator.spawn_procs(
        per_host={"npus": 1},
        bootstrap=_npu_bootstrap(int(os.environ.get("TEST_STORAGE_NPU", "1"))),
    )

    # HiXL on CANN 9.0 limits the number of registered memory regions per
    # engine pair: creating a second RDMABuffer on the same pair makes
    # TransferSync return 503900 for the whole pair.  A 338-tensor Qwen
    # state_dict blows past the limit both for ts.put_batch and for a
    # serialized ts.put loop (Python GC of the first RDMABuffer is not
    # deterministic enough).  The documented workaround
    # (monarch/docs/npu_install_guide.md section "同 engine pair 只能有一
    # 个已注册 region") is "single large buffer + offset slicing".  We do
    # exactly that: pack the full state_dict into one flat, 2 MB-aligned
    # NPU buffer on the trainer, transfer it in a single ts.put, and on
    # the generator side receive into an identically-shaped flat buffer.
    # Per-parameter access is still available as views over the flat
    # buffer, so checksums / future vLLM load_weights still work unchanged.

    _ALIGN = 16  # byte-align every param start (safe for fp32, bf16, int*).
    # HiXL RoCE requires the buffer *size* (not just address) to be a
    # multiple of 2 MB, otherwise TransferSync returns 503900 on the
    # final partial block.  We therefore pad the flat buffer total
    # length up to the next 2 MB boundary.
    _HIXL_BLOCK = 2 * 1024 * 1024
    # HiXL read_into / write_from fails (503900) when the single transfer
    # size exceeds ~2 GB (Rust-side signed int32 length).  We therefore
    # split the flat buffer into sub-GB chunks and do them serially; each
    # chunk uses a single RDMABuffer, which also lets us cleanly release
    # the old chunk's registration before setting up the next one and
    # avoids any chance of an overlapping hixl_register_mem (103900).
    _HIXL_CHUNK = 1 * 1024 * 1024 * 1024  # 1 GiB, must be 2 MB-aligned.

    def _chunk_offsets(
        total_bytes: int, chunk: int = _HIXL_CHUNK
    ) -> list[tuple[int, int]]:
        """Return [(offset, size), ...] chunks covering ``total_bytes``.

        Every ``size`` is a multiple of 2 MB except possibly the last
        chunk, and the caller must ensure ``total_bytes`` itself is
        already 2 MB-aligned.
        """
        assert chunk % _HIXL_BLOCK == 0, "chunk must be 2 MB-aligned"
        out = []
        off = 0
        while off < total_bytes:
            sz = min(chunk, total_bytes - off)
            out.append((off, sz))
            off += sz
        return out

    def _plan_layout(meta: list) -> tuple[list, int]:
        """Assign an offset to every param, return (plan, total_bytes).

        ``meta`` is a list of ``(name, shape, dtype_str, nbytes)`` tuples;
        returns ``[(name, shape, dtype_str, offset, nbytes), ...]`` and
        the 2 MB-padded total size so the underlying flat NPU buffer can
        be allocated with an RDMA-friendly length.
        """
        plan = []
        offset = 0
        for name, shape, dtype_str, nbytes in meta:
            if offset % _ALIGN != 0:
                offset += _ALIGN - (offset % _ALIGN)
            plan.append((name, shape, dtype_str, offset, nbytes))
            offset += nbytes
        if offset % _HIXL_BLOCK != 0:
            offset += _HIXL_BLOCK - (offset % _HIXL_BLOCK)
        return plan, offset

    _DTYPE_MAP = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.uint8": torch.uint8,
        "torch.int8": torch.int8,
    }

    async def _do_hccl_prewarm(
        master_addr: str, master_port: int, rank: int, world_size: int
    ) -> dict:
        """Init a cross-host HCCL PG and run one all_reduce to force
        communicator creation.  Reproduces the 'FSDP already alive' side
        of the HiXL+HCCL coexistence test.  Any 0x13 / ret=0x13 here
        would mean HCCL itself is broken before HiXL even runs.
        """
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)

        if not dist.is_initialized():
            dist.init_process_group(
                backend="hccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank,
                world_size=world_size,
                timeout=__import__("datetime").timedelta(seconds=120),
            )

        probe = torch.ones(8, dtype=torch.float32, device="npu:0") * (rank + 1)
        dist.all_reduce(probe)
        torch.npu.synchronize()
        return {
            "rank": rank,
            "world_size": world_size,
            "probe_sum": float(probe.sum().item()),
        }

    class TrainerActor(Actor):
        """Loads Qwen state_dict and pushes it to torchstore per step."""

        def __init__(self) -> None:
            self.flat: torch.Tensor | None = None
            self.views: dict[str, torch.Tensor] = {}
            self.plan: list = []
            self.total_bytes = 0

        @endpoint
        async def whoami(self) -> dict:
            return {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            }

        @endpoint
        async def hccl_prewarm(
            self, master_addr: str, master_port: int, rank: int, world_size: int
        ) -> dict:
            return await _do_hccl_prewarm(master_addr, master_port, rank, world_size)

        @endpoint
        async def load_safetensors(self, path: str) -> list:
            """Load the safetensors file into a single flat NPU buffer.

            Returns the layout plan ``[(name, shape, dtype_str, offset,
            nbytes), ...]`` so the generator can pre-allocate and slice
            an identical buffer.
            """
            from monarch._src.rdma.xdma import alloc_aligned_tensor
            from safetensors import safe_open

            meta = []
            with safe_open(path, framework="pt", device="cpu") as f:
                names = list(f.keys())
                for name in names:
                    slc = f.get_slice(name)
                    shape = tuple(slc.get_shape())
                    dtype = str(slc.get_dtype()).upper()
                    torch_dtype_str = (
                        "torch.bfloat16"
                        if dtype == "BF16"
                        else f"torch.{dtype.lower()}"
                    )
                    td = _DTYPE_MAP[torch_dtype_str]
                    elsize = torch.empty((), dtype=td).element_size()
                    nbytes = elsize
                    for d in shape:
                        nbytes *= d
                    meta.append((name, list(shape), torch_dtype_str, nbytes))

            plan, total_bytes = _plan_layout(meta)
            # Single 2MB-aligned NPU buffer (uint8 flat) sized to hold all params.
            flat, _raw = alloc_aligned_tensor(
                (total_bytes,), dtype=torch.uint8, device="npu:0"
            )
            flat.zero_()

            # Second pass: copy each param into its slot as a typed view.
            views: dict[str, torch.Tensor] = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for name, shape, dtype_str, offset, nbytes in plan:
                    td = _DTYPE_MAP[dtype_str]
                    view = flat[offset : offset + nbytes].view(td).view(shape)
                    cpu_t = f.get_tensor(name)
                    view.copy_(cpu_t)
                    views[name] = view
                    del cpu_t
            torch.npu.synchronize()

            self.flat = flat
            self.views = views
            self.plan = plan
            self.total_bytes = total_bytes
            return plan

        @endpoint
        async def push_weights_torchstore(self, policy_version: int) -> dict:
            t0 = time.perf_counter()
            # Split the 3 GB flat buffer into ≤1 GiB chunks and put them
            # sequentially.  Each ts.put uses a single RDMABuffer that
            # is released before the next one -- keeps memory footprint
            # predictable and every single HiXL transfer stays well
            # under the 2 GB single-buffer (int32) limit.
            chunks = _chunk_offsets(self.total_bytes)
            for idx, (off, sz) in enumerate(chunks):
                key = get_param_key(policy_version, f"state_dict_flat.chunk{idx}")
                await ts.put(key, self.flat[off : off + sz])
            put_s = time.perf_counter() - t0
            return {
                "num_keys": len(self.plan),
                "bytes": self.total_bytes,
                "num_chunks": len(chunks),
                "build_state_dict_s": 0.0,
                "put_batch_s": put_s,
            }

        @endpoint
        async def checksum_of(self, name: str) -> float:
            t = self.views[name]
            return float(t.float().sum().cpu().item())

    class GeneratorActor(Actor):
        """Receives weights from torchstore into a flat NPU buffer."""

        def __init__(self) -> None:
            self.flat: torch.Tensor | None = None
            self.views: dict[str, torch.Tensor] = {}
            self.plan: list = []
            self.total_bytes = 0

        @endpoint
        async def whoami(self) -> dict:
            return {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            }

        @endpoint
        async def hccl_prewarm(
            self, master_addr: str, master_port: int, rank: int, world_size: int
        ) -> dict:
            return await _do_hccl_prewarm(master_addr, master_port, rank, world_size)

        @endpoint
        async def allocate_dst(self, plan: list) -> dict:
            from monarch._src.rdma.xdma import alloc_aligned_tensor

            total_bytes = 0
            if plan:
                total_bytes = plan[-1][3] + plan[-1][4]
            # Must match the trainer's 2 MB-padded flat buffer size so
            # torchstore's shape assertion passes and HiXL's TransferSync
            # sees an aligned length on both ends.
            if total_bytes % _HIXL_BLOCK != 0:
                total_bytes += _HIXL_BLOCK - (total_bytes % _HIXL_BLOCK)
            flat, _raw = alloc_aligned_tensor(
                (total_bytes,), dtype=torch.uint8, device="npu:0"
            )
            flat.zero_()

            views: dict[str, torch.Tensor] = {}
            for name, shape, dtype_str, offset, nbytes in plan:
                td = _DTYPE_MAP[dtype_str]
                views[name] = flat[offset : offset + nbytes].view(td).view(tuple(shape))
            torch.npu.synchronize()

            self.flat = flat
            self.views = views
            self.plan = plan
            self.total_bytes = total_bytes
            return {"num_keys": len(views), "bytes": total_bytes}

        @endpoint
        async def update_weights_sync(
            self, version: int, method: str, payload: dict
        ) -> dict:
            if method != "torchstore":
                return {
                    "success": False,
                    "message": f"unsupported method: {method}",
                }
            t0 = time.perf_counter()
            chunks = _chunk_offsets(self.total_bytes)
            for idx, (off, sz) in enumerate(chunks):
                key = get_param_key(version, f"state_dict_flat.chunk{idx}")
                # in-place get into the matching slice of the flat buffer.
                await ts.get(key, inplace_tensor=self.flat[off : off + sz])
            torch.npu.synchronize()
            pull_s = time.perf_counter() - t0
            return {
                "success": True,
                "version": version,
                "bytes": self.total_bytes,
                "pull_s": pull_s,
            }

        @endpoint
        async def checksum_of(self, name: str) -> float:
            t = self.views[name]
            return float(t.float().sum().cpu().item())

    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    generator = generator_mesh.spawn("generator", GeneratorActor)

    info_t = await trainer.whoami.call_one()
    info_g = await generator.whoami.call_one()
    print(f"[driver] trainer   = {info_t}", flush=True)
    print(f"[driver] generator = {info_g}", flush=True)

    # ---- HiXL + HCCL coexistence pre-warm ---------------------------------
    # Reproduce the exact scenario that used to fail with
    # ``HcclCommPrepare ret=0x13``: an HCCL process group is already live
    # in the same proc that HiXL later tries to ``Connect`` from.  Previously
    # FSDP inside the trainer would init its own HCCL PG on the default
    # single-slot port (16666) and HiXL's Connect handshake would race it
    # and lose.  With ``HCCL_NPU_SOCKET_PORT_RANGE=60000-60031`` exported in
    # every bootstrap, HCCL has 32 slots and HiXL gets a free one.
    #
    # This pre-warm step stands in for "FSDP is alive": it pins a 2-rank
    # HCCL process group across the two trainer/generator hosts, runs one
    # ``all_reduce`` to force communicator creation (HCCL is lazy), and then
    # leaves the group in place while we do all torchstore operations.
    if os.environ.get("TEST_HCCL_PREWARM", "0") == "1":
        rdvz_host = info_t["host"]
        rdvz_port = int(os.environ.get("TEST_HCCL_PREWARM_PORT", "29501"))
        print(
            f"[driver] Pre-warming cross-node HCCL PG @ {rdvz_host}:{rdvz_port} "
            f"(simulates FSDP-HCCL + HiXL coexistence)...",
            flush=True,
        )
        t0 = time.time()
        await asyncio.gather(
            trainer.hccl_prewarm.call_one(
                master_addr=rdvz_host, master_port=rdvz_port, rank=0, world_size=2
            ),
            generator.hccl_prewarm.call_one(
                master_addr=rdvz_host, master_port=rdvz_port, rank=1, world_size=2
            ),
        )
        print(
            f"[driver] HCCL PG alive in both actors ({time.time() - t0:.1f}s)",
            flush=True,
        )

    print(f"[driver] Loading Qwen safetensors on trainer: {model_path}", flush=True)
    t0 = time.time()
    plan = await trainer.load_safetensors.call_one(model_path)
    num_keys = len(plan)
    total_bytes = plan[-1][3] + plan[-1][4] if plan else 0
    if total_bytes % _HIXL_BLOCK != 0:
        total_bytes += _HIXL_BLOCK - (total_bytes % _HIXL_BLOCK)
    print(
        f"[driver] Packed {num_keys} tensors into one {total_bytes / 1e9:.2f} GB "
        f"flat NPU buffer in {time.time() - t0:.1f}s",
        flush=True,
    )

    print("[driver] Pre-allocating generator flat buffer...", flush=True)
    t0 = time.time()
    alloc = await generator.allocate_dst.call_one(plan)
    print(
        f"[driver] Pre-allocated {alloc['bytes'] / 1e9:.2f} GB dst flat buffer "
        f"({alloc['num_keys']} views) in {time.time() - t0:.1f}s",
        flush=True,
    )

    print("[driver] Initializing TorchstoreWeightSync...", flush=True)
    strategy = create_weight_sync("torchstore")
    cfg = WeightSyncConfig(
        method=WeightSyncMethod.TORCHSTORE,
        extra={"storage_mesh": storage_mesh},
    )
    await strategy.initialize(trainer, generator, cfg)
    print("[driver] TorchstoreWeightSync ready", flush=True)

    # Pick a representative key for correctness validation (something big).
    sample_name = max(
        (
            p
            for p in plan
            if p[2] in ("torch.bfloat16", "torch.float16", "torch.float32")
        ),
        key=lambda p: p[4],
    )[0]
    expected_sum = await trainer.checksum_of.call_one(sample_name)
    print(
        f"[driver] Sample key for checksum: {sample_name!r} "
        f"(src sum={expected_sum:.3f})",
        flush=True,
    )

    print("=" * 72, flush=True)
    print(
        " step |  push_s |  pull_s | total_s | push_GBps | pull_GBps | num_keys",
        flush=True,
    )
    print("=" * 72, flush=True)

    results = []
    for step in range(iters):
        r = await strategy.push(version=step)
        results.append(r)
        print(
            f" {step:4d} | {r['push_s']:7.2f} | {r['pull_s']:7.2f} | "
            f"{r['total_s']:7.2f} | {r['push_gbps']:9.2f} | "
            f"{r['pull_gbps']:9.2f} | {r['num_keys']:8d}",
            flush=True,
        )

        if step == 0:
            got_sum = await generator.checksum_of.call_one(sample_name)
            rel_err = abs(got_sum - expected_sum) / max(abs(expected_sum), 1e-12)
            if rel_err > 1e-3:
                print(
                    f"[driver] FAIL: checksum mismatch for {sample_name!r}: "
                    f"got={got_sum} expected={expected_sum} "
                    f"rel_err={rel_err:.2e}",
                    flush=True,
                )
                await strategy.shutdown()
                return 1
            print(
                f"[driver] OK checksum for {sample_name!r}: "
                f"got={got_sum:.3f} expected={expected_sum:.3f} "
                f"rel_err={rel_err:.2e}",
                flush=True,
            )

    print("=" * 72, flush=True)
    # Skip the first step for steady-state averages (first put+get primes
    # HiXL registration caches; subsequent steps are ping-pong slot reuse).
    warm = results[1:] if len(results) > 1 else results
    avg_push = sum(r["push_s"] for r in warm) / len(warm)
    avg_pull = sum(r["pull_s"] for r in warm) / len(warm)
    avg_total = sum(r["total_s"] for r in warm) / len(warm)
    avg_push_bw = total_bytes / avg_push / (1024**3) if avg_push > 0 else 0.0
    avg_pull_bw = total_bytes / avg_pull / (1024**3) if avg_pull > 0 else 0.0
    print(
        f"[driver] Steady-state ({len(warm)} steps, warmup skipped): "
        f"push={avg_push:.2f}s ({avg_push_bw:.2f} GB/s), "
        f"pull={avg_pull:.2f}s ({avg_pull_bw:.2f} GB/s), "
        f"total={avg_total:.2f}s",
        flush=True,
    )

    await strategy.shutdown()
    print("[driver] OK", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workers",
        required=True,
        help="Comma-separated Monarch worker addresses, e.g. "
        "tcp://192.168.0.26:22222,tcp://192.168.0.23:22222",
    )
    parser.add_argument(
        "--model-path",
        default="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct/model.safetensors",
        help="Path to the .safetensors file (must exist on the trainer host).",
    )
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    return asyncio.run(_run(workers, args.model_path, args.iters))


if __name__ == "__main__":
    sys.exit(main())
