"""Shared helpers for the "pack a full state_dict into one 2MB-aligned NPU
flat buffer + chunked ``ts.put`` / ``ts.get``" fast-path used by
``TorchstoreWeightSync``.

Why flat-buffer is dramatically faster than ``ts.put_batch(per-param dict)``
on CANN/HiXL
------------------------------------------------------------------------

``torchstore.MonarchRDMATransportBuffer.allocate`` has a staging-pool
safety net: for every tensor that is *not* already inside the caller's
pre-registered RDMA pool, it allocates a fresh pool slot, memcpy's the
source tensor in, and wraps that slot in a new ``RDMABuffer`` (which on
Ascend triggers a fresh HiXL ``register_mem``).  For a full FSDP
state_dict that is 311 small per-parameter tensors, this fires 311
intra-proc copies + 311 HiXL registrations -- dominated by per-call
overhead rather than NIC bandwidth, delivering ~500 MB/s on our
2-machine RoCE setup.

If instead the trainer packs the gathered state_dict into a single
2 MB-aligned NPU buffer (via
``monarch._src.rdma.xdma.alloc_aligned_tensor``), ``RDMABuffer`` can wrap
that buffer directly -- no staging-pool copy, no re-registration -- and
torchstore's subsequent ``ts.put(flat_slice)`` each run as a single
HiXL ``TransferSync`` of up to 1 GiB.  On the same hardware our
standalone ``test_weight_sync_2node.py`` measures this at ~17 GB/s
steady-state.  This helper exists so the ``TrainerActor`` /
``Generator`` actors can adopt the same layout/chunking without
duplicating the math.

Layout/chunking invariants
--------------------------

* Each parameter is placed at an offset that is a multiple of
  ``ALIGN`` (16 bytes, safe for fp32 / bf16 / int* element alignment).
* The total flat buffer size is padded up to a multiple of
  ``HIXL_BLOCK`` (2 MiB), which is HiXL's minimum granularity for HCCS
  transfers and the required RDMA-buffer size alignment in RoCE mode.
* Every put/get spans at most ``HIXL_CHUNK`` (1 GiB) bytes.  HiXL's
  ``TransferSync`` uses a signed int32 length internally, so single
  transfers that hit or exceed 2 GiB bounce with ``ret=503900``.  1 GiB
  leaves comfortable headroom and also keeps the per-put RDMABuffer
  registration short-lived enough that CANN's in-flight registration
  state cannot overlap (which used to hit ``ret=103900`` on register).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


ALIGN: int = 16
HIXL_BLOCK: int = 2 * 1024 * 1024  # 2 MiB
HIXL_CHUNK: int = 1 * 1024 * 1024 * 1024  # 1 GiB


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def plan_layout(
    meta: Sequence[tuple[str, Sequence[int], str, int]],
) -> tuple[list[tuple[str, tuple[int, ...], str, int, int]], int]:
    """Assign (offset, size) to every parameter.

    Args:
        meta: sequence of ``(name, shape, dtype_str, nbytes)`` tuples, in
            push order.  ``shape`` is any iterable of ints (typically a
            ``torch.Size`` or a ``list[int]``).  ``dtype_str`` is the
            torch dtype repr with the leading ``torch.`` dropped
            (e.g. ``"bfloat16"``, ``"float32"``).

    Returns:
        ``(plan, total_bytes)`` where:

        * ``plan`` -- list of
          ``(name, shape_tuple, dtype_str, offset, nbytes)`` tuples,
          same ordering as ``meta``, with offsets 16 B-aligned.
        * ``total_bytes`` -- 2 MiB-padded buffer size required to hold
          the packed layout (this is the size the caller must allocate
          on both the source and destination side).
    """
    plan: list[tuple[str, tuple[int, ...], str, int, int]] = []
    offset = 0
    for name, shape, dtype_str, nbytes in meta:
        offset = _round_up(offset, ALIGN)
        plan.append(
            (name, tuple(int(d) for d in shape), dtype_str, offset, int(nbytes))
        )
        offset += int(nbytes)
    total_bytes = _round_up(offset, HIXL_BLOCK)
    return plan, total_bytes


def chunk_offsets(total_bytes: int, chunk: int = HIXL_CHUNK) -> list[tuple[int, int]]:
    """Return ``[(offset, size), ...]`` covering ``[0, total_bytes)``.

    Every returned ``size`` is a multiple of ``HIXL_BLOCK`` (2 MiB)
    except possibly the last chunk -- but since ``total_bytes`` itself
    is expected to be 2 MiB-padded (see :func:`plan_layout`), the last
    chunk is also 2 MiB-aligned in practice.
    """
    if chunk <= 0 or chunk % HIXL_BLOCK != 0:
        raise ValueError(f"chunk must be positive and a multiple of 2 MiB; got {chunk}")
    out: list[tuple[int, int]] = []
    off = 0
    while off < total_bytes:
        sz = min(chunk, total_bytes - off)
        out.append((off, sz))
        off += sz
    return out


def build_meta_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> list[tuple[str, tuple[int, ...], str, int]]:
    """Extract ``(name, shape, dtype_str, nbytes)`` metadata from a state_dict.

    Preserves insertion order of ``state_dict``.
    """
    meta: list[tuple[str, tuple[int, ...], str, int]] = []
    for name, tensor in state_dict.items():
        shape = tuple(int(d) for d in tensor.shape)
        nbytes = tensor.numel() * tensor.element_size()
        dtype_str = str(tensor.dtype).replace("torch.", "")
        meta.append((name, shape, dtype_str, nbytes))
    return meta


_DTYPE_MAP: dict[str, str] = {
    # Names torchstore/test_weight_sync_2node accept.  Keyed by the
    # string we produce in ``build_meta_from_state_dict`` (i.e. after
    # stripping "torch." from ``str(dtype)``).
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float64": "float64",
    "int64": "int64",
    "int32": "int32",
    "int16": "int16",
    "int8": "int8",
    "uint8": "uint8",
    "bool": "bool",
}


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    """Resolve ``"bfloat16"`` / ``"float32"`` / ... to a ``torch.dtype``."""
    import torch

    canonical = _DTYPE_MAP.get(dtype_str, dtype_str)
    return getattr(torch, canonical)
