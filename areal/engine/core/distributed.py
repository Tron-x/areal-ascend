"""Backward-compat shim.

The real implementation now lives in :mod:`areal.weight_sync.distributed`.
This module re-exports the original symbols so existing imports such as

    from areal.engine.core.distributed import init_custom_process_group

keep working unchanged.  New code should import directly from
``areal.weight_sync``.
"""

from areal.weight_sync.distributed import (
    init_custom_process_group,
    patch_dist_group_timeout,
)

__all__ = ["init_custom_process_group", "patch_dist_group_timeout"]
