"""Shared renderer for ``forge list-{rewards,agents,workflows}``.

All three subcommands are structurally identical: walk a registry
dict, group by source module, print as indented bullets.  Keeping the
formatting here means a future tweak (colors, JSON mode) lands in one
place rather than three.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def render_registry(
    *,
    label: str,
    registry: Mapping[str, Callable[..., Any]],
    out=None,
) -> int:
    """Print ``registry`` as a grouped-by-module table.

    Args:
        label: Singular noun for the registry (``"reward"``,
            ``"agent"``, ``"workflow"``).  Used in the header.
        registry: The underlying name -> callable mapping.
        out: File-like object; defaults to ``sys.stdout`` when
            ``None``.

    Returns exit code ``0``.
    """
    import sys

    if out is None:
        out = sys.stdout

    if not registry:
        print(f"(no {label}s registered)", file=out)
        return 0

    by_module: dict[str, list[tuple[str, str]]] = {}
    for name, obj in sorted(registry.items()):
        mod = getattr(obj, "__module__", "<unknown>")
        qual = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        by_module.setdefault(mod, []).append((name, qual))

    width = max(len(name) for name in registry)
    print(f"Found {len(registry)} registered {label}(s):\n", file=out)
    for mod in sorted(by_module):
        print(f"  {mod}", file=out)
        for name, qual in by_module[mod]:
            print(f"    \u2022 {name:<{width}}  \u2192 {qual}", file=out)
        print("", file=out)

    return 0
