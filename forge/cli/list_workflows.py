"""``forge list-workflows`` subcommand.

Quick diagnostic: print every workflow currently registered via
``@register_workflow``.  Warm-up is slower than ``list-agents`` or
``list-rewards`` because workflow modules import torch / transformers
during auto-discovery -- that's the honest cost of pulling in the
real classes rather than parsing decorators from source.
"""

from __future__ import annotations


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help"):
        print(
            "Usage: python -m forge list-workflows\n"
            "\n"
            "Prints every workflow registered via @register_workflow,\n"
            "grouped by module.  Handy for verifying that custom\n"
            "workflows under areal/workflow/ were picked up by\n"
            "auto-discovery."
        )
        return 0

    from forge.cli._list_registry import render_registry

    from areal.workflow import _REGISTRY, ensure_loaded

    ensure_loaded()
    return render_registry(label="workflow", registry=_REGISTRY)
