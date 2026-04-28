"""``forge list-tools`` subcommand.

Quick diagnostic: print every tool currently registered via
``@register_tool``, grouped by the source module.  Use this to verify
a custom tool file was picked up by auto-discovery before launching a
real training job.
"""

from __future__ import annotations


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help"):
        print(
            "Usage: python -m forge list-tools\n"
            "\n"
            "Prints every tool registered via @register_tool, grouped\n"
            "by module.  Handy for verifying that custom tool files\n"
            "under forge/tools/ were picked up by auto-discovery."
        )
        return 0

    from forge.cli._list_registry import render_registry
    from forge.tools import _REGISTRY, ensure_loaded

    ensure_loaded()
    return render_registry(label="tool", registry=_REGISTRY)
