"""``forge list-agents`` subcommand.

Quick diagnostic: print every agent currently registered via
``@register_agent``, grouped by the source module.  Use this to
verify a custom agent file was picked up by auto-discovery before
launching a real training job.
"""

from __future__ import annotations


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help"):
        print(
            "Usage: python -m forge list-agents\n"
            "\n"
            "Prints every agent registered via @register_agent, grouped\n"
            "by module.  Handy for verifying that custom agent files\n"
            "under forge/agents/ were picked up by auto-discovery."
        )
        return 0

    from forge.agents import _REGISTRY, ensure_loaded
    from forge.cli._list_registry import render_registry

    ensure_loaded()
    return render_registry(label="agent", registry=_REGISTRY)
