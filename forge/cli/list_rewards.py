"""``forge list-rewards`` subcommand.

Quick diagnostic: print every reward currently registered in
``forge.reward``, including the module they came from so users can see
at a glance which files contributed entries.  Saves a trip into a
Python REPL just to run ``available_rewards()``.
"""

from __future__ import annotations


def main(argv: list[str]) -> int:
    # Accept --help purely so users get a nicer signal than argparse
    # noise; there are no knobs here.
    if argv and argv[0] in ("-h", "--help"):
        print(
            "Usage: python -m forge list-rewards\n"
            "\n"
            "Prints every reward registered via @register_reward, grouped\n"
            "by module, so you can verify your custom reward got picked\n"
            "up before launching a real training job."
        )
        return 0

    from forge.cli._list_registry import render_registry
    from forge.reward import _REGISTRY, ensure_loaded

    ensure_loaded()
    return render_registry(label="reward", registry=_REGISTRY)
