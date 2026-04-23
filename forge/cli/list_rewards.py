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

    from forge.reward import _REGISTRY, ensure_loaded

    ensure_loaded()

    if not _REGISTRY:
        print("(no rewards registered)")
        return 0

    # Group by module for readability.
    by_module: dict[str, list[tuple[str, str]]] = {}
    for name, fn in sorted(_REGISTRY.items()):
        mod = getattr(fn, "__module__", "<unknown>")
        qual = getattr(fn, "__qualname__", fn.__name__)
        by_module.setdefault(mod, []).append((name, qual))

    width = max(len(name) for name in _REGISTRY)
    print(f"Found {len(_REGISTRY)} registered reward(s):\n")
    for mod in sorted(by_module):
        print(f"  {mod}")
        for name, qual in by_module[mod]:
            print(f"    • {name:<{width}}  → {qual}")
        print()

    return 0
