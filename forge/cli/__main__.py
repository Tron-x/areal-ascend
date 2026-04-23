"""Entry point for ``python -m forge`` (dispatches to subcommands).

Supported subcommands:

* ``launch`` — full multi-node launch; see :mod:`forge.cli.launch`.

More subcommands (``stop``, ``diagnose``, ``status``) can be added by
mirroring the ``launch`` module's structure.  Dispatch is kept as a
plain ``dict`` so adding a subcommand is a one-line change.
"""

from __future__ import annotations

import sys


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_usage()
        return 0 if argv else 2

    sub = argv[0]
    sub_argv = argv[1:]

    if sub == "launch":
        from forge.cli.launch import main as launch_main

        return launch_main(sub_argv)

    if sub == "list-rewards":
        from forge.cli.list_rewards import main as list_rewards_main

        return list_rewards_main(sub_argv)

    print(f"forge: unknown subcommand {sub!r}", file=sys.stderr)
    _print_usage()
    return 2


def _print_usage() -> None:
    print(
        "Usage: python -m forge <subcommand> [args...]\n"
        "\n"
        "Subcommands:\n"
        "  launch <config.yaml> [options]  Multi-node GRPO launch\n"
        "  list-rewards                    Show registered reward functions\n"
        "\n"
        "For subcommand options: python -m forge <subcommand> --help",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
