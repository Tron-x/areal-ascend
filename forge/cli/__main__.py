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

    if sub == "sync":
        from forge.cli.sync import main as sync_main

        return sync_main(sub_argv)

    if sub == "list-rewards":
        from forge.cli.list_rewards import main as list_rewards_main

        return list_rewards_main(sub_argv)

    if sub == "list-agents":
        from forge.cli.list_agents import main as list_agents_main

        return list_agents_main(sub_argv)

    if sub == "list-workflows":
        from forge.cli.list_workflows import main as list_workflows_main

        return list_workflows_main(sub_argv)

    if sub == "list-tools":
        from forge.cli.list_tools import main as list_tools_main

        return list_tools_main(sub_argv)

    print(f"forge: unknown subcommand {sub!r}", file=sys.stderr)
    _print_usage()
    return 2


def _print_usage() -> None:
    print(
        "Usage: python -m forge <subcommand> [args...]\n"
        "\n"
        "Subcommands:\n"
        "  launch <algo-or-launcher.yaml> [options]\n"
        "       Multi-node GRPO launch.  Positional YAML is either an\n"
        "       algo YAML (with launcher_preset:) or a raw launcher YAML.\n"
        "  sync --hostfile HF [--path P ...] Push local source trees to every host\n"
        "  list-rewards                    Show registered reward functions\n"
        "  list-agents                     Show registered agent classes\n"
        "  list-workflows                  Show registered rollout workflows\n"
        "  list-tools                      Show registered tool classes\n"
        "\n"
        "For subcommand options: python -m forge <subcommand> --help",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
