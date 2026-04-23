"""forge CLI package.

Entry point: ``python -m forge <subcommand>``.

Subcommands:

* ``launch`` — single-command multi-node launch.  Replaces the
  ``bash forge/scripts/run_multinode.sh ...`` + ``worker_manager.sh start`` +
  manual SSH flow with one Python invocation that runs pre-flight checks,
  starts workers, SSHs into the driver host to run the training entry point,
  and cleans up workers on exit (or Ctrl+C).

The CLI is deliberately thin: it delegates the heavy lifting to
``worker_manager.sh`` (which already handles SSH-based worker start/stop
correctly) and ``forge.apps.grpo`` (which does the actual training).  The
Python layer adds pre-flight checks, friendly error translation, and a
guarantee that Ctrl+C cleans up remote state.
"""

from __future__ import annotations
