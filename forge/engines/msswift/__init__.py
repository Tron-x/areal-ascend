"""ms-swift backend adapter for Forge.

This package contains glue that lets ms-swift's GRPO trainer + rollout
server speak the framework-agnostic Forge protocols (weight sync via
``areal.weight_sync``, actor-driven launch via Monarch).  It deliberately
does **not** carry any GRPO algorithm logic: ms-swift owns its trainer,
we only adapt its boundary surfaces (rollout HTTP server, vLLM client,
worker extension class) to the shared protocols defined in ``areal/`` and
``forge/core/``.

Layout (mirror of ``forge/engines/{areal,fsdp,titan}/``):

* ``glue.py``  -- runtime monkey-patches applied before ``swift.cli.{rlhf,rollout}.main()``
                  to wire ms-swift into ``areal.weight_sync``.
"""

from forge.engines.msswift.glue import (
    install_client_patches,
    install_server_patches,
    install_worker_patches,
)

__all__ = [
    "install_client_patches",
    "install_server_patches",
    "install_worker_patches",
]
