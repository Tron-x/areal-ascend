"""Backward-compat shim.

The real router implementation now lives in
:mod:`areal.weight_sync.vllm_ext.server_router`.  This module re-exports
the ``router``, request models, and the ``__main__`` entrypoint so legacy
invocations such as

    python -m areal.engine.vllm_ext.areal_vllm_server <vllm-cli-args>

(emitted by ``areal.api.cli_args.VLLMConfig.cmd``) keep working unchanged.

New code should import or run from ``areal.weight_sync.vllm_ext.server_router``.
"""

from areal.weight_sync.vllm_ext.server_router import (
    UpdateGroupRequest,
    UpdateWeightsFromXcclRequest,
    UpdateWeightsFromXcclRequestLora,
    UpdateWeightsRequest,
    UpdateWeightsRequestLora,
    build_response,
    router,
    to_json_response,
)

__all__ = [
    "UpdateGroupRequest",
    "UpdateWeightsFromXcclRequest",
    "UpdateWeightsFromXcclRequestLora",
    "UpdateWeightsRequest",
    "UpdateWeightsRequestLora",
    "build_response",
    "router",
    "to_json_response",
]


if __name__ == "__main__":
    # Delegate to the real entrypoint.  We invoke the new module via
    # ``runpy`` instead of importing it because the original ``__main__``
    # block does ``cli_env_setup()`` + ``argparse`` work that should run
    # under the ``__main__`` namespace.
    import runpy
    import sys

    runpy.run_module(
        "areal.weight_sync.vllm_ext.server_router",
        run_name="__main__",
        alter_sys=True,
    )
    sys.exit(0)
