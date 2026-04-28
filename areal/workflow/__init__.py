"""AReaL rollout-workflow registry + lazy exports.

Algorithm-engineer-facing contract (mirrors :mod:`forge.reward` and
:mod:`forge.agents`): subclass :class:`areal.api.RolloutWorkflow`,
slap ``@register_workflow("my_name")`` on it, and reference it in
YAML as ``workflow: my_name``.

    # my_custom_workflow.py
    from areal.api import RolloutWorkflow
    from areal.workflow import register_workflow

    @register_workflow("my_first_workflow")
    class MyFirstWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            ...

    # training YAML
    workflow: my_first_workflow

Design notes:

* What the registry stores is the **class**.  The caller
  (typically :mod:`forge.apps.grpo` or :mod:`forge.apps.agent_rl`)
  constructs it with the appropriate engine / tokenizer / reward
  dependencies.

* Auto-discovery imports every non-private submodule of
  ``areal.workflow`` at :func:`ensure_loaded` time -- same pattern as
  :mod:`forge.reward`.  Heavy integration modules
  (``openai_agent``, ``anthropic``, ``langchain``) are imported too
  so their decorators fire, but failures are silently skipped so a
  missing optional dep never breaks the registry lookup.

* This module preserves the pre-existing lazy-import mechanism for
  ``RLVRWorkflow`` / ``MultiTurnWorkflow`` / ``VisionRLVRWorkflow``
  so direct imports (``from areal.workflow import RLVRWorkflow``)
  still behave the same and still avoid pulling in Torch at module
  load.

Not in scope for this Phase-A1 MVP:

* Protocol normalization beyond ``RolloutWorkflow``.
* Workflow factory / config binding -- callers still construct the
  class themselves.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("WorkflowRegistry")

# Name -> workflow class (or factory).
_REGISTRY: dict[str, Callable[..., Any]] = {}

# Lazy-init guard.
_AUTO_DISCOVER_DONE = False


def register_workflow(name: str):
    """Decorator: register ``cls`` under ``name`` in the workflow registry.

    Duplicates raise ``ValueError`` at decoration time.
    """

    def wrap(cls: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"workflow {name!r} is already registered to "
                f"{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}; "
                f"refusing to overwrite with "
                f"{cls.__module__}.{cls.__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_workflow(name: str) -> Callable[..., Any]:
    """Lookup a registered workflow by short name.

    Raises ``ValueError`` with placement hints when the short name is
    unknown -- same style as :func:`forge.reward.get_reward`.
    """
    ensure_loaded()
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"unknown workflow {name!r}\n"
            f"\n"
            f"Available workflows: {available}\n"
            f"\n"
            f"If you wrote `@register_workflow({name!r})` but it's not\n"
            f"listed above, the decorator's module wasn't imported.\n"
            f"Pick ONE of:\n"
            f"\n"
            f"  (a) Auto-discovery (recommended).  Place the file at:\n"
            f"        areal/workflow/<anything>.py\n"
            f"      Auto-discovery scans that package at lookup time,\n"
            f"      so a plain file copy is enough.\n"
            f"\n"
            f"  (b) Explicit import from your launch entry script.\n"
            f"      Add before training starts:\n"
            f"        import my_workflow_module  # noqa: F401\n"
            f"\n"
            f"Run `python -m forge list-workflows` to verify registration."
        )
    return _REGISTRY[name]


def available_workflows() -> list[str]:
    """Return the sorted list of registered workflow short names."""
    ensure_loaded()
    return sorted(_REGISTRY)


def ensure_loaded() -> None:
    """Import all ``areal.workflow`` submodules so ``@register_workflow``
    decorators fire.

    Idempotent; safe to call from any hot path.  Sub-module import
    failures are logged at DEBUG and skipped -- a missing optional
    dependency (``anthropic``, ``openai``, ...) never breaks
    registry lookup for the workflows that ARE available.
    """
    global _AUTO_DISCOVER_DONE
    if _AUTO_DISCOVER_DONE:
        return
    _AUTO_DISCOVER_DONE = True  # set early to prevent re-entry on errors

    try:
        pkg = importlib.import_module("areal.workflow")
    except Exception as e:  # pragma: no cover -- defensive
        logger.debug("ensure_loaded: cannot import areal.workflow (%s); skipping", e)
        return
    if not hasattr(pkg, "__path__"):
        return
    for modinfo in pkgutil.iter_modules(pkg.__path__):
        if modinfo.name.startswith("_"):
            continue
        full_name = f"areal.workflow.{modinfo.name}"
        try:
            importlib.import_module(full_name)
        except Exception as e:
            logger.debug("ensure_loaded: cannot import %s (%s); skipping", full_name, e)


def reset_for_tests() -> None:
    """Test-only: clear the registry + re-run discovery on next call."""
    global _AUTO_DISCOVER_DONE
    _REGISTRY.clear()
    _AUTO_DISCOVER_DONE = False


# ---------------------------------------------------------------------------
# Legacy lazy-import surface.  Preserves ``from areal.workflow import
# RLVRWorkflow`` behaviour without pulling torch in on bare imports of the
# registry helpers above.
# ---------------------------------------------------------------------------

__all__ = [
    "register_workflow",
    "get_workflow",
    "available_workflows",
    "ensure_loaded",
    "reset_for_tests",
    "RLVRWorkflow",
    "MultiTurnWorkflow",
    "VisionRLVRWorkflow",
]

_LAZY_IMPORTS = {
    "RLVRWorkflow": "areal.workflow.rlvr",
    "MultiTurnWorkflow": "areal.workflow.multi_turn",
    "VisionRLVRWorkflow": "areal.workflow.vision_rlvr",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
