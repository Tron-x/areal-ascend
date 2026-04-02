"""Thread-local hooks for PPOTrainer extensibility.

Allows external orchestrators (e.g. Monarch TrainerActor) to inject custom
behaviour into PPOTrainer without monkey-patching class methods.

Usage (orchestrator side)::

    from areal.utils.trainer_hooks import set_hooks, clear_hooks

    set_hooks(
        rollout_engine_factory=my_factory,
        weight_update_alloc_mode_override=alloc_mode,
        suppress_context_exit=True,
    )
    try:
        # PPOTrainer.__init__ reads hooks automatically
        ...
    finally:
        clear_hooks()

The three supported hooks:

``rollout_engine_factory``
    ``Callable[[InferenceEngineConfig, bool, str | None, AllocationMode],
    InferenceEngine | None]``.
    When set, ``PPOTrainer._init_rollout`` delegates to this factory instead
    of building the default ``RemotevLLMEngine`` / ``RemoteSGLangEngine``.
    If the factory returns ``None`` the default path is used.

``weight_update_alloc_mode_override``
    ``AllocationMode | None``.
    When set, overrides ``self.weight_update_meta.alloc_mode`` *before*
    ``self.actor.connect_engine()`` is called.  Used to inject the correct
    XCCL process-group layout for Monarch.

``suppress_context_exit``
    ``bool``.  When ``True``, ``PPOTrainer.__exit__`` becomes a no-op so
    the orchestrator can keep the trainer alive past the ``with`` block.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_hooks = threading.local()

_HOOK_NAMES = (
    "rollout_engine_factory",
    "weight_update_alloc_mode_override",
    "suppress_context_exit",
)


def set_hooks(
    *,
    rollout_engine_factory: Callable | None = None,
    weight_update_alloc_mode_override: Any | None = None,
    suppress_context_exit: bool = False,
) -> None:
    """Set trainer hooks for the current thread."""
    _hooks.rollout_engine_factory = rollout_engine_factory
    _hooks.weight_update_alloc_mode_override = weight_update_alloc_mode_override
    _hooks.suppress_context_exit = suppress_context_exit


def clear_hooks() -> None:
    """Remove all trainer hooks for the current thread."""
    for name in _HOOK_NAMES:
        try:
            delattr(_hooks, name)
        except AttributeError:
            pass


def get_hook(name: str) -> Any:
    """Return the value of a named hook, or ``None`` if unset."""
    return getattr(_hooks, name, None)
