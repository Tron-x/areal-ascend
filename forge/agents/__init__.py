"""Forge agent registry + built-in agent logic implementations.

Algorithm-engineer-facing contract (mirrors
:mod:`forge.reward`): write an agent class, slap a
``@register_agent("my_name")`` on it, and reference it in YAML as
``agent: my_name``.

    # my_custom_agent.py
    from forge.agents import register_agent

    @register_agent("my_first_agent")
    class MyFirstAgent:
        def __init__(self, max_turns: int = 5):
            self.max_turns = max_turns
        async def run_episode(self, prompt, model_proxy, **kw): ...

    # training YAML
    agent: my_first_agent         # short name (preferred)
    agent_config:                 # forwarded to constructor
      max_turns: 8

Design notes:

* What the registry stores is the **class** (or factory).  The caller
  (typically ``forge.apps.agent_rl``) is responsible for constructing
  it with the right arguments.  The registry does not prescribe a
  single constructor signature because the 4 built-in agents
  (ReAct / ReTool / Harbor / External) genuinely have different
  interfaces.

* Auto-discovery imports every submodule of ``forge.agents`` at
  ``ensure_loaded`` time, so decorators fire without each caller
  needing to import them explicitly.

* This module also re-exports the 4 legacy agent classes so existing
  ``from forge.agents import ReToolAgent`` imports keep working.

Not in scope for this Phase-A1 MVP:

* Protocol normalization across the 4 agent styles (that's a later
  arc; the design doc calls it out explicitly).
* Hot-reload / replacement.  Duplicates raise at decoration time.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("AgentRegistry")

# Name -> agent class (or factory).  Agents are normally classes, but
# the registry is permissive: anything callable that returns an
# ``AgentLogic``-compatible object is acceptable.
_REGISTRY: dict[str, Callable[..., Any]] = {}

# Lazy-init guard -- scan built-in agent modules on first lookup.
_AUTO_DISCOVER_DONE = False


def register_agent(name: str):
    """Decorator: register ``cls`` under ``name`` in the agent registry.

    Duplicates raise ``ValueError`` at decoration time.  If you
    intentionally want to replace an existing entry, delete from
    ``_REGISTRY`` first (or call :func:`reset_for_tests`).
    """

    def wrap(cls: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"agent {name!r} is already registered to "
                f"{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}; "
                f"refusing to overwrite with "
                f"{cls.__module__}.{cls.__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_agent(name: str) -> Callable[..., Any]:
    """Lookup a registered agent by short name.

    Raises ``ValueError`` listing available names + placement hints
    when the short name is unknown.  Same hint style as
    :func:`forge.reward.get_reward` so the failure mode is consistent
    across registries.
    """
    ensure_loaded()
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"unknown agent {name!r}\n"
            f"\n"
            f"Available agents: {available}\n"
            f"\n"
            f"If you wrote `@register_agent({name!r})` but it's not\n"
            f"listed above, the decorator's module wasn't imported.\n"
            f"Pick ONE of:\n"
            f"\n"
            f"  (a) Auto-discovery (recommended).  Place the file at:\n"
            f"        forge/agents/<anything>.py\n"
            f"      Auto-discovery scans that package at lookup time,\n"
            f"      so a plain file copy is enough.\n"
            f"\n"
            f"  (b) Explicit import from your launch entry script.\n"
            f"      Add before training starts:\n"
            f"        import my_agent_module  # noqa: F401\n"
            f"\n"
            f"Run `python -m forge list-agents` to verify registration."
        )
    return _REGISTRY[name]


def available_agents() -> list[str]:
    """Return the sorted list of registered agent short names."""
    ensure_loaded()
    return sorted(_REGISTRY)


def ensure_loaded() -> None:
    """Import the built-in ``forge.agents`` submodules so
    ``@register_agent`` decorators fire.

    Idempotent; safe to call from any hot path.
    """
    global _AUTO_DISCOVER_DONE
    if _AUTO_DISCOVER_DONE:
        return
    _AUTO_DISCOVER_DONE = True  # set early to prevent re-entry on errors

    try:
        pkg = importlib.import_module("forge.agents")
    except Exception as e:  # pragma: no cover -- defensive
        logger.debug("ensure_loaded: cannot import forge.agents (%s); skipping", e)
        return
    if not hasattr(pkg, "__path__"):
        return
    for modinfo in pkgutil.iter_modules(pkg.__path__):
        if modinfo.name.startswith("_"):
            continue
        full_name = f"forge.agents.{modinfo.name}"
        try:
            importlib.import_module(full_name)
        except Exception as e:
            logger.debug("ensure_loaded: cannot import %s (%s); skipping", full_name, e)


def reset_for_tests() -> None:
    """Test-only: clear the registry + re-run discovery on next call."""
    global _AUTO_DISCOVER_DONE
    _REGISTRY.clear()
    _AUTO_DISCOVER_DONE = False


# Legacy re-exports -- keep existing ``from forge.agents import X`` paths alive.
# These imports also double as eager registration so the decorators fire at
# package-import time (``ensure_loaded`` is still the guarantee point for
# out-of-order callers).
from forge.agents.external import ExternalAgentRunner  # noqa: E402
from forge.agents.harbor import HarborAgentLogic  # noqa: E402
from forge.agents.react import SimpleReActAgent  # noqa: E402
from forge.agents.retool import ReToolAgent  # noqa: E402

__all__ = [
    "register_agent",
    "get_agent",
    "available_agents",
    "ensure_loaded",
    "reset_for_tests",
    "ExternalAgentRunner",
    "HarborAgentLogic",
    "ReToolAgent",
    "SimpleReActAgent",
]
