"""Forge reward registry.

Algorithm-engineer-facing contract: write a function, slap a
``@register_reward("my_name")`` on it, and reference it in YAML as
``reward: my_name``. No more dotted-import-path plumbing, no more
editing a dispatch dict in two places.

    # my_custom_reward.py
    from forge.reward import register_reward

    @register_reward("my_first_reward")
    def my_first_reward(prompt: str, response: str, **kw) -> float:
        return 1.0 if kw.get("ground_truth", "") in response else 0.0

    # gsm8k_grpo_npu.yaml
    reward: my_first_reward          # short name  (preferred)
    # or:
    reward_fn_path: my.module.fn     # long path   (backward compat)

Design notes:

* This registry is strictly additive — it **coexists** with the
  pre-existing dotted-path mechanism
  (``areal.reward.get_custom_reward_fn`` / ``forge_cfg.reward_fn_path``)
  so no existing YAML or script breaks.  ``grpo.py`` resolves short
  name first, falls back to long path for legacy configs.

* **Auto-discovery** imports every submodule of ``forge.reward`` and
  ``areal.reward`` at ``ensure_loaded`` time so decorators in those
  modules fire without each caller needing to ``import`` them
  explicitly.  External packages can also call
  ``register_reward(name)`` from anywhere — as long as their module
  is imported once before ``get_reward`` is called, the function is
  in the registry.

* Error messages list the available names, so a typo immediately
  tells the user what they could have meant.

Not in scope for this MVP:

* Scoring/priority for duplicate names (we reject duplicates).
* Remote registries / plugin discovery via entry_points (fine to add
  later; the wrap function is the injection point).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("RewardRegistry")

# Name -> reward callable.
_REGISTRY: dict[str, Callable[..., Any]] = {}

# Lazy-init guard.  The first call to ``ensure_loaded`` scans the built-in
# reward modules; subsequent calls are no-ops.  External packages that
# register rewards from elsewhere should have their module imported by
# the time the user calls ``get_reward``.
_AUTO_DISCOVER_DONE = False


def register_reward(name: str):
    """Decorator: register ``fn`` under ``name`` in the reward registry.

    Usage::

        @register_reward("gsm8k")
        def gsm8k_reward_fn(prompt, response, **kwargs) -> float:
            ...

    Duplicates raise ``ValueError`` at decoration time — this surfaces
    collisions early rather than producing silently inconsistent
    behavior at run time.  If you intentionally want to replace an
    existing entry (hot-reload during iteration), delete from
    ``_REGISTRY`` first.
    """

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(
                f"reward {name!r} is already registered to "
                f"{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}; "
                f"refusing to overwrite with "
                f"{fn.__module__}.{fn.__qualname__}"
            )
        _REGISTRY[name] = fn
        return fn

    return wrap


def get_reward(name: str) -> Callable[..., Any]:
    """Lookup a registered reward by short name.

    Raises ``ValueError`` with the list of available names when the
    short name is unknown.  Callers should catch and translate if they
    want a different error class.
    """
    ensure_loaded()
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"unknown reward {name!r}, available: {available}. "
            f"Register new ones with @register_reward(...) in a module "
            f"under forge.reward.* / areal.reward.* (auto-imported at "
            f"lookup time), or import the module explicitly before "
            f"calling get_reward()."
        )
    return _REGISTRY[name]


def available_rewards() -> list[str]:
    """Return the sorted list of registered reward names."""
    ensure_loaded()
    return sorted(_REGISTRY)


def ensure_loaded() -> None:
    """Import the built-in reward packages so their
    ``@register_reward`` decorators fire.

    Idempotent; safe to call from any hot path.  Runs at most one
    walk of ``forge.reward`` and ``areal.reward``.
    """
    global _AUTO_DISCOVER_DONE
    if _AUTO_DISCOVER_DONE:
        return
    _AUTO_DISCOVER_DONE = True  # set early to prevent re-entry on errors

    for pkg_name in ("forge.reward", "areal.reward"):
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            logger.debug(
                "ensure_loaded: cannot import %s (%s); skipping",
                pkg_name,
                e,
            )
            continue
        # Walk submodules.  ``walk_packages`` needs a list of package
        # paths; iterate and import each one lazily so a single
        # submodule's import error doesn't nuke the whole discovery.
        if not hasattr(pkg, "__path__"):
            continue
        for modinfo in pkgutil.iter_modules(pkg.__path__):
            # Skip private / dunder modules.
            if modinfo.name.startswith("_"):
                continue
            full_name = f"{pkg_name}.{modinfo.name}"
            try:
                importlib.import_module(full_name)
            except Exception as e:
                logger.debug(
                    "ensure_loaded: cannot import %s (%s); skipping",
                    full_name,
                    e,
                )


def reset_for_tests() -> None:
    """Test-only: clear the registry + re-run discovery on next call.

    Real callers never need this.  Exists so unit tests can reset the
    module between cases without process restarts.
    """
    global _AUTO_DISCOVER_DONE
    _REGISTRY.clear()
    _AUTO_DISCOVER_DONE = False


__all__ = [
    "register_reward",
    "get_reward",
    "available_rewards",
    "ensure_loaded",
    "reset_for_tests",
]
