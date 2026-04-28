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

# Reward "scope" determines how :class:`RewardPipeline` invokes the
# registered function:
#
#   * ``final``   -- single scalar for the whole trajectory.  Called
#                    with the legacy ``(prompt, response, **kw)``
#                    signature.  This is the default for backward
#                    compatibility; every reward that predated A3
#                    keeps working unchanged.
#   * ``process`` -- per-turn scalars.  Called with the trajectory
#                    object and expected to return a list[float] the
#                    same length as ``trajectory.turns``.
#
# Scope is stored both in an auxiliary registry (``_SCOPES``) and as
# an attribute on the function (``fn._forge_reward_scope``) so either
# introspection path works.
_VALID_SCOPES = ("final", "process")

# Name -> reward callable.
_REGISTRY: dict[str, Callable[..., Any]] = {}
# Name -> scope string ("final" | "process").  Parallel dict rather
# than a combined struct so the hot-path ``get_reward`` lookup stays
# a single dict access.
_SCOPES: dict[str, str] = {}

# Lazy-init guard.  The first call to ``ensure_loaded`` scans the built-in
# reward modules; subsequent calls are no-ops.  External packages that
# register rewards from elsewhere should have their module imported by
# the time the user calls ``get_reward``.
_AUTO_DISCOVER_DONE = False


def register_reward(name: str, *, scope: str = "final"):
    """Decorator: register ``fn`` under ``name`` in the reward registry.

    Usage::

        # Final-scope: one scalar per trajectory (legacy, default).
        @register_reward("gsm8k")
        def gsm8k_reward_fn(prompt, response, **kwargs) -> float:
            ...

        # Process-scope: per-turn scalars (Phase A3).
        @register_reward("tool_call_valid", scope="process")
        def tool_call_valid_reward(trajectory) -> list[float]:
            return [0.1 if turn.tool_calls else 0.0
                    for turn in trajectory.turns]

    Backward compatibility: ``scope`` defaults to ``"final"``, so
    every legacy reward decorated with ``@register_reward("x")``
    keeps exactly the same behavior and signature.

    Duplicates raise ``ValueError`` at decoration time -- this surfaces
    collisions early rather than producing silently inconsistent
    behavior at run time.  If you intentionally want to replace an
    existing entry (hot-reload during iteration), delete from
    ``_REGISTRY`` first.
    """
    if scope not in _VALID_SCOPES:
        raise ValueError(f"invalid scope {scope!r}; must be one of {_VALID_SCOPES}")

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(
                f"reward {name!r} is already registered to "
                f"{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}; "
                f"refusing to overwrite with "
                f"{fn.__module__}.{fn.__qualname__}"
            )
        _REGISTRY[name] = fn
        _SCOPES[name] = scope
        # Stash on the function too so callers that have a
        # reference to ``fn`` (without going through the registry)
        # can still introspect the scope.
        try:
            fn._forge_reward_scope = scope  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Builtins / C extensions may reject attribute assignment;
            # the parallel ``_SCOPES`` dict still covers us.
            pass
        return fn

    return wrap


def get_reward_scope(name: str) -> str:
    """Return the registered scope (``"final"`` or ``"process"``) for
    ``name``.  Unknown names raise ``ValueError`` via :func:`get_reward`
    so the error message already lists placement hints.
    """
    get_reward(name)  # reuses the rich error path
    return _SCOPES.get(name, "final")


def rewards_by_scope(scope: str) -> list[str]:
    """List registered short names whose scope equals ``scope``.

    Useful for :class:`RewardPipeline` to discover all
    process-scope rewards without hard-coding their names.
    """
    ensure_loaded()
    return sorted(n for n, s in _SCOPES.items() if s == scope)


def get_reward(name: str) -> Callable[..., Any]:
    """Lookup a registered reward by short name.

    Raises ``ValueError`` with the list of available names AND
    placement hints when the short name is unknown.  The hints are
    the single biggest UX issue the first walk-through caught: new
    users write a function + decorator, put the file under
    ``examples/custom_rewards/`` (because that's where the sample
    lives), run training, and hit an opaque "unknown reward" error
    because auto-discovery only scans ``forge.reward.*`` /
    ``areal.reward.*``.  This error message tells them exactly what
    to do.
    """
    ensure_loaded()
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"unknown reward {name!r}\n"
            f"\n"
            f"Available rewards: {available}\n"
            f"\n"
            f"If you wrote `@register_reward({name!r})` but it's not\n"
            f"listed above, the decorator's module wasn't imported.\n"
            f"Pick ONE of:\n"
            f"\n"
            f"  (a) Auto-discovery (recommended).  Place the file at\n"
            f"      one of:\n"
            f"        forge/reward/<anything>.py\n"
            f"        areal/reward/<anything>.py\n"
            f"      Auto-discovery scans those two packages at lookup\n"
            f"      time, so a plain file copy is enough.\n"
            f"\n"
            f"  (b) Explicit import from your launch entry script.\n"
            f"      Add before you start training:\n"
            f"        import my_reward_module  # noqa: F401\n"
            f"\n"
            f"See examples/custom_rewards/README.md for the full\n"
            f"3-step extension guide."
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
    _SCOPES.clear()
    _AUTO_DISCOVER_DONE = False


__all__ = [
    "register_reward",
    "get_reward",
    "get_reward_scope",
    "rewards_by_scope",
    "available_rewards",
    "ensure_loaded",
    "reset_for_tests",
]
