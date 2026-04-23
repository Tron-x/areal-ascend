"""Precedence-aware YAML > env > default value resolver.

Centralizes the lookup rule described in
``forge/docs/env_to_yaml_mapping.md``:

    YAML field (authoritative) > env var (legacy/override) > default

Every weight-sync / launcher knob that used to be read as
``os.environ.get(NAME, default)`` should go through :func:`resolve`
or :func:`resolve_bool` / :func:`resolve_int` after the refactor.
Benefits:

* **Single precedence rule** across the codebase — no more
  file-by-file variations of "prefer env" vs "prefer YAML".
* **Deprecation warning for free** — when a user sets the env var,
  the helper emits one structured warning line pointing to the YAML
  field they should migrate to.  Warnings silenced via
  ``FORGE_SUPPRESS_DEPRECATION=1`` for CI automation.
* **Type coercion in one place** — ``_bool``/``_int`` variants handle
  the ``"0"/"1"/"true"/"false"`` soup uniformly.

Usage (typical):

.. code-block:: python

    from forge.engines.weight_sync._config_resolver import resolve_int

    pool_mb = resolve_int(
        yaml_value=cfg.launcher.weight_sync.pool_mb,
        env_name="TORCHSTORE_MONARCH_RDMA_POOL_MB",
        default=8192,
    )

If ``cfg.launcher.weight_sync.pool_mb`` is set in YAML, it wins.
Otherwise if ``TORCHSTORE_MONARCH_RDMA_POOL_MB`` is in the shell env,
its value wins and a deprecation warning is emitted.  Otherwise 8192.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TypeVar

logger = logging.getLogger("ConfigResolver")

T = TypeVar("T")

# One-shot guard so each (env_name, yaml_field) pair warns at most once
# per process -- otherwise a per-step read would flood the log.
_WARNED_ENVS: set[str] = set()


def _should_suppress_warnings() -> bool:
    return os.environ.get("FORGE_SUPPRESS_DEPRECATION", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _warn_legacy_env(env_name: str, yaml_field_hint: str, env_value: str) -> None:
    """Emit a single-line deprecation warning for a legacy env var.

    ``yaml_field_hint`` is a user-readable dotted path to the YAML field
    that replaces ``env_name`` (e.g. ``"launcher.weight_sync.pool_mb"``).
    The warning is emitted once per env_name per process, and silenced
    entirely by ``FORGE_SUPPRESS_DEPRECATION=1``.
    """
    if env_name in _WARNED_ENVS:
        return
    _WARNED_ENVS.add(env_name)
    if _should_suppress_warnings():
        return
    msg = (
        f"[forge deprecation] env var {env_name}={env_value!r} is still "
        f"honored as an override, but the canonical location is the "
        f"YAML field `{yaml_field_hint}`. Move the setting into YAML; "
        f"env support will be removed in the next major release. "
        f"Silence with FORGE_SUPPRESS_DEPRECATION=1."
    )
    # Write directly to stderr so the warning always reaches the user,
    # regardless of how the ambient logging chain is configured. forge
    # tooling installs per-actor colored handlers + multiple named
    # loggers, and a framework-level ``ConfigResolver`` logger can get
    # filtered by those handlers before reaching the console. stderr
    # bypasses that entirely -- exactly matches how Python's own
    # ``DeprecationWarning`` surfaces to users.
    print(msg, file=sys.stderr, flush=True)
    # Still log at WARNING on the named logger so structured log sinks
    # (file, syslog, monitoring agents) can pick it up too.
    logger.warning(msg)


def resolve(
    *,
    yaml_value: T | None,
    env_name: str,
    default: T,
    yaml_field_hint: str | None = None,
    env_parser: callable = str,  # type: ignore[valid-type]
) -> T:
    """Return the effective value following ``YAML > env > default``.

    Args:
        yaml_value: value read out of the parsed config dataclass, or
            ``None`` when the user didn't set it in YAML.
        env_name: name of the legacy env var.  Used for both lookup and
            deprecation-warning text.
        default: fallback when neither YAML nor env provides a value.
        yaml_field_hint: human-readable dotted path to the YAML field
            that replaces ``env_name`` (used in the deprecation
            message).  Defaults to ``env_name.lower()`` if omitted.
        env_parser: converts the raw env-var string into the target type.
            Use ``str`` (default), ``int``, or a custom parser; the
            convenience functions ``resolve_bool`` and ``resolve_int``
            below wrap the common cases.

    Returns:
        The resolved value.  Type is whatever ``env_parser`` returns,
        or the type of ``yaml_value`` if YAML wins.
    """
    if yaml_value is not None:
        return yaml_value

    env_raw = os.environ.get(env_name)
    if env_raw is not None and env_raw != "":
        _warn_legacy_env(env_name, yaml_field_hint or env_name.lower(), env_raw)
        try:
            return env_parser(env_raw)  # type: ignore[return-value]
        except (ValueError, TypeError) as exc:
            logger.error(
                "env var %s=%r failed to parse as %s; falling back to default %r. "
                "Error: %s",
                env_name,
                env_raw,
                getattr(env_parser, "__name__", env_parser),
                default,
                exc,
            )

    return default


def _parse_bool(s: str) -> bool:
    """Liberal bool parser matching shell conventions.

    ``"1"/"true"/"yes"/"on"`` -> True,
    ``"0"/"false"/"no"/"off"/""`` -> False, everything else raises.
    """
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"cannot parse {s!r} as bool")


def resolve_bool(
    *,
    yaml_value: bool | None,
    env_name: str,
    default: bool,
    yaml_field_hint: str | None = None,
) -> bool:
    """``resolve`` specialized for bool env vars."""
    return resolve(
        yaml_value=yaml_value,
        env_name=env_name,
        default=default,
        yaml_field_hint=yaml_field_hint,
        env_parser=_parse_bool,
    )


def resolve_int(
    *,
    yaml_value: int | None,
    env_name: str,
    default: int,
    yaml_field_hint: str | None = None,
) -> int:
    """``resolve`` specialized for int env vars."""
    return resolve(
        yaml_value=yaml_value,
        env_name=env_name,
        default=default,
        yaml_field_hint=yaml_field_hint,
        env_parser=int,
    )


def resolve_str(
    *,
    yaml_value: str | None,
    env_name: str,
    default: str,
    yaml_field_hint: str | None = None,
) -> str:
    """``resolve`` specialized for str env vars.  Alias of ``resolve``
    with ``env_parser=str``; exists for symmetry with the bool/int
    helpers so call sites all look identical."""
    return resolve(
        yaml_value=yaml_value,
        env_name=env_name,
        default=default,
        yaml_field_hint=yaml_field_hint,
        env_parser=str,
    )


def reset_warning_cache() -> None:
    """Test-only hook: clears the "warned once per proc" cache.

    Real callers should never need this; it exists so unit tests that
    exercise the deprecation-warning path can run multiple times in
    the same process.
    """
    _WARNED_ENVS.clear()


__all__ = [
    "resolve",
    "resolve_bool",
    "resolve_int",
    "resolve_str",
    "reset_warning_cache",
]
