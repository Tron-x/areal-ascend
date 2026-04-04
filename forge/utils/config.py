"""Configuration utilities -- YAML loading + CLI override.

Provides a simple but flexible configuration system:

1. Load a YAML config file into a dict
2. Apply CLI ``key=value`` overrides (dotpath notation)
3. Merge into a ``ForgeConfig`` dataclass
4. ``@parse`` decorator for entry points

Inspired by TorchForge's ``util/config.py`` but simplified
(no pydantic, no _component_ pattern).

Usage::

    # In a script:
    from forge.utils.config import load_config, parse

    @parse
    def main(cfg):
        print(cfg.experiment_name)

    # CLI:
    python my_script.py --config config.yaml experiment_name=my_exp
"""

from __future__ import annotations

import argparse
import functools
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Supports OmegaConf-style ``${var}`` references if omegaconf is
    installed, otherwise returns raw dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(raw)
        return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        return raw


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI ``key=value`` overrides using dotpath notation.

    Supports:
    - ``key=value`` (set a value)
    - ``nested.key=value`` (set a nested value)
    - ``~key`` (delete a key)
    - Values auto-cast: int, float, bool, null, json lists/dicts

    Args:
        cfg: Base config dict.
        overrides: List of ``key=value`` strings.

    Returns:
        Modified config dict.
    """
    for override in overrides:
        if override.startswith("~"):
            _delete_dotpath(cfg, override[1:])
            continue

        if "=" not in override:
            logger.warning(f"Skipping invalid override (no '='): {override}")
            continue

        key, value = override.split("=", 1)
        key = key.strip()
        value = _coerce_value(value.strip())
        _set_dotpath(cfg, key, value)

    return cfg


def dict_to_dataclass(cfg: dict[str, Any], cls: type) -> Any:
    """Convert a dict to a dataclass, ignoring unknown keys.

    Args:
        cfg: Config dict.
        cls: Dataclass type (e.g. ``ForgeConfig``).

    Returns:
        Dataclass instance with values from cfg.
    """
    valid_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in cfg.items() if k in valid_fields}
    return cls(**filtered)


def load_config(
    config_path: str | None = None,
    overrides: list[str] | None = None,
    config_cls: type | None = None,
) -> Any:
    """Load config from YAML + CLI overrides.

    Args:
        config_path: Path to YAML config file. None for empty config.
        overrides: CLI ``key=value`` override list.
        config_cls: Dataclass to convert to. None returns raw dict.

    Returns:
        Config dict or dataclass instance.
    """
    cfg = {}
    if config_path:
        cfg = load_yaml(config_path)

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    if config_cls is not None:
        return dict_to_dataclass(cfg, config_cls)

    return cfg


def parse(fn):
    """Decorator that parses ``--config`` + CLI overrides and calls ``fn(cfg)``.

    Usage::

        @parse
        def main(cfg):
            print(cfg)  # dict or ForgeConfig

        # CLI: python script.py --config config.yaml key=value
    """

    @functools.wraps(fn)
    def wrapper():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=None, help="YAML config file")
        args, unknown = parser.parse_known_args()

        cfg = load_config(config_path=args.config, overrides=unknown)
        return fn(cfg)

    return wrapper


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _set_dotpath(d: dict, dotpath: str, value: Any) -> None:
    """Set a value in a nested dict using dotpath notation."""
    keys = dotpath.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _delete_dotpath(d: dict, dotpath: str) -> None:
    """Delete a key from a nested dict using dotpath notation."""
    keys = dotpath.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current:
            return
        current = current[key]
    current.pop(keys[-1], None)


def _coerce_value(s: str) -> Any:
    """Auto-cast a string value to the appropriate Python type."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in ("null", "none"):
        return None

    try:
        return int(s)
    except ValueError:
        pass

    try:
        return float(s)
    except ValueError:
        pass

    if (s.startswith("[") and s.endswith("]")) or (
        s.startswith("{") and s.endswith("}")
    ):
        import json

        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            pass

    return s
