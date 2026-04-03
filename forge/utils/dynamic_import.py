"""Dynamic import utility for Forge (no areal dependency)."""

from __future__ import annotations

import importlib
from typing import Any


def import_from_string(dotted_path: str) -> Any:
    """Import an object from a dotted path like ``package.module.ClassName``.

    Tries ``module:attr`` first, then ``module.attr`` split at the last dot.
    """
    if ":" in dotted_path:
        module_path, _, attr_name = dotted_path.partition(":")
        mod = importlib.import_module(module_path)
        return getattr(mod, attr_name)

    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        return importlib.import_module(dotted_path)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)
