"""Shared conftest for AReaL / Forge tests.

Provides helpers for mocking heavy dependencies so test modules can
import without triggering the full dependency chain.
"""

import sys
import types


def _make_module(name, attrs=None):
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    return mod


def _install_package_mock(name, sub_names=None):
    pkg = _make_module(name)
    sys.modules[name] = pkg
    for sub in sub_names or []:
        submod = types.ModuleType(f"{name}.{sub}")
        sys.modules[f"{name}.{sub}"] = submod


def _install_mock_package(name, submods=None):
    pkg = types.ModuleType(name)
    sys.modules[name] = pkg
    for sub in submods or []:
        sys.modules[f"{name}.{sub}"] = types.ModuleType(f"{name}.{sub}")
    return pkg
