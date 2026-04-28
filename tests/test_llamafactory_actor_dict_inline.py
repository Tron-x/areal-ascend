"""Lock-in test for the LlamaFactoryTrainerActor dict-inline contract.

Background
----------
Multi-host LF runs ship the algorithm-author YAMLs (``lf_config`` and
``accelerate_config``) from the driver to every actor as **parsed
dicts**, not as paths.  This makes the actor's filesystem contract
host-agnostic: workers don't need ``examples/sft`` or
``examples/accelerate`` on disk -- a property the bare-metal PoC
relies on (``forge/docs/external_backends.md``) and that container
deployments inherit transparently.

If anyone refactors the actor to "just open the file again" the worker
contract breaks silently -- a worker without ``examples/`` would only
fail at runtime, not in CI.  This file is the cheap CI-runnable guard
against that regression.

Pure CPU + monarch-mocked: runs in seconds, no GPU, no LlamaFactory
checkout, no torch_npu.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Monarch mocks -- mirror tests/test_forge_unit.py so we don't need the
# Rust bindings installed.
# ---------------------------------------------------------------------------


class _AutoMockModule(types.ModuleType):
    """Module that returns MagicMock for any attribute not explicitly set."""

    def __getattr__(self, name):  # noqa: D401 -- mock helper
        if name.startswith("_"):
            raise AttributeError(name)
        return MagicMock()


def _endpoint_decorator(fn):
    """No-op @endpoint decorator -- the test never actually dispatches."""
    return fn


def _install_monarch_mocks() -> None:
    prefixes = [
        "monarch",
        "monarch.actor",
        "monarch._src",
        "monarch._src.actor",
        "monarch._src.actor.actor_mesh",
        "monarch._src.actor.endpoint",
        "monarch._src.spmd",
        "monarch._src.spmd.actor",
    ]
    for p in prefixes:
        if p not in sys.modules:
            sys.modules[p] = _AutoMockModule(p)

    actor_mod = sys.modules["monarch.actor"]
    actor_mod.endpoint = _endpoint_decorator

    spmd_mod = sys.modules["monarch._src.spmd.actor"]
    # Minimal SPMDActor stub -- the actor under test inherits from it
    # but for these unit tests we only exercise the pure helpers and
    # never instantiate the actor, so an empty parent class is enough.
    spmd_mod.SPMDActor = type("SPMDActor", (), {"__init__": lambda self, *a, **k: None})


_install_monarch_mocks()


# Import the module under test AFTER the mocks are installed.
from forge.actors.llamafactory_trainer import (  # noqa: E402
    _export_fsdp2_env,
    _load_yaml_or_dict,
)

# ---------------------------------------------------------------------------
# _load_yaml_or_dict
# ---------------------------------------------------------------------------


class TestLoadYamlOrDict:
    def test_dict_pass_through(self):
        """A dict is returned unchanged -- this is the dict-inline path."""
        spec = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        out = _load_yaml_or_dict(spec, label="lf")
        assert out is spec

    def test_path_is_parsed(self, tmp_path):
        """A path falls back to YAML parse -- the legacy path."""
        p = tmp_path / "lf.yaml"
        p.write_text("model_name_or_path: foo\nmax_steps: 3\n")
        out = _load_yaml_or_dict(str(p), label="lf")
        assert out == {"model_name_or_path": "foo", "max_steps": 3}

    def test_pathlike_accepted(self, tmp_path):
        """``pathlib.Path`` is honoured (real driver passes Path objects)."""
        p = tmp_path / "lf.yaml"
        p.write_text("k: v\n")
        out = _load_yaml_or_dict(p, label="lf")
        assert out == {"k": "v"}

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """``yaml.safe_load`` returns ``None`` for empty files; coerce to dict."""
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert _load_yaml_or_dict(str(p), label="lf") == {}

    def test_bad_type_raises_typeerror(self):
        """Anything other than dict/path is a programmer error."""
        with pytest.raises(TypeError) as exc:
            _load_yaml_or_dict(42, label="lf")
        assert "lf" in str(exc.value)


# ---------------------------------------------------------------------------
# _export_fsdp2_env -- end-to-end FSDP env mapping from a parsed dict.
# Crucially, this exercises the dict-inline path with NO filesystem at all,
# proving the actor never has to reopen the YAML on a worker.
# ---------------------------------------------------------------------------


def _accelerate_dict() -> dict:
    """A representative ``accelerate`` config for FSDP2 on NPU."""
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "mixed_precision": "bf16",
        "fsdp_config": {
            "fsdp_version": 2,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen3VLDecoderLayer",
            "fsdp_offload_params": False,
            "fsdp_reshard_after_forward": True,
            "fsdp_use_orig_params": True,
            "fsdp_sync_module_states": True,
            "fsdp_cpu_ram_efficient_loading": True,
        },
    }


class TestExportFsdp2EnvDictInline:
    def test_dict_input_sets_expected_env(self, monkeypatch):
        """Dict input drives the same env-var mapping as YAML input."""
        # Snapshot env so leakage from earlier tests doesn't fool us.
        for key in (
            "ACCELERATE_USE_FSDP",
            "ACCELERATE_MIXED_PRECISION",
            "FSDP_VERSION",
            "FSDP_AUTO_WRAP_POLICY",
            "FSDP_TRANSFORMER_CLS_TO_WRAP",
            "FSDP_OFFLOAD_PARAMS",
            "FSDP_RESHARD_AFTER_FORWARD",
            "FSDP_USE_ORIG_PARAMS",
            "FSDP_SYNC_MODULE_STATES",
            "FSDP_CPU_RAM_EFFICIENT_LOADING",
        ):
            monkeypatch.delenv(key, raising=False)

        out = _export_fsdp2_env(_accelerate_dict())

        assert out["ACCELERATE_USE_FSDP"] == "true"
        assert out["ACCELERATE_MIXED_PRECISION"] == "bf16"
        assert out["FSDP_VERSION"] == "2"
        assert out["FSDP_AUTO_WRAP_POLICY"] == "TRANSFORMER_BASED_WRAP"
        assert out["FSDP_TRANSFORMER_CLS_TO_WRAP"] == "Qwen3VLDecoderLayer"
        # Booleans go through the lowercase coercion path.
        assert out["FSDP_OFFLOAD_PARAMS"] == "false"
        assert out["FSDP_RESHARD_AFTER_FORWARD"] == "true"
        assert out["FSDP_USE_ORIG_PARAMS"] == "true"
        assert out["FSDP_SYNC_MODULE_STATES"] == "true"
        assert out["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "true"

    def test_dict_and_path_agree(self, monkeypatch, tmp_path):
        """Same config produces the same env regardless of dict vs path."""
        import yaml as _yaml

        cfg = _accelerate_dict()
        p = tmp_path / "accel.yaml"
        p.write_text(_yaml.safe_dump(cfg))

        for key in (
            "ACCELERATE_USE_FSDP",
            "ACCELERATE_MIXED_PRECISION",
            "FSDP_VERSION",
        ):
            monkeypatch.delenv(key, raising=False)

        from_dict = _export_fsdp2_env(cfg)
        from_path = _export_fsdp2_env(str(p))
        assert from_dict == from_path

    def test_non_fsdp_distributed_type_raises(self):
        """Actor is FSDP-only -- DDP / DeepSpeed configs must fail loudly."""
        cfg = _accelerate_dict()
        cfg["distributed_type"] = "MULTI_GPU"
        with pytest.raises(ValueError, match="distributed_type must be FSDP"):
            _export_fsdp2_env(cfg)

    def test_cpu_ram_efficient_requires_sync_module_states(self):
        """Mirror accelerate's own constraint -- fail at the actor, not in FSDP wrap."""
        cfg = _accelerate_dict()
        cfg["fsdp_config"]["fsdp_sync_module_states"] = False
        with pytest.raises(ValueError, match="fsdp_cpu_ram_efficient_loading"):
            _export_fsdp2_env(cfg)

    def test_unknown_fsdp_keys_are_silently_ignored(self, monkeypatch):
        """Tolerate accelerate schema drift -- only mapped keys propagate."""
        monkeypatch.delenv("FSDP_FAKE", raising=False)
        cfg = _accelerate_dict()
        cfg["fsdp_config"]["fsdp_definitely_not_a_real_field"] = "ignored"
        out = _export_fsdp2_env(cfg)
        assert "FSDP_DEFINITELY_NOT_A_REAL_FIELD" not in out
        assert "FSDP_FAKE" not in out
