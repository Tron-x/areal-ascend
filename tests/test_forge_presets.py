"""Unit tests for ``forge.cli.presets`` — the two-layer YAML composer.

These tests cover the preset lookup precedence, the role override
whitelist / escape-hatch / deny-list semantics, and end-to-end
composition.  They are pure-python and do not need any GPU / monarch /
network setup (unlike the integration smoke tests under
``tests/test_forge_ssh_job.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.cli.presets import (
    ALGO_OVERRIDE_ESCAPE_HATCH,
    ALGO_OVERRIDE_WHITELIST,
    PresetError,
    compose_launcher_yaml,
    merge_role_overrides,
    resolve_cluster_preset,
)


# ---------------------------------------------------------------------------
# resolve_cluster_preset
# ---------------------------------------------------------------------------


def test_resolve_absolute_path_returns_as_is(tmp_path: Path) -> None:
    preset = tmp_path / "my_cluster.yaml"
    preset.write_text("launcher:\n  pool: []\n")

    result = resolve_cluster_preset(str(preset))

    assert result == preset.resolve()


def test_resolve_absolute_path_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"

    with pytest.raises(PresetError, match="does not exist"):
        resolve_cluster_preset(str(missing))


def test_resolve_project_local_override_wins_over_builtin(tmp_path: Path) -> None:
    """Project-local ``<algo_dir>/clusters/<name>.yaml`` beats the in-tree preset.

    This is the "vendor a cluster YAML next to my experiment" path.
    """
    # Builtin preset with the same stem.
    builtin_dir = tmp_path / "builtin"
    builtin_dir.mkdir()
    (builtin_dir / "2node.yaml").write_text("launcher: {pool: []}\n# builtin\n")

    # Project-local override.
    algo_dir = tmp_path / "algo"
    (algo_dir / "clusters").mkdir(parents=True)
    local = algo_dir / "clusters" / "2node.yaml"
    local.write_text("launcher: {pool: []}\n# local\n")

    result = resolve_cluster_preset(
        "2node",
        algo_yaml_dir=algo_dir,
        builtin_dir=builtin_dir,
    )

    assert result == local.resolve()


def test_resolve_falls_back_to_builtin(tmp_path: Path) -> None:
    builtin_dir = tmp_path / "builtin"
    builtin_dir.mkdir()
    (builtin_dir / "2node.yaml").write_text("launcher: {pool: []}\n")

    algo_dir = tmp_path / "algo"
    algo_dir.mkdir()  # no clusters/ subfolder

    result = resolve_cluster_preset(
        "2node",
        algo_yaml_dir=algo_dir,
        builtin_dir=builtin_dir,
    )

    assert result == (builtin_dir / "2node.yaml").resolve()


def test_resolve_unknown_lists_available_presets(tmp_path: Path) -> None:
    builtin_dir = tmp_path / "builtin"
    builtin_dir.mkdir()
    (builtin_dir / "2node_colocated.yaml").write_text("launcher: {}\n")
    (builtin_dir / "4node_dedicated_ps.yaml").write_text("launcher: {}\n")

    with pytest.raises(PresetError) as exc:
        resolve_cluster_preset(
            "does_not_exist",
            algo_yaml_dir=None,
            builtin_dir=builtin_dir,
        )

    msg = str(exc.value)
    assert "does_not_exist" in msg
    assert "2node_colocated" in msg
    assert "4node_dedicated_ps" in msg


# ---------------------------------------------------------------------------
# merge_role_overrides
# ---------------------------------------------------------------------------


def _preset_roles() -> dict[str, dict]:
    """Canonical preset roles for 2-node colocated layout."""
    return {
        "trainer": {"devices": 4, "hardware": "npu"},
        "generator": {"devices": 4, "hardware": "npu"},
        "storage": {"devices": 4, "hardware": "npu", "colocate": "trainer"},
        "reward": {"devices": 0, "hardware": "npu", "colocate": "trainer"},
    }


def test_merge_returns_copy_not_mutating_preset() -> None:
    preset = _preset_roles()
    algo = {"trainer": {"devices": 8}}

    merged = merge_role_overrides(preset, algo)
    merged["trainer"]["devices"] = 999

    assert preset["trainer"]["devices"] == 4


def test_merge_devices_override_silent() -> None:
    preset = _preset_roles()
    algo = {
        "trainer": {"devices": 8},
        "generator": {"devices": 2},
    }

    merged = merge_role_overrides(preset, algo)

    assert merged["trainer"]["devices"] == 8
    assert merged["generator"]["devices"] == 2
    # Untouched preset fields survive.
    assert merged["storage"] == {"devices": 4, "hardware": "npu", "colocate": "trainer"}


def test_merge_extras_is_whitelisted() -> None:
    preset = _preset_roles()
    algo = {"trainer": {"devices": 4, "extras": {"fsdp_dp": 4, "fsdp_tp": 1}}}

    merged = merge_role_overrides(preset, algo)

    assert merged["trainer"]["extras"] == {"fsdp_dp": 4, "fsdp_tp": 1}


def test_merge_no_algo_roles_is_noop() -> None:
    preset = _preset_roles()

    merged = merge_role_overrides(preset, None)
    assert merged == preset
    merged_empty = merge_role_overrides(preset, {})
    assert merged_empty == preset


def test_merge_escape_hatch_triggers_callback() -> None:
    preset = _preset_roles()
    algo = {"storage": {"colocate": "generator"}}
    events: list[tuple[str, str, object]] = []

    merged = merge_role_overrides(
        preset, algo, on_escape_hatch=lambda r, f, v: events.append((r, f, v))
    )

    assert merged["storage"]["colocate"] == "generator"
    assert events == [("storage", "colocate", "generator")]


def test_merge_unknown_role_raises() -> None:
    preset = _preset_roles()
    algo = {"my_new_role": {"devices": 4}}

    with pytest.raises(PresetError, match="no such role"):
        merge_role_overrides(preset, algo)


def test_merge_non_whitelisted_field_raises() -> None:
    preset = _preset_roles()
    # ``hardware`` is an infra decision — algo YAML cannot flip
    # npu -> gpu silently.
    algo = {"trainer": {"hardware": "gpu"}}

    with pytest.raises(PresetError, match="cannot override roles.trainer.hardware"):
        merge_role_overrides(preset, algo)


def test_merge_role_value_must_be_mapping() -> None:
    preset = _preset_roles()
    algo = {"trainer": 4}  # wrong shape — forgot `{devices: 4}`

    with pytest.raises(PresetError, match="must be a mapping"):
        merge_role_overrides(preset, algo)


def test_whitelist_and_escape_hatch_are_disjoint() -> None:
    """Guard against accidental overlap — each field lives in exactly one set."""
    assert ALGO_OVERRIDE_WHITELIST.isdisjoint(ALGO_OVERRIDE_ESCAPE_HATCH)


# ---------------------------------------------------------------------------
# compose_launcher_yaml
# ---------------------------------------------------------------------------


def _preset_yaml() -> dict:
    return {
        "launcher": {
            "type": "bare_metal",
            "launcher_impl": "ssh_job",
            "pool": [
                {
                    "host": "192.168.0.26",
                    "port": 22222,
                    "hardware": "npu",
                    "n_devices": 8,
                },
                {
                    "host": "192.168.0.23",
                    "port": 22222,
                    "hardware": "npu",
                    "n_devices": 8,
                    "role": "driver",
                },
            ],
            "roles": _preset_roles(),
            "weight_sync": {
                "method": "torchstore",
                "backend": "torchstore_multi_vol",
                "storage_role": "storage",
            },
        }
    }


def test_compose_end_to_end_overrides_devices() -> None:
    preset = _preset_yaml()
    algo = {
        "experiment_name": "gsm8k-grpo",
        "actor": {"path": "Qwen/Qwen2.5-1.5B-Instruct"},
        "roles": {
            "trainer": {"devices": 4},
            "generator": {"devices": 4},
            "storage": {"devices": 4},
        },
    }

    composed = compose_launcher_yaml(algo, preset)

    assert composed["launcher"]["pool"] == preset["launcher"]["pool"]
    assert composed["launcher"]["weight_sync"] == preset["launcher"]["weight_sync"]
    assert composed["launcher"]["launcher_impl"] == "ssh_job"
    assert composed["launcher"]["roles"]["trainer"]["devices"] == 4
    assert composed["launcher"]["roles"]["trainer"]["hardware"] == "npu"
    assert composed["launcher"]["roles"]["storage"]["colocate"] == "trainer"


def test_compose_no_algo_roles_passes_preset_unchanged() -> None:
    """If algo YAML declares no ``roles:`` block, preset is used verbatim."""
    preset = _preset_yaml()
    algo = {"experiment_name": "gsm8k-grpo"}

    composed = compose_launcher_yaml(algo, preset)

    assert composed["launcher"]["roles"] == preset["launcher"]["roles"]


def test_compose_algo_roles_wrong_shape_raises() -> None:
    preset = _preset_yaml()
    algo = {"roles": "trainer=4"}  # operator typo

    # Non-dict algo_roles is silently ignored (see compose_launcher_yaml
    # docstring: we only consume dict-shaped roles from the algo).
    composed = compose_launcher_yaml(algo, preset)

    assert composed["launcher"]["roles"] == preset["launcher"]["roles"]


def test_compose_missing_launcher_block_raises() -> None:
    bad_preset = {"not_a_launcher": {}}
    algo = {"roles": {"trainer": {"devices": 4}}}

    with pytest.raises(PresetError, match="missing a top-level 'launcher:' block"):
        compose_launcher_yaml(algo, bad_preset)


def test_compose_deep_copies_preset() -> None:
    """Composing twice from the same preset must not accumulate mutations."""
    preset = _preset_yaml()
    algo1 = {"roles": {"trainer": {"devices": 2}}}
    algo2 = {"roles": {"trainer": {"devices": 8}}}

    c1 = compose_launcher_yaml(algo1, preset)
    c2 = compose_launcher_yaml(algo2, preset)

    assert c1["launcher"]["roles"]["trainer"]["devices"] == 2
    assert c2["launcher"]["roles"]["trainer"]["devices"] == 8
    # Preset still pristine.
    assert preset["launcher"]["roles"]["trainer"]["devices"] == 4


def test_compose_escape_hatch_callback_is_forwarded() -> None:
    preset = _preset_yaml()
    algo = {"roles": {"storage": {"colocate": "generator"}}}
    events: list[tuple[str, str, object]] = []

    compose_launcher_yaml(
        algo,
        preset,
        on_escape_hatch=lambda r, f, v: events.append((r, f, v)),
    )

    assert events == [("storage", "colocate", "generator")]
