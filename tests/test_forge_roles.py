"""Pure-Python unit tests for the R1.5 role abstraction and the A2
tool-server placement resolver.

No Monarch, no distributed runtime, no torch.  These tests cover the
config translation layer in isolation:

* ``LauncherConfig.roles`` -> flat :class:`RoleConfig`
* R1.5 bidirectional bridge between ``roles`` and legacy ``meshes``
* R1.5 ``storage_role`` / ``storage_mesh`` deprecation alias
* R1.5 cross-validation (``weight_sync.storage_role`` must name a role)
* ``resolve_tool_server_options`` precedence + backward compat
"""

from __future__ import annotations

import pytest

from forge.core.types import LauncherConfig, RoleConfig, WeightSyncBlock
from forge.tools.server_config import (
    ToolServerOptions,
    resolve_tool_server_options,
)

# ======================================================================
# R1.5: role schema parsing (flat, 5 fields)
# ======================================================================


class TestRoleSchemaParsing:
    def test_flat_dict_coerced(self):
        cfg = LauncherConfig(
            roles={
                "trainer": {
                    "devices": 4,
                    "hardware": "npu",
                    "host_idx": 0,
                },
            },
        )
        role = cfg.roles["trainer"]
        assert isinstance(role, RoleConfig)
        assert role.devices == 4
        assert role.hardware == "npu"
        assert role.host_idx == 0
        assert role.colocate is None
        assert role.extras == {}

    def test_colocate_is_stored(self):
        cfg = LauncherConfig(
            roles={
                "storage": {"devices": 4, "colocate": "trainer"},
            },
        )
        assert cfg.roles["storage"].colocate == "trainer"

    def test_unknown_keys_flow_into_extras(self):
        """Workload-specific knobs (``num_replicas`` / ``timeout`` /
        ``tools`` / ...) aren't first-class resource fields so they
        must flow into ``extras`` for downstream consumers."""
        cfg = LauncherConfig(
            roles={
                "tool_server": {
                    "devices": 0,
                    "num_replicas": 4,
                    "timeout": 30.0,
                    "mesh_name": "tools",
                    "tools": ["python_sandbox"],
                },
            },
        )
        extras = cfg.roles["tool_server"].extras
        assert extras == {
            "num_replicas": 4,
            "timeout": 30.0,
            "mesh_name": "tools",
            "tools": ["python_sandbox"],
        }

    def test_explicit_extras_block_merges(self):
        """Caller-authored ``extras:`` merges on top of loose fields;
        tests that round-trip through YAML composition stay stable."""
        cfg = LauncherConfig(
            roles={
                "tool_server": {
                    "devices": 0,
                    "num_replicas": 2,  # loose
                    "extras": {"timeout": 60.0},  # explicit
                },
            },
        )
        assert cfg.roles["tool_server"].extras == {
            "num_replicas": 2,
            "timeout": 60.0,
        }

    def test_legacy_nested_hardware_block_flattens(self):
        """R1's short-lived nested ``hardware: {type: npu}`` shape
        must round-trip into the flat ``hardware: npu`` string.
        This keeps in-flight YAMLs working across the rename."""
        cfg = LauncherConfig(
            roles={
                "trainer": {
                    "devices": 4,
                    "hardware": {"type": "npu", "model": "910b"},
                },
            },
        )
        assert cfg.roles["trainer"].hardware == "npu"

    def test_legacy_nested_placement_block_flattens(self):
        """Legacy ``placement: {host_idx: 0, colocate_with: X}``
        flattens into ``host_idx`` / ``colocate``."""
        cfg = LauncherConfig(
            roles={
                "storage": {
                    "devices": 4,
                    "placement": {"host_idx": 1, "colocate_with": "trainer"},
                },
            },
        )
        role = cfg.roles["storage"]
        assert role.host_idx == 1
        assert role.colocate == "trainer"

    def test_empty_config_stays_empty(self):
        cfg = LauncherConfig()
        assert cfg.roles == {}
        assert cfg.meshes == {}

    def test_devices_must_be_non_negative(self):
        with pytest.raises(ValueError, match="devices must be >= 0"):
            LauncherConfig(roles={"trainer": {"devices": -1}})

    def test_hardware_tier_is_whitelisted(self):
        """Silently accepting unknown hardware tiers would let YAMLs
        pass that have no corresponding launcher path -- fail fast."""
        with pytest.raises(ValueError, match="hardware="):
            LauncherConfig(roles={"trainer": {"devices": 4, "hardware": "gpu"}})


# ======================================================================
# R1.5: roles <-> meshes bridge
# ======================================================================


class TestRolesMeshesBridge:
    def test_forward_bridge_host_idx(self):
        cfg = LauncherConfig(
            roles={
                "trainer": {"devices": 4, "host_idx": 0},
                "generator": {"devices": 4, "host_idx": 1},
            },
        )
        assert cfg.meshes == {
            "trainer": {"host_idx": 0},
            "generator": {"host_idx": 1},
        }

    def test_reverse_bridge_from_meshes(self):
        cfg = LauncherConfig(meshes={"trainer": 0, "generator": 1})
        assert "trainer" in cfg.roles
        assert cfg.roles["trainer"].host_idx == 0
        assert cfg.roles["generator"].host_idx == 1

    def test_mixed_authorship_bridges_both_ways(self):
        cfg = LauncherConfig(
            roles={"trainer": {"devices": 4, "host_idx": 0}},
            meshes={"generator": 1},
        )
        assert cfg.meshes == {"generator": 1, "trainer": {"host_idx": 0}}
        assert set(cfg.roles) == {"trainer", "generator"}
        assert cfg.roles["trainer"].host_idx == 0
        assert cfg.roles["generator"].host_idx == 1

    def test_conflict_leaves_both_sides_intact(self):
        """User authored both sides with different values -- we
        refuse to guess which one wins, we trust what the user
        wrote on each side."""
        cfg = LauncherConfig(
            roles={"trainer": {"devices": 4, "host_idx": 0}},
            meshes={"trainer": 5},
        )
        assert cfg.meshes["trainer"] == 5
        assert cfg.roles["trainer"].host_idx == 0

    def test_role_without_host_idx_skips_forward_bridge(self):
        """Pool-scheduled roles (no explicit ``host_idx``) must not
        pollute ``meshes``: the launcher would then try to index
        with ``None``.  Such roles rely on the devices-based
        scheduler (R1.5c) for placement."""
        cfg = LauncherConfig(
            roles={
                "pinned": {"devices": 4, "host_idx": 0},
                "floating": {"devices": 4},
            },
        )
        assert "pinned" in cfg.meshes
        assert "floating" not in cfg.meshes
        assert cfg.roles["floating"].host_idx is None


# ======================================================================
# R1.5: storage_role / storage_mesh deprecation
# ======================================================================


class TestStorageRoleDeprecation:
    def test_storage_role_alone_is_canonical(self):
        cfg = LauncherConfig(
            roles={"storage": {"devices": 4}},
            weight_sync={"storage_role": "storage"},
        )
        assert cfg.weight_sync.storage_role == "storage"

    def test_storage_mesh_alone_copies_and_warns(self):
        with pytest.warns(DeprecationWarning, match="storage_mesh is deprecated"):
            cfg = LauncherConfig(
                roles={"storage": {"devices": 4}},
                weight_sync={"storage_mesh": "storage"},
            )
        assert cfg.weight_sync.storage_role == "storage"
        # Legacy readers keep seeing a value in the old field too.
        assert cfg.weight_sync.storage_mesh == "storage"

    def test_both_set_identical_is_idempotent(self):
        with pytest.warns(DeprecationWarning):
            cfg = LauncherConfig(
                roles={"storage": {"devices": 4}},
                weight_sync={
                    "storage_role": "storage",
                    "storage_mesh": "storage",
                },
            )
        assert cfg.weight_sync.storage_role == "storage"

    def test_both_set_conflicting_raises(self):
        with pytest.raises(ValueError, match="conflicts with"):
            LauncherConfig(
                roles={"storage": {"devices": 4}, "trainer": {"devices": 4}},
                weight_sync={
                    "storage_role": "storage",
                    "storage_mesh": "trainer",
                },
            )


# ======================================================================
# R1.5: cross-validation (storage_role must reference a real role)
# ======================================================================


class TestCrossValidation:
    def test_unknown_storage_role_raises(self):
        with pytest.raises(ValueError, match="does not match any entry"):
            LauncherConfig(
                roles={"trainer": {"devices": 4}},
                weight_sync={"storage_role": "nonexistent"},
            )

    def test_storage_role_resolvable_via_meshes(self):
        """Legacy YAMLs that authored ``meshes`` but not ``roles``
        still get credit for the mesh entry through the reverse
        bridge.  A ``storage_role`` that names a mesh passes
        validation because the bridge synthesized a matching role."""
        cfg = LauncherConfig(
            meshes={"storage": 0},
            weight_sync={"storage_role": "storage"},
        )
        assert cfg.weight_sync.storage_role == "storage"

    def test_validation_skipped_when_no_roles_authored(self):
        """If neither ``roles`` nor ``meshes`` was authored, we don't
        have enough information to validate references -- leave it
        to the Provisioner to surface a clearer runtime error."""
        cfg = LauncherConfig(weight_sync={"storage_role": "whatever"})
        assert cfg.weight_sync.storage_role == "whatever"


# ======================================================================
# A2: tool-server resolver (reads role.extras now)
# ======================================================================


class TestToolServerResolver:
    def test_no_launcher_config_returns_default(self):
        opts = resolve_tool_server_options(None)
        assert isinstance(opts, ToolServerOptions)
        assert opts.enabled is True
        assert opts.procs == 1
        assert opts.num_replicas == 1
        assert opts.as_service is False
        assert opts.mesh_name == "sandbox"
        assert opts.tool_types == ("python_sandbox",)

    def test_missing_tool_server_role_falls_back(self):
        cfg = LauncherConfig(roles={"trainer": {"devices": 4}})
        opts = resolve_tool_server_options(cfg)
        assert opts.procs == 1
        assert opts._source == "default:no-role-entry"

    def test_procs_and_replicas_from_extras(self):
        cfg = LauncherConfig(
            roles={
                "tool_server": {
                    "devices": 0,
                    "procs": 2,
                    "num_replicas": 4,
                    "mesh_name": "tools",
                },
            },
        )
        opts = resolve_tool_server_options(cfg)
        assert opts.procs == 2
        assert opts.num_replicas == 4
        assert opts.as_service is True
        assert opts.mesh_name == "tools"

    def test_replicas_from_count_extra(self):
        """``count`` in extras is treated as replica count -- this
        matches the legacy ``placement.count`` semantic the resolver
        used under the R1 nested schema."""
        cfg = LauncherConfig(
            roles={
                "tool_server": {
                    "devices": 0,
                    "count": 3,
                    "host_idx": 2,
                },
            },
        )
        opts = resolve_tool_server_options(cfg)
        assert opts.num_replicas == 3
        assert opts.host_idx == 2

    def test_tools_list_overrides_default(self):
        cfg = LauncherConfig(
            roles={
                "tool_server": {
                    "devices": 0,
                    "tools": ["python_sandbox", "browser"],
                },
            },
        )
        opts = resolve_tool_server_options(cfg)
        assert opts.tool_types == ("python_sandbox", "browser")

    def test_tool_types_string_fallback(self):
        cfg = LauncherConfig(
            roles={
                "tool_server": {
                    "devices": 0,
                    "tool_types": "python_sandbox",
                },
            },
        )
        opts = resolve_tool_server_options(cfg)
        assert opts.tool_types == ("python_sandbox",)

    def test_disabled_flag(self):
        cfg = LauncherConfig(
            roles={"tool_server": {"devices": 0, "enabled": False}},
        )
        opts = resolve_tool_server_options(cfg)
        assert opts.enabled is False

    def test_mesh_name_defaults_to_role_name(self):
        cfg = LauncherConfig(
            roles={"tool_server": {"devices": 0, "host_idx": 2}},
        )
        opts = resolve_tool_server_options(cfg)
        assert opts.mesh_name == "tool_server"
        assert opts.host_idx == 2

    def test_custom_role_name_lookup(self):
        cfg = LauncherConfig(
            roles={"code_runner": {"devices": 0, "num_replicas": 2}},
        )
        opts = resolve_tool_server_options(cfg, role_name="code_runner")
        assert opts.num_replicas == 2
        assert opts.mesh_name == "code_runner"

    def test_pristine_extras_after_resolution(self):
        cfg = LauncherConfig(
            roles={
                "tool_server": {
                    "devices": 0,
                    "procs": 1,
                    "num_replicas": 4,
                    "mesh_name": "custom",
                    "enabled": True,
                    "timeout": 60.0,
                    "max_output_len": 8192,
                },
            },
        )
        opts = resolve_tool_server_options(cfg)
        assert "num_replicas" not in opts.extras
        assert "mesh_name" not in opts.extras
        assert "enabled" not in opts.extras
        assert "procs" not in opts.extras
        assert opts.extras == {"timeout": 60.0, "max_output_len": 8192}


# ======================================================================
# A2-full: tool registry (decorator + ensure_loaded + error UX)
# ======================================================================


class TestToolRegistry:
    def setup_method(self):
        from forge.tools import reset_for_tests

        reset_for_tests()

    def teardown_method(self):
        from forge.tools import reset_for_tests

        reset_for_tests()

    def test_decorator_registers_and_gets(self):
        from forge.tools import get_tool, register_tool

        @register_tool("my_cool_tool")
        class _MyCoolTool:
            name = "my_cool_tool"

        assert get_tool("my_cool_tool") is _MyCoolTool

    def test_duplicate_registration_raises(self):
        from forge.tools import register_tool

        @register_tool("dupe")
        class _First:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_tool("dupe")
            class _Second:
                pass

    def test_unknown_tool_error_message_has_actionable_hints(self):
        from forge.tools import get_tool

        with pytest.raises(ValueError) as exc:
            get_tool("nope_not_a_tool")
        msg = str(exc.value)
        assert "nope_not_a_tool" in msg
        assert "Available tools" in msg
        assert "list-tools" in msg

    def test_ensure_loaded_finds_python_sandbox(self):
        import importlib

        import forge.tools.python_sandbox as _ps

        importlib.reload(_ps)

        from forge.tools import available_tools, ensure_loaded

        ensure_loaded()
        assert "python_sandbox" in available_tools()


# Ensure we never accidentally drop WeightSyncBlock from the public API.
def test_weight_sync_block_exported():
    ws = WeightSyncBlock()
    assert ws.storage_role is None
    assert ws.storage_mesh is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
