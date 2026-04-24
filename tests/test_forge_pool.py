"""Unit tests for the R1.5c pool-based placement scheduler.

Covers:

* ``PoolHost`` validation (host non-empty, n_devices >= 0, role tier)
* ``_normalize_pool`` list coercion and error paths
* ``schedule_roles_on_pool`` greedy algorithm:
    - independents placed largest-first
    - colocate dependents pinned to their anchor's host
    - legacy ``host_idx`` pins preserved, budget debited
    - not-enough-devices -> ValueError with actionable message
    - colocate cycles -> ValueError
* ``pool_driver_idx`` behavior (default, explicit, ambiguous)
* ``LauncherConfig`` end-to-end: pool -> workers + scheduler ->
  ``meshes.<role>.host_idx``
"""

from __future__ import annotations

import pytest

from forge.core.types import (
    LauncherConfig,
    PoolHost,
    RoleConfig,
    pool_driver_idx,
    schedule_roles_on_pool,
)

# ======================================================================
# PoolHost validation
# ======================================================================


class TestPoolHost:
    def test_minimal_fields(self):
        h = PoolHost(host="10.0.0.1", n_devices=8)
        assert h.host == "10.0.0.1"
        assert h.port == 22222
        assert h.hardware == "npu"
        assert h.n_devices == 8
        assert h.role is None

    def test_driver_tag_accepted(self):
        h = PoolHost(host="10.0.0.1", n_devices=8, role="driver")
        assert h.role == "driver"

    def test_empty_host_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            PoolHost(host="", n_devices=8)

    def test_negative_devices_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            PoolHost(host="x", n_devices=-1)

    def test_unknown_role_tag_raises(self):
        with pytest.raises(ValueError, match="not a recognized tag"):
            PoolHost(host="x", n_devices=8, role="worker")


# ======================================================================
# pool_driver_idx
# ======================================================================


class TestDriverIdx:
    def test_default_is_zero(self):
        pool = [PoolHost(host="a", n_devices=8), PoolHost(host="b", n_devices=8)]
        assert pool_driver_idx(pool) == 0

    def test_explicit_tag(self):
        pool = [
            PoolHost(host="a", n_devices=8),
            PoolHost(host="b", n_devices=8, role="driver"),
        ]
        assert pool_driver_idx(pool) == 1

    def test_multiple_tags_raises(self):
        pool = [
            PoolHost(host="a", n_devices=8, role="driver"),
            PoolHost(host="b", n_devices=8, role="driver"),
        ]
        with pytest.raises(ValueError, match="multiple hosts tagged"):
            pool_driver_idx(pool)

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError, match="empty"):
            pool_driver_idx([])


# ======================================================================
# schedule_roles_on_pool -- greedy placement
# ======================================================================


class TestGreedyScheduler:
    def _two_node_8npu_pool(self) -> list[PoolHost]:
        return [
            PoolHost(host="monarch2", n_devices=8),
            PoolHost(host="monarch1", n_devices=8, role="driver"),
        ]

    def test_basic_4_4_placement(self):
        """Two 4-device roles both land on the driver host (has
        capacity 8 >= 4+4).  Driver-first scan order means the
        driver host is tried before anything else for every role
        in pass 1, so greedy packing concentrates on it.  Users who
        want separation declare colocate or use explicit
        ``host_idx``."""
        pool = self._two_node_8npu_pool()  # host 1 is driver
        roles = {
            "trainer": RoleConfig(devices=4),
            "generator": RoleConfig(devices=4),
        }
        assignment = schedule_roles_on_pool(roles, pool)
        assert assignment == {"trainer": 1, "generator": 1}

    def test_separation_via_full_host_requests(self):
        """When both roles ask for the full host capacity, the
        scheduler naturally spreads them.  The first one picks the
        driver host; the second spills to the non-driver host."""
        pool = self._two_node_8npu_pool()  # host 1 is driver
        roles = {
            "trainer": RoleConfig(devices=8),
            "generator": RoleConfig(devices=8),
        }
        assignment = schedule_roles_on_pool(roles, pool)
        # Tie-break on devices is alphabetical -> ``generator`` is
        # placed first and grabs the driver host.
        assert assignment == {"generator": 1, "trainer": 0}

    def test_colocate_follows_anchor(self):
        """``storage.colocate: trainer`` lands on the same host as
        trainer.  Under strict device semantics the colocated role's
        ``devices`` IS debited from the anchor host's budget
        (trainer=4 + storage=4 = 8, which fits the 8-NPU host)."""
        pool = self._two_node_8npu_pool()  # host 1 is driver
        roles = {
            "trainer": RoleConfig(devices=4),
            "storage": RoleConfig(devices=4, colocate="trainer"),
            "generator": RoleConfig(devices=8),
        }
        assignment = schedule_roles_on_pool(roles, pool)
        # generator (8 devices) goes first and takes the driver host;
        # trainer (4) then takes the non-driver host; storage (4)
        # follows trainer -- non-driver host ends at 0 free.
        assert assignment["generator"] == 1
        assert assignment["trainer"] == 0
        assert assignment["storage"] == 0

    def test_colocate_cpu_sidecar_skips_debit(self):
        """``devices: 0`` lets a CPU-only sidecar (reward eval, tool
        server, ...) ride the colocate without consuming accelerator
        budget."""
        pool = self._two_node_8npu_pool()
        roles = {
            "trainer": RoleConfig(devices=8),
            "reward": RoleConfig(devices=0, colocate="trainer"),
        }
        assignment = schedule_roles_on_pool(roles, pool)
        # trainer fills the driver host; reward colocates with 0
        # debit (otherwise the 8+0 wouldn't overflow anyway, but
        # this verifies the no-op debit path).
        assert assignment["trainer"] == assignment["reward"]

    def test_colocate_overflow_raises(self):
        """Strict semantics: colocate + devices > anchor host free
        is a hard error, not a silent oversubscription.  The
        effective-footprint pass (1) catches this at scheduling
        time, not at dependent-placement time."""
        pool = self._two_node_8npu_pool()
        roles = {
            "trainer": RoleConfig(devices=8),  # alone fills the host
            "storage": RoleConfig(devices=4, colocate="trainer"),
        }
        # trainer's effective footprint is 8+4=12, no 12-device host.
        with pytest.raises(ValueError, match="needs 12 'npu' device"):
            schedule_roles_on_pool(roles, pool)

    def test_colocate_pass2_overflow_with_legacy_pin(self):
        """Legacy ``host_idx`` pinning skips the effective-footprint
        check at pass 1 because the anchor is pre-placed.  Verify
        pass 2's own overflow check still fires -- the contract is
        "no silent oversubscription" regardless of which path the
        role takes."""
        pool = self._two_node_8npu_pool()
        roles = {
            # Legacy pin: trainer explicitly on host 0, consuming all 8.
            "trainer": RoleConfig(devices=8, host_idx=0),
            # Colocate a dependent that can't fit there.
            "storage": RoleConfig(devices=4, colocate="trainer"),
        }
        with pytest.raises(ValueError, match="colocates with 'trainer'"):
            schedule_roles_on_pool(roles, pool)

    def test_legacy_host_idx_is_honored(self):
        """Explicit pins win over greedy.  Budget is debited so
        subsequent roles see the correct capacity."""
        pool = self._two_node_8npu_pool()
        roles = {
            "trainer": RoleConfig(devices=4, host_idx=1),
            "generator": RoleConfig(devices=8),  # must go to host 0
        }
        assignment = schedule_roles_on_pool(roles, pool)
        assert assignment["trainer"] == 1
        assert assignment["generator"] == 0

    def test_impossible_placement_raises(self):
        pool = [PoolHost(host="a", n_devices=4)]
        roles = {"trainer": RoleConfig(devices=8)}
        with pytest.raises(ValueError, match="no pool host has that much free"):
            schedule_roles_on_pool(roles, pool)

    def test_pin_exceeds_capacity_raises(self):
        pool = [PoolHost(host="a", n_devices=4)]
        roles = {"trainer": RoleConfig(devices=8, host_idx=0)}
        with pytest.raises(ValueError, match="asks for 8 devices"):
            schedule_roles_on_pool(roles, pool)

    def test_pin_out_of_range_raises(self):
        pool = [PoolHost(host="a", n_devices=8)]
        roles = {"trainer": RoleConfig(devices=4, host_idx=5)}
        with pytest.raises(ValueError, match="out of range"):
            schedule_roles_on_pool(roles, pool)

    def test_unresolvable_colocate_raises(self):
        pool = [PoolHost(host="a", n_devices=8)]
        roles = {"storage": RoleConfig(devices=4, colocate="trainer")}
        # trainer never declared -> storage's anchor never resolves
        with pytest.raises(ValueError, match="colocate chain"):
            schedule_roles_on_pool(roles, pool)

    def test_chained_colocate(self):
        """A -> B -> C: A colocates with B, B colocates with C, C is
        independent.  Scheduler must resolve C first, then B, then A,
        via its fixed-point pass."""
        pool = [PoolHost(host="a", n_devices=8), PoolHost(host="b", n_devices=8)]
        roles = {
            "c": RoleConfig(devices=4),
            "b": RoleConfig(devices=2, colocate="c"),
            "a": RoleConfig(devices=1, colocate="b"),
        }
        assignment = schedule_roles_on_pool(roles, pool)
        assert assignment["a"] == assignment["b"] == assignment["c"]

    def test_hardware_mismatch_skipped(self):
        """A role asking for a hardware tier no host provides can't
        be placed (once we have multiple tiers); the error message
        should be actionable."""
        # Forge the ``hardware`` attribute by constructing PoolHost
        # without validation via object.__setattr__ -- the dataclass
        # won't let us set ``hardware="gpu"`` because RoleConfig
        # whitelists only ``npu`` today.  So we bypass validation to
        # simulate a future multi-tier pool.
        pool = [PoolHost(host="a", n_devices=8)]
        object.__setattr__(pool[0], "hardware", "gpu")
        # Role is ``npu`` (default); no matching host.
        roles = {"trainer": RoleConfig(devices=4)}
        with pytest.raises(ValueError, match="no pool host has that much free"):
            schedule_roles_on_pool(roles, pool)


# ======================================================================
# LauncherConfig -- end-to-end pool + scheduler wiring
# ======================================================================


class TestLauncherConfigPoolIntegration:
    def test_pool_derives_workers(self):
        cfg = LauncherConfig(
            pool=[
                {"host": "10.0.0.1", "port": 22222, "n_devices": 8},
                {"host": "10.0.0.2", "port": 22222, "n_devices": 8},
            ],
        )
        assert cfg.workers == [
            "tcp://10.0.0.1:22222",
            "tcp://10.0.0.2:22222",
        ]
        assert cfg.worker_port == 22222
        assert len(cfg.pool) == 2
        assert isinstance(cfg.pool[0], PoolHost)

    def test_pool_with_roles_populates_meshes(self):
        """End-to-end: declare a pool + roles, the scheduler + bridge
        work together to produce a ``meshes`` dict that the
        provisioner understands.  Under strict device semantics
        the colocated ``storage`` also consumes accelerator budget
        on the anchor host, so the YAML must declare real (non-
        overlapping) card counts."""
        cfg = LauncherConfig(
            pool=[
                {"host": "h1", "n_devices": 8},
                {"host": "h2", "n_devices": 8},
            ],
            roles={
                "trainer": {"devices": 4},
                "generator": {"devices": 8},
                "storage": {"devices": 4, "colocate": "trainer"},
            },
        )
        # No host tagged driver -> defaults to pool[0]=h1.  Driver-
        # first scan places generator (8 devices) on h1 first, then
        # trainer (4) on h2, and storage (4) colocates with trainer.
        assert cfg.meshes["generator"] == {"host_idx": 0}
        assert cfg.meshes["trainer"] == {"host_idx": 1}
        assert cfg.meshes["storage"] == {"host_idx": 1}

    def test_explicit_host_idx_survives_scheduler(self):
        """Hand-authored ``host_idx`` isn't overwritten by the
        greedy pass -- legacy YAMLs keep working."""
        cfg = LauncherConfig(
            pool=[
                {"host": "h1", "n_devices": 8},
                {"host": "h2", "n_devices": 8},
            ],
            roles={
                "trainer": {"devices": 4, "host_idx": 1},
                "generator": {"devices": 4},
            },
        )
        assert cfg.roles["trainer"].host_idx == 1
        assert cfg.meshes["trainer"] == {"host_idx": 1}

    def test_driver_host_idx_default(self):
        cfg = LauncherConfig(
            pool=[{"host": "h1", "n_devices": 8}, {"host": "h2", "n_devices": 8}],
        )
        assert cfg.driver_host_idx() == 0

    def test_driver_host_idx_explicit(self):
        cfg = LauncherConfig(
            pool=[
                {"host": "h1", "n_devices": 8},
                {"host": "h2", "n_devices": 8, "role": "driver"},
            ],
        )
        assert cfg.driver_host_idx() == 1

    def test_no_pool_no_scheduler(self):
        """When ``pool`` is empty, the scheduler is a no-op -- legacy
        configs that only author ``workers`` + ``meshes`` are
        untouched."""
        cfg = LauncherConfig(
            workers=["tcp://h1:22222", "tcp://h2:22222"],
            meshes={"trainer": {"host_idx": 1}},
            roles={"trainer": {"devices": 4}},
        )
        # ``roles.trainer.host_idx`` is still None (no scheduler ran)
        assert cfg.roles["trainer"].host_idx is None
        # But the legacy meshes entry survives.
        assert cfg.meshes["trainer"] == {"host_idx": 1}
        # And the R1.5 bridge reverses meshes -> roles so the role
        # gets an ``host_idx`` from the reverse direction.
        # (Actually NOT in this case because the user authored both
        # sides; the conflict rule leaves both intact.)

    def test_conflicting_workers_and_pool_warns(self):
        """User-authored ``workers`` that disagrees with the pool is
        a stale override -- warn but don't crash."""
        with pytest.warns(UserWarning, match="disagree"):
            LauncherConfig(
                pool=[
                    {"host": "h1", "n_devices": 8},
                    {"host": "h2", "n_devices": 8},
                ],
                workers=["tcp://stale:22222"],  # 1 entry vs pool's 2
            )

    def test_impossible_schedule_raises_with_hint(self):
        with pytest.raises(ValueError, match="Edit ``launcher.pool``"):
            LauncherConfig(
                pool=[{"host": "h1", "n_devices": 4}],
                roles={"trainer": {"devices": 8}},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
