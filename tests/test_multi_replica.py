"""Tests for multi-replica generator support.

Covers:
  - topology.compute_replica_placements
  - GeneratorService routing / fanout / payload adjustment

All tests are pure Python -- no GPU or Monarch runtime required.
Monarch is mocked via sys.modules injection (same pattern as
test_monarch_actor_registry.py).
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock monarch.actor before any areal imports
# ---------------------------------------------------------------------------

_mock_monarch_actor = types.ModuleType("monarch.actor")


class _MockActor:
    pass


def _mock_endpoint(fn):
    return fn


_mock_monarch_actor.Actor = _MockActor
_mock_monarch_actor.endpoint = _mock_endpoint
_mock_monarch_actor.this_host = lambda: MagicMock()
_mock_monarch_actor.this_proc = lambda: MagicMock()

sys.modules.setdefault("monarch", types.ModuleType("monarch"))
sys.modules.setdefault("monarch.actor", _mock_monarch_actor)
sys.modules.setdefault("monarch._src", types.ModuleType("monarch._src"))
sys.modules.setdefault("monarch._src.actor", types.ModuleType("monarch._src.actor"))
sys.modules.setdefault(
    "monarch._src.actor.actor_mesh", types.ModuleType("monarch._src.actor.actor_mesh")
)
sys.modules.setdefault(
    "monarch._src.actor.bootstrap", types.ModuleType("monarch._src.actor.bootstrap")
)
sys.modules.setdefault(
    "monarch._src.actor.future", types.ModuleType("monarch._src.actor.future")
)

# Now import from areal.monarch_plugin  # noqa: E402
from areal.monarch_plugin.generator_service import (  # noqa: E402
    GeneratorService,
    _SyncFuture,
)
from areal.monarch_plugin.topology import (  # noqa: E402
    ClusterTopology,
    DevicePlacement,
    NodeDevices,
    ReplicaPlacement,
    RolePlacement,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_placement(inf_ids: list[str], train_ids: list[str]) -> DevicePlacement:
    local = "127.0.0.1"
    return DevicePlacement(
        inference=RolePlacement([NodeDevices(local, inf_ids)]),
        training=RolePlacement([NodeDevices(local, train_ids)]),
        cpu_services=RolePlacement([NodeDevices(local, [])]),
    )


def _make_topology(inf_ids: list[str], train_ids: list[str]) -> ClusterTopology:
    """Create a ClusterTopology with pre-set placement (bypasses __init__)."""
    topo = ClusterTopology.__new__(ClusterTopology)
    topo._placement = _make_placement(inf_ids, train_ids)
    return topo


class _FakeFuture:
    """Mimics a Monarch Future for testing."""

    def __init__(self, result):
        self._result = result

    def get(self, timeout=None):
        return self._result

    def __await__(self):
        return self._async().__await__()

    async def _async(self):
        return self._result


class _FakeActorRef:
    """Mimics a Monarch actor ref for testing."""

    def __init__(self, name: str, responses: dict[str, Any] | None = None):
        self._name = name
        self._responses = responses or {}

    @property
    def handle_request(self):
        return self

    @property
    def shutdown(self):
        return self

    def call_one(self, ep: str = "", payload: dict | None = None, **kw):
        if ep in self._responses:
            return _FakeFuture(self._responses[ep])
        return _FakeFuture({"success": True, "message": f"{self._name}:{ep}"})


class _CapturingRef(_FakeActorRef):
    """Actor ref that captures payloads sent to it."""

    def __init__(self, name: str):
        super().__init__(name)
        self.captured: list[tuple[str, dict]] = []

    def call_one(self, ep: str = "", payload: dict | None = None, **kw):
        self.captured.append((ep, dict(payload) if payload else {}))
        return _FakeFuture({"success": True, "message": f"{self._name}:{ep}"})


# ===========================================================================
# ReplicaPlacement tests
# ===========================================================================


class TestComputeReplicaPlacements(unittest.TestCase):
    """Tests for ClusterTopology.compute_replica_placements."""

    def test_single_replica_backward_compat(self):
        topo = _make_topology(["0", "1", "2", "3"], ["4", "5"])
        result = topo.compute_replica_placements(num_replicas=1)
        assert len(result) == 1
        assert result[0].actor_name == "generator"
        assert result[0].device_ids == ["0", "1", "2", "3"]
        assert result[0].replica_index == 0

    def test_two_replicas_equal_split(self):
        topo = _make_topology(["0", "1", "2", "3"], ["4", "5"])
        result = topo.compute_replica_placements(num_replicas=2)
        assert len(result) == 2
        assert result[0].actor_name == "generator_0"
        assert result[0].device_ids == ["0", "1"]
        assert result[1].actor_name == "generator_1"
        assert result[1].device_ids == ["2", "3"]

    def test_four_replicas(self):
        topo = _make_topology(["0", "1", "2", "3", "4", "5", "6", "7"], ["8"])
        result = topo.compute_replica_placements(num_replicas=4)
        assert len(result) == 4
        for i, rp in enumerate(result):
            assert rp.replica_index == i
            assert rp.device_ids == [str(2 * i), str(2 * i + 1)]

    def test_uneven_split_raises(self):
        topo = _make_topology(["0", "1", "2"], ["3"])
        with self.assertRaises(ValueError) as ctx:
            topo.compute_replica_placements(num_replicas=2)
        assert "Cannot split" in str(ctx.exception)

    def test_zero_replicas_raises(self):
        topo = _make_topology(["0"], ["1"])
        with self.assertRaises(ValueError) as ctx:
            topo.compute_replica_placements(num_replicas=0)
        assert "num_replicas must be >= 1" in str(ctx.exception)

    def test_eight_devices_four_replicas(self):
        topo = _make_topology(["0", "1", "2", "3", "4", "5", "6", "7"], ["8", "9"])
        result = topo.compute_replica_placements(num_replicas=4)
        assert len(result) == 4
        for i, rp in enumerate(result):
            assert rp.actor_name == f"generator_{i}"
            assert rp.device_ids == [str(2 * i), str(2 * i + 1)]


# ===========================================================================
# GeneratorService tests
# ===========================================================================


class TestGeneratorService(unittest.TestCase):
    """Tests for the GeneratorService proxy."""

    def _make_service(self, num_replicas=2, per_replica_workers=2):
        placements = [
            ReplicaPlacement(i, [str(i)], f"generator_{i}") for i in range(num_replicas)
        ]
        replicas = {
            f"generator_{i}": _FakeActorRef(f"gen_{i}") for i in range(num_replicas)
        }
        return GeneratorService(replicas, placements, per_replica_workers)

    def test_round_robin_inference(self):
        svc = self._make_service(num_replicas=3)
        names = set()
        for _ in range(3):
            fut = svc.handle_request.call_one("/v1/completions", {"prompt": "hi"})
            result = fut.get()
            names.add(result["message"])
        # Each request hit a different replica
        assert len(names) == 3

    def test_round_robin_wraps_around(self):
        svc = self._make_service(num_replicas=2)
        results = []
        for _ in range(4):
            results.append(
                svc.handle_request.call_one("/v1/completions", {}).get()["message"]
            )
        assert results[0] == results[2]  # same replica
        assert results[1] == results[3]  # same replica
        assert results[0] != results[1]  # different replicas

    def test_fanout_weight_sync(self):
        svc = self._make_service(num_replicas=2, per_replica_workers=2)
        result = svc.handle_request.call_one(
            "/areal_set_update_weight_meta", {"names": [], "dtypes": []}
        ).get()
        assert result["success"] is True

    def test_fanout_init_weights_rank_offset(self):
        placements = [
            ReplicaPlacement(0, ["0", "1"], "generator_0"),
            ReplicaPlacement(1, ["2", "3"], "generator_1"),
        ]
        replicas = {
            "generator_0": _CapturingRef("gen_0"),
            "generator_1": _CapturingRef("gen_1"),
        }
        svc = GeneratorService(replicas, placements, per_replica_workers=2)

        svc.handle_request.call_one(
            "/areal_init_weights_update_group",
            {"rank_offset": 4, "world_size": 8},
        ).get()

        captured_0 = replicas["generator_0"].captured
        captured_1 = replicas["generator_1"].captured

        assert len(captured_0) == 1
        assert len(captured_1) == 1
        # Replica 0: rank_offset stays 4
        assert captured_0[0][1]["rank_offset"] == 4
        # Replica 1: rank_offset = 4 + 1 * 2 = 6
        assert captured_1[0][1]["rank_offset"] == 6

    def test_fanout_single_replica_no_adjustment(self):
        placements = [ReplicaPlacement(0, ["0"], "generator")]
        ref = _CapturingRef("gen_0")
        svc = GeneratorService({"generator": ref}, placements, per_replica_workers=1)

        svc.handle_request.call_one(
            "/areal_init_weights_update_group",
            {"rank_offset": 4, "world_size": 5},
        ).get()

        assert ref.captured[0][1]["rank_offset"] == 4  # unchanged

    def test_non_init_payload_not_adjusted(self):
        placements = [
            ReplicaPlacement(0, ["0"], "generator_0"),
            ReplicaPlacement(1, ["1"], "generator_1"),
        ]
        ref0 = _CapturingRef("gen_0")
        ref1 = _CapturingRef("gen_1")
        svc = GeneratorService(
            {"generator_0": ref0, "generator_1": ref1},
            placements,
            per_replica_workers=1,
        )

        svc.handle_request.call_one(
            "/areal_update_weights_xccl", {"model_path": "/tmp/ckpt"}
        ).get()

        # payload passed through unchanged for non-init endpoints
        assert ref0.captured[0][1]["model_path"] == "/tmp/ckpt"
        assert ref1.captured[0][1]["model_path"] == "/tmp/ckpt"

    def test_health_endpoint_fanout(self):
        svc = self._make_service(num_replicas=2)
        result = svc.handle_request.call_one("/health", {}).get()
        assert result["success"] is True

    def test_shutdown_fanout(self):
        svc = self._make_service(num_replicas=2)
        asyncio.run(svc.shutdown.call_one())

    def test_shutdown_broadcast_alias(self):
        svc = self._make_service(num_replicas=2)
        asyncio.run(svc.shutdown.call())

    def test_num_replicas_property(self):
        svc = self._make_service(num_replicas=3)
        assert svc.num_replicas == 3

    def test_replicas_property(self):
        svc = self._make_service(num_replicas=2)
        r = svc.replicas
        assert "generator_0" in r
        assert "generator_1" in r


# ===========================================================================
# _SyncFuture tests
# ===========================================================================


class TestSyncFuture(unittest.TestCase):
    def test_get(self):
        f = _SyncFuture({"ok": True})
        assert f.get() == {"ok": True}

    def test_get_with_timeout(self):
        f = _SyncFuture(42)
        assert f.get(timeout=10) == 42

    def test_await(self):
        f = _SyncFuture("hello")

        async def _check():
            result = await f
            assert result == "hello"

        asyncio.run(_check())


if __name__ == "__main__":
    unittest.main()
