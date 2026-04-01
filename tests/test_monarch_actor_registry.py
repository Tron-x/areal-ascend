"""Unit tests for ActorRegistry: topological sort, reference resolution, rollback.

All tests are pure Python -- no GPU or Monarch runtime required.
Monarch is mocked via in-file sys.modules injection.
"""

import sys
import types
import unittest
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
sys.modules["monarch.actor"] = _mock_monarch_actor
sys.modules["monarch"] = types.ModuleType("monarch")
sys.modules["monarch._src"] = types.ModuleType("monarch._src")
sys.modules["monarch._src.actor"] = types.ModuleType("monarch._src.actor")
sys.modules["monarch._src.actor.actor_mesh"] = types.ModuleType("monarch._src.actor.actor_mesh")
sys.modules["monarch._src.actor.bootstrap"] = types.ModuleType("monarch._src.actor.bootstrap")

# Now import from areal.monarch_plugin
from areal.monarch_plugin.actor_registry import ActorRegistry
from areal.monarch_plugin.actor_spec import (
    ActorContext,
    ActorRef,
    CtxRef,
    ResourceKind,
)
from areal.monarch_plugin.actor_base import MonarchActor


# ---------------------------------------------------------------------------
# Dummy actor classes for testing
# ---------------------------------------------------------------------------


class DummyActor(MonarchActor):
    resource = ResourceKind.CPU
    dependencies: list[str] = []


class DummyA(DummyActor):
    pass


class DummyB(DummyActor):
    dependencies = ["dummy_a"]


class DummyC(DummyActor):
    dependencies = ["dummy_a", "dummy_b"]


def _ctx(**extra) -> ActorContext:
    """Build a context with extra dict populated from kwargs."""
    return ActorContext(
        config=MagicMock(),
        alloc_mode=MagicMock(),
        placement=MagicMock(),
        host=MagicMock(),
        extra=extra,
    )


# ===========================================================================
# Topological sort
# ===========================================================================


class TestTopoSort(unittest.TestCase):
    def test_correct_order(self):
        """Dependencies come before dependents."""
        reg = ActorRegistry([DummyC, DummyA, DummyB])
        order = reg._topo_sort()
        self.assertLess(order.index("dummy_a"), order.index("dummy_b"))
        self.assertLess(order.index("dummy_a"), order.index("dummy_c"))
        self.assertLess(order.index("dummy_b"), order.index("dummy_c"))

    def test_no_deps_any_order(self):
        """No dependencies: all valid permutations accepted."""
        reg = ActorRegistry([DummyA, DummyB, DummyActor])
        order = reg._topo_sort()
        self.assertEqual(set(order), {"dummy_a", "dummy_b", "dummy"})

    def test_circular_dependency_raises(self):
        class CircA(DummyActor):
            dependencies = ["circ_c"]

        class CircC(DummyActor):
            dependencies = ["circ_a"]

        reg = ActorRegistry([CircA, CircC])
        with self.assertRaises(ValueError):
            reg._topo_sort()

    def test_missing_dependency_raises(self):
        class DepA(DummyActor):
            dependencies = ["nonexistent"]

        reg = ActorRegistry([DepA])
        with self.assertRaises(ValueError):
            reg._topo_sort()

    def test_duplicate_names_rejected(self):
        class Actor1(DummyActor):
            @classmethod
            def actor_name(cls):
                return "same_name"

        class Actor2(DummyActor):
            @classmethod
            def actor_name(cls):
                return "same_name"

        with self.assertRaises(ValueError):
            ActorRegistry([Actor1, Actor2])


# ===========================================================================
# Reference resolution
# ===========================================================================


class TestRefResolution(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx(host="host_mesh_obj", port=12345)
        self.ctx.actors["generator"] = MagicMock(name="generator_ref")

    def test_actor_ref_resolved(self):
        result = MonarchActor._resolve_refs(
            {"gen": ActorRef("generator")}, self.ctx
        )
        self.assertEqual(result["gen"], self.ctx.actors["generator"])

    def test_actor_ref_not_found(self):
        with self.assertRaises(KeyError):
            MonarchActor._resolve_refs({"gen": ActorRef("missing")}, self.ctx)
    def test_ctx_ref(self):
        result = MonarchActor._resolve_refs(
            {"mesh": CtxRef("host")}, self.ctx
        )
        self.assertEqual(result["mesh"], "host_mesh_obj")

    def test_mixed_args(self):
        result = MonarchActor._resolve_refs(
            {
                "gen": ActorRef("generator"),
                "mesh": CtxRef("host"),
                "plain": 42,
                "text": "hello",
            },
            self.ctx,
        )
        self.assertEqual(result["gen"], self.ctx.actors["generator"])
        self.assertEqual(result["mesh"], "host_mesh_obj")
        self.assertEqual(result["plain"], 42)
        self.assertEqual(result["text"], "hello")

    def test_plain_dict_passthrough(self):
        args = {"a": 1, "b": "two", "c": [3]}
        result = MonarchActor._resolve_refs(args, self.ctx)
        self.assertEqual(result, args)

    def test_extra_actors_resolved(self):
        self.ctx.extra_actors["worker_registry"] = MagicMock(name="wr_ref")
        result = MonarchActor._resolve_refs(
            {"wr": ActorRef("worker_registry")}, self.ctx
        )
        self.assertEqual(result["wr"], self.ctx.extra_actors["worker_registry"])


# ===========================================================================
# Actor naming
# ===========================================================================


class TestActorNaming(unittest.TestCase):
    def test_snake_case_conversion(self):
        self.assertEqual(DummyA.actor_name(), "dummy_a")
        self.assertEqual(DummyB.actor_name(), "dummy_b")
        # _to_snake strips _Actor suffix: DummyActor -> "dummy"
        self.assertEqual(DummyActor.actor_name(), "dummy")


if __name__ == "__main__":
    unittest.main()
