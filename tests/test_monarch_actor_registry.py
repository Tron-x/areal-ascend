"""Unit tests for ActorRegistry: topological sort, reference resolution, rollback.

All tests are pure Python — no GPU or Monarch runtime required.
"""

import unittest
from unittest.mock import MagicMock

from areal.monarch_plugin.actor_registry import ActorRegistry
from areal.monarch_plugin.actor_spec import (
    ActorContext,
    ActorRef,
    ActorSpec,
    CtxRef,
    ResourceKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_spec(
    name: str,
    *,
    deps: list[str] | None = None,
    resource: ResourceKind = ResourceKind.CPU,
    constructor_args=None,
    nprocs: int = 1,
) -> ActorSpec:
    """Build a minimal ActorSpec for testing."""
    return ActorSpec(
        name=name,
        actor_class=type(f"Dummy{name.title()}", (), {}),
        resource=resource,
        bootstrap_factory=lambda: lambda: None,
        constructor_args=constructor_args or {},
        dependencies=deps or [],
        nprocs=nprocs,
    )


def _ctx(**extra) -> ActorContext:
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
        specs = [
            _dummy_spec("c", deps=["a", "b"]),
            _dummy_spec("a"),
            _dummy_spec("b", deps=["a"]),
        ]
        reg = ActorRegistry(specs)
        order = reg._topo_sort()
        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("a"), order.index("c"))
        self.assertLess(order.index("b"), order.index("c"))

    def test_no_deps_any_order(self):
        """No dependencies: all valid permutations accepted."""
        names = ["x", "y", "z"]
        specs = [_dummy_spec(n) for n in names]
        reg = ActorRegistry(specs)
        order = reg._topo_sort()
        self.assertEqual(set(order), set(names))

    def test_circular_dependency_raises(self):
        specs = [
            _dummy_spec("a", deps=["b"]),
            _dummy_spec("b", deps=["c"]),
            _dummy_spec("c", deps=["a"]),
        ]
        reg = ActorRegistry(specs)
        with self.assertRaises(ValueError):
            reg._topo_sort()

    def test_missing_dependency_raises(self):
        specs = [_dummy_spec("a", deps=["nonexistent"])]
        reg = ActorRegistry(specs)
        with self.assertRaises(ValueError):
            reg._topo_sort()

    def test_duplicate_names_rejected(self):
        specs = [_dummy_spec("a"), _dummy_spec("a")]
        with self.assertRaises(ValueError):
            ActorRegistry(specs)


# ===========================================================================
# Reference resolution
# ===========================================================================


class TestRefResolution(unittest.TestCase):
    def setUp(self):
        self.specs = [
            _dummy_spec("generator"),
            _dummy_spec("reward"),
            _dummy_spec("trainer", deps=["generator", "reward"]),
        ]
        self.reg = ActorRegistry(self.specs)
        # Simulate already-spawned actors
        gen_mock = MagicMock(name="generator_ref")
        reward_mock = MagicMock(name="reward_ref")
        self.reg._actors = {"generator": gen_mock, "reward": reward_mock}
        self.ctx = _ctx(host="host_mesh_obj", port=12345)

    def test_actor_ref(self):
        result = self.reg._resolve_refs({"gen": ActorRef("generator")}, self.ctx)
        self.assertEqual(result["gen"], self.reg._actors["generator"])

    def test_ctx_ref(self):
        result = self.reg._resolve_refs({"mesh": CtxRef("host")}, self.ctx)
        self.assertEqual(result["mesh"], "host_mesh_obj")

    def test_mixed_args(self):
        result = self.reg._resolve_refs(
            {
                "gen": ActorRef("generator"),
                "mesh": CtxRef("host"),
                "plain": 42,
                "text": "hello",
            },
            self.ctx,
        )
        self.assertEqual(result["gen"], self.reg._actors["generator"])
        self.assertEqual(result["mesh"], "host_mesh_obj")
        self.assertEqual(result["plain"], 42)
        self.assertEqual(result["text"], "hello")

    def test_plain_dict_passthrough(self):
        args = {"a": 1, "b": "two", "c": [3]}
        result = self.reg._resolve_refs(args, self.ctx)
        self.assertEqual(result, args)

    def test_resolve_args_dict(self):
        result = self.reg._resolve_args(
            {"gen": ActorRef("generator")}, self.ctx
        )
        self.assertEqual(result["gen"], self.reg._actors["generator"])

    def test_resolve_args_callable(self):
        """Callable path still works (backward compat)."""
        self.ctx.actors = dict(self.reg._actors)
        result = self.reg._resolve_args(
            lambda ctx: {"gen": ctx.actors["generator"]}, self.ctx
        )
        self.assertEqual(result["gen"], self.reg._actors["generator"])

    def test_resolve_args_none(self):
        result = self.reg._resolve_args(None, self.ctx)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
