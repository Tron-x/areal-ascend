"""Unit tests for MonarchActor declarative class metadata.

All tests are pure Python -- no GPU or Monarch runtime required.
Monarch mocked via in-file sys.modules injection.
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
sys.modules["monarch._src.actor.bootstrap"].attach_to_workers = lambda **kw: None
sys.modules["monarch._src.actor.bootstrap"].run_worker_loop_forever = lambda **kw: None

# Now import from areal.monarch_plugin
from areal.api import AllocationType
from areal.monarch_plugin.actor_registry import ActorRegistry
from areal.monarch_plugin.actor_spec import (
    ActorContext,
    ActorRef,
    CtxRef,
    ResourceKind,
)
from areal.monarch_plugin.agent_actor import AgentActor
from areal.monarch_plugin.generator_actor import GeneratorActor
from areal.monarch_plugin.replay_buffer_actor import ReplayBufferActor
from areal.monarch_plugin.reward_actor import RewardActor
from areal.monarch_plugin.rollout_actor import RolloutActor
from areal.monarch_plugin.sandbox_actor import SandboxActor
from areal.monarch_plugin.trainer_actor import TrainerActor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_ctx(nprocs=4):
    alloc_mode = MagicMock()
    alloc_mode.type_ = AllocationType.COLOCATE
    alloc_mode.gen = MagicMock(tp_size=1, pp_size=1, world_size=1)
    alloc_mode.train = MagicMock(world_size=nprocs)
    placement = MagicMock()
    placement.inference = MagicMock(all_device_ids=["0", "1", "2", "3"])
    placement.training = MagicMock(all_device_ids=["4", "5", "6", "7"])
    placement.master_addr = "127.0.0.1"

    return ActorContext(
        config=MagicMock(
            vllm=MagicMock(),
            rollout=MagicMock(),
            actor=MagicMock(scheduling_spec=[]),
            get=MagicMock(return_value=False),
        ),
        alloc_mode=alloc_mode,
        placement=placement,
        host=MagicMock(),
        master_port=29500,
        extra={"host": MagicMock(), "env_var": "ASCEND_RT_VISIBLE_DEVICES"},
    )


# ===========================================================================
# Actor names
# ===========================================================================


class TestActorNames(unittest.TestCase):
    def test_generator_name(self):
        self.assertEqual(GeneratorActor.actor_name(), "generator")

    def test_reward_name(self):
        self.assertEqual(RewardActor.actor_name(), "reward")

    def test_sandbox_name(self):
        self.assertEqual(SandboxActor.actor_name(), "sandbox")

    def test_agent_name(self):
        self.assertEqual(AgentActor.actor_name(), "agent")

    def test_replay_buffer_name(self):
        self.assertEqual(ReplayBufferActor.actor_name(), "replay_buffer")

    def test_rollout_name(self):
        self.assertEqual(RolloutActor.actor_name(), "rollout")

    def test_trainer_name(self):
        self.assertEqual(TrainerActor.actor_name(), "trainer")


# ===========================================================================
# Resource declarations
# ===========================================================================


class TestResourceDeclarations(unittest.TestCase):
    def test_generator_npu_single(self):
        self.assertEqual(GeneratorActor.resource, ResourceKind.NPU_SINGLE)

    def test_cpu_actors(self):
        for cls in [RewardActor, SandboxActor, AgentActor, ReplayBufferActor, RolloutActor]:
            self.assertEqual(cls.resource, ResourceKind.CPU)

    def test_trainer_dynamic_multi(self):
        ctx = _make_mock_ctx(nprocs=4)
        self.assertEqual(TrainerActor.resolve_resource(ctx), ResourceKind.NPU_MULTI)

    def test_trainer_dynamic_single(self):
        ctx = _make_mock_ctx(nprocs=1)
        self.assertEqual(TrainerActor.resolve_resource(ctx), ResourceKind.NPU_SINGLE)


# ===========================================================================
# Dependencies
# ===========================================================================


class TestDependencies(unittest.TestCase):
    def test_no_deps(self):
        for cls in [GeneratorActor, RewardActor, SandboxActor, ReplayBufferActor]:
            self.assertEqual(cls.dependencies, [])

    def test_agent_deps(self):
        self.assertEqual(AgentActor.dependencies, ["generator", "sandbox", "reward"])

    def test_rollout_deps(self):
        self.assertEqual(RolloutActor.dependencies, ["generator", "reward", "agent"])

    def test_trainer_deps(self):
        self.assertEqual(TrainerActor.dependencies, ["generator", "reward", "agent"])


# ===========================================================================
# Init methods
# ===========================================================================


class TestInitMethods(unittest.TestCase):
    def test_generator_init(self):
        self.assertEqual(GeneratorActor.init_method(), "setup")

    def test_no_init(self):
        for cls in [RewardActor, SandboxActor, AgentActor]:
            self.assertIsNone(cls.init_method())

    def test_rollout_init(self):
        self.assertEqual(RolloutActor.init_method(), "setup")

    def test_trainer_init(self):
        self.assertEqual(TrainerActor.init_method(), "initialize")


# ===========================================================================
# Constructor args — ActorRef / CtxRef
# ===========================================================================


class TestConstructorArgs(unittest.TestCase):
    def test_agent_constructor_args(self):
        ctx = _make_mock_ctx()
        args = AgentActor.constructor_args(ctx)
        self.assertIsInstance(args["generator_actor"], ActorRef)
        self.assertIsInstance(args["sandbox_actor"], ActorRef)
        self.assertIsInstance(args["reward_actor"], ActorRef)
        self.assertEqual(args["generator_actor"].name, "generator")
        self.assertEqual(args["sandbox_actor"].name, "sandbox")
        self.assertEqual(args["reward_actor"].name, "reward")

    def test_generator_init_args_ctx_ref(self):
        ctx = _make_mock_ctx()
        init_args = GeneratorActor.init_args(ctx)
        self.assertIsInstance(init_args["host_mesh"], CtxRef)
        self.assertIsInstance(init_args["worker_registry"], ActorRef)
        self.assertEqual(init_args["host_mesh"].key, "host")

    def test_rollout_constructor_args(self):
        ctx = _make_mock_ctx()
        args = RolloutActor.constructor_args(ctx)
        self.assertIsInstance(args["generator_actor"], ActorRef)
        self.assertIsInstance(args["reward_actor"], ActorRef)
        self.assertIsInstance(args["agent_actor"], ActorRef)


# ===========================================================================
# Topological sort with real actor classes
# ===========================================================================


class TestTopoSortFromActors(unittest.TestCase):
    def test_no_circular_deps(self):
        actors = [
            GeneratorActor, RewardActor, SandboxActor,
            AgentActor, ReplayBufferActor, RolloutActor, TrainerActor,
        ]
        reg = ActorRegistry(actors)
        order = reg._topo_sort()
        self.assertEqual(len(order), 7)

    def test_spawn_order_respects_deps(self):
        actors = [
            GeneratorActor, RewardActor, SandboxActor,
            AgentActor, ReplayBufferActor, RolloutActor, TrainerActor,
        ]
        reg = ActorRegistry(actors)
        order = reg._topo_sort()
        self.assertLess(order.index("generator"), order.index("trainer"))
        self.assertLess(order.index("reward"), order.index("trainer"))
        self.assertLess(order.index("agent"), order.index("trainer"))
        self.assertLess(order.index("generator"), order.index("agent"))
        self.assertLess(order.index("sandbox"), order.index("agent"))
        self.assertLess(order.index("reward"), order.index("agent"))
        self.assertLess(order.index("generator"), order.index("rollout"))
        self.assertLess(order.index("reward"), order.index("rollout"))
        self.assertLess(order.index("agent"), order.index("rollout"))


if __name__ == "__main__":
    unittest.main()
