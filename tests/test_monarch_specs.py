"""Unit tests for make_actor_specs: declarative spec building.

All tests are pure Python — no GPU or Monarch runtime required.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from areal.api import AllocationMode, AllocationType
from areal.monarch_plugin.actor_registry import ActorRegistry
from areal.monarch_plugin.actor_spec import ActorRef, CtxRef


def _make_mock_placement(
    inf_ids=("0", "1", "2", "3"),
    train_ids=("4", "5", "6", "7"),
):
    placement = MagicMock()
    placement.inference = MagicMock()
    placement.inference.all_device_ids = list(inf_ids)
    placement.training = MagicMock()
    placement.training.all_device_ids = list(train_ids)
    placement.master_addr = "127.0.0.1"
    return placement


def _make_specs(alloc_type=AllocationType.COLOCATE, nprocs=4):
    """Build specs with all heavy deps mocked."""
    from areal.monarch_plugin.specs import make_actor_specs

    alloc_mode = MagicMock()
    alloc_mode.type_ = alloc_type
    alloc_mode.gen = MagicMock(tp_size=1, pp_size=1, world_size=1)
    alloc_mode.train = MagicMock(world_size=nprocs)

    placement = _make_mock_placement()

    with patch("areal.monarch_plugin.specs.to_structured_cfg") as mock_to_cfg, \
         patch("areal.monarch_plugin.specs.build_vllm_cli_args", return_value=["--model", "test"]), \
         patch("areal.monarch_plugin.specs.get_scheduling_spec") as mock_sched, \
         patch("areal.monarch_plugin.specs.get_thread_env_vars", return_value={}), \
         patch("areal.monarch_plugin.specs.resolve_xccl_alloc_mode", return_value=None):

        mock_to_cfg.side_effect = lambda x, cls: x if x is not None else cls()
        mock_sched.return_value = MagicMock(cpu=4, env_vars={})

        return make_actor_specs(
            config=MagicMock(
                vllm=MagicMock(),
                rollout=MagicMock(),
                actor=MagicMock(scheduling_spec=[]),
                get=MagicMock(return_value=False),
            ),
            alloc_mode=alloc_mode,
            placement=placement,
            master_port=29500,
            env_var="ASCEND_RT_VISIBLE_DEVICES",
            inf_device_ids=list(map(str, range(4))),
            train_device_ids=list(map(str, range(4, 4 + nprocs))),
        )


class TestMakeActorSpecs(unittest.TestCase):
    def test_specs_count_with_trainer(self):
        specs = _make_specs()
        names = [s.name for s in specs]
        self.assertIn("trainer", names)
        self.assertEqual(len(specs), 7)

    def test_specs_count_llm_server_only(self):
        specs = _make_specs(alloc_type=AllocationType.LLM_SERVER_ONLY)
        names = [s.name for s in specs]
        self.assertNotIn("trainer", names)
        self.assertEqual(len(specs), 6)

    def test_trainer_deps(self):
        specs = _make_specs()
        trainer = next(s for s in specs if s.name == "trainer")
        self.assertIn("generator", trainer.dependencies)
        self.assertIn("reward", trainer.dependencies)
        self.assertIn("agent", trainer.dependencies)

    def test_all_deps_resolvable(self):
        specs = _make_specs()
        names = {s.name for s in specs}
        for s in specs:
            for dep in s.dependencies:
                self.assertIn(dep, names, f"'{s.name}' depends on '{dep}'")

    def test_generator_has_post_spawn(self):
        specs = _make_specs()
        gen = next(s for s in specs if s.name == "generator")
        self.assertIsNotNone(gen.post_spawn)

    def test_no_circular_deps(self):
        specs = _make_specs()
        reg = ActorRegistry(specs)
        order = reg._topo_sort()
        self.assertEqual(len(order), len(specs))

    def test_constructor_args_are_dicts(self):
        specs = _make_specs()
        for s in specs:
            self.assertIsInstance(
                s.constructor_args,
                dict,
                f"'{s.name}' constructor_args should be dict, "
                f"got {type(s.constructor_args).__name__}",
            )

    def test_actor_refs_in_constructor_args(self):
        specs = _make_specs()
        agent = next(s for s in specs if s.name == "agent")
        self.assertIsInstance(agent.constructor_args["generator_actor"], ActorRef)
        self.assertIsInstance(agent.constructor_args["sandbox_actor"], ActorRef)
        self.assertIsInstance(agent.constructor_args["reward_actor"], ActorRef)

    def test_ctx_ref_in_generator_init_args(self):
        specs = _make_specs()
        gen = next(s for s in specs if s.name == "generator")
        self.assertIsInstance(gen.init_args["host_mesh"], CtxRef)

    def test_spawn_order_respects_deps(self):
        specs = _make_specs()
        reg = ActorRegistry(specs)
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
