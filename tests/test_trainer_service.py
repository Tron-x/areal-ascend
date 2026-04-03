"""Tests for TrainerService Protocol and FSDPTrainerService.

Covers:
  - TrainerService Protocol conformance
  - FSDPTrainerService lifecycle delegation
  - TrainerActor delegation to service

All tests are pure Python -- no GPU or Monarch runtime required.
Monarch is mocked via sys.modules injection.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

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

# Now import from areal.monarch_plugin  # noqa: E402
from areal.monarch_plugin.trainer_actor import TrainerActor  # noqa: E402
from areal.monarch_plugin.trainer_service import (  # noqa: E402
    FSDPTrainerService,
    TrainerService,
)


# ===========================================================================
# TrainerService Protocol conformance
# ===========================================================================


class TestTrainerServiceProtocol(unittest.TestCase):
    """Verify TrainerService Protocol and FSDPTrainerService conformance."""

    def test_fsdp_service_isinstance_protocol(self):
        """FSDPTrainerService satisfies the TrainerService Protocol."""
        service = FSDPTrainerService.__new__(FSDPTrainerService)
        self.assertIsInstance(service, TrainerService)

    def test_protocol_requires_initialize(self):
        """Protocol defines initialize()."""
        self.assertIn("initialize", dir(TrainerService))

    def test_protocol_requires_run_training_step(self):
        """Protocol defines run_training_step()."""
        self.assertIn("run_training_step", dir(TrainerService))

    def test_protocol_requires_run_training_step_with_batch(self):
        """Protocol defines run_training_step_with_batch()."""
        self.assertIn("run_training_step_with_batch", dir(TrainerService))

    def test_protocol_requires_shutdown(self):
        """Protocol defines shutdown()."""
        self.assertIn("shutdown", dir(TrainerService))

    def test_custom_impl_conforms(self):
        """A custom class implementing the Protocol passes isinstance check."""

        class DummyService:
            def initialize(self) -> dict:
                return {"status": "ready"}

            def run_training_step(self, global_step: int) -> dict:
                return {"global_step": global_step}

            def run_training_step_with_batch(
                self, batch_data: dict, global_step: int
            ) -> dict:
                return {"global_step": global_step}

            def shutdown(self) -> None:
                pass

        self.assertIsInstance(DummyService(), TrainerService)


# ===========================================================================
# FSDPTrainerService delegation tests (mocked PPOTrainer)
# ===========================================================================


def _make_service(**overrides):
    """Create an FSDPTrainerService with sensible defaults for testing."""
    defaults = dict(
        cli_args=["dummy_script.py"],
        env_vars={},
        rank=0,
        world_size=1,
        master_addr="127.0.0.1",
        master_port=29500,
        generator_actor=MagicMock(),
        reward_actor=None,
        agent_actor=None,
        xccl_weight_update_alloc_mode=None,
        train_dp_size=1,
    )
    defaults.update(overrides)
    return FSDPTrainerService(**defaults)


class TestFSDPTrainerServiceDelegation(unittest.TestCase):
    """Test that FSDPTrainerService correctly delegates to PPOTrainer."""

    def test_run_training_step_delegates(self):
        """run_training_step calls trainer.run_single_step."""
        service = _make_service()
        mock_trainer = MagicMock()
        mock_trainer.run_single_step.return_value = {
            "global_step": 5,
            "epoch": 0,
            "epoch_step": 5,
        }
        service._trainer = mock_trainer
        service._train_kwargs = {"workflow": None}

        result = service.run_training_step(5)
        mock_trainer.run_single_step.assert_called_once_with(5, workflow=None)
        self.assertEqual(result["global_step"], 5)

    def test_run_training_step_with_batch_delegates(self):
        """run_training_step_with_batch passes rollout_batch kwarg."""
        service = _make_service()
        mock_trainer = MagicMock()
        mock_trainer.run_single_step.return_value = {
            "global_step": 3,
            "epoch": 0,
            "epoch_step": 3,
        }
        service._trainer = mock_trainer
        service._train_kwargs = {}

        batch = {"prompts": ["hello"], "rewards": [1.0]}
        result = service.run_training_step_with_batch(batch, 3)
        mock_trainer.run_single_step.assert_called_once_with(
            3, rollout_batch=batch
        )
        self.assertEqual(result["global_step"], 3)

    def test_shutdown_calls_trainer_close(self):
        """shutdown() calls trainer.close()."""
        service = _make_service()
        mock_trainer = MagicMock()
        service._trainer = mock_trainer

        service.shutdown()
        mock_trainer.close.assert_called_once()
        self.assertIsNone(service._trainer)

    def test_shutdown_handles_exception(self):
        """shutdown() catches exceptions from trainer.close()."""
        service = _make_service()
        mock_trainer = MagicMock()
        mock_trainer.close.side_effect = RuntimeError("boom")
        service._trainer = mock_trainer

        # Should not raise
        service.shutdown()
        self.assertIsNone(service._trainer)

    def test_shutdown_idempotent(self):
        """shutdown() is safe when trainer is already None."""
        service = _make_service()
        service._trainer = None
        service.shutdown()  # Should not raise


# ===========================================================================
# TrainerActor delegation tests
# ===========================================================================


class TestTrainerActorDelegation(unittest.TestCase):
    """Test that TrainerActor delegates to its service."""

    def _make_actor(self):
        """Create a TrainerActor with mocked service."""
        actor = TrainerActor(
            cli_args=["dummy.py"],
            env_vars={},
            rank=0,
            world_size=1,
            master_addr="127.0.0.1",
            master_port=29500,
            generator_actor=MagicMock(),
        )
        return actor

    def test_actor_stores_service(self):
        """TrainerActor.__init__ creates an FSDPTrainerService."""
        actor = self._make_actor()
        self.assertIsInstance(actor._service, FSDPTrainerService)

    def test_train_step_delegates(self):
        """train_step delegates to service.run_training_step."""
        actor = self._make_actor()
        actor._service = MagicMock()
        actor._service.run_training_step.return_value = {
            "global_step": 1,
            "epoch": 0,
            "epoch_step": 1,
        }

        result = actor.train_step(1)
        actor._service.run_training_step.assert_called_once_with(1)
        self.assertEqual(result["global_step"], 1)

    def test_do_rollout_delegates(self):
        """do_rollout delegates to service.run_training_step."""
        actor = self._make_actor()
        actor._service = MagicMock()
        actor._service.run_training_step.return_value = {
            "global_step": 2,
            "epoch": 0,
            "epoch_step": 2,
        }

        result = actor.do_rollout(2)
        actor._service.run_training_step.assert_called_once_with(2)

    def test_train_on_batch_delegates(self):
        """train_on_batch delegates to service.run_training_step_with_batch."""
        actor = self._make_actor()
        actor._service = MagicMock()
        actor._service.run_training_step_with_batch.return_value = {
            "global_step": 3,
            "epoch": 0,
            "epoch_step": 3,
        }

        batch = {"data": [1, 2, 3]}
        result = actor.train_on_batch(batch, 3)
        actor._service.run_training_step_with_batch.assert_called_once_with(batch, 3)

    def test_shutdown_delegates(self):
        """shutdown delegates to service.shutdown."""
        actor = self._make_actor()
        actor._service = MagicMock()

        actor.shutdown()
        actor._service.shutdown.assert_called_once()

    def test_initialize_delegates_to_service(self):
        """initialize delegates to service.initialize after rank detection."""
        actor = self._make_actor()
        actor._service = MagicMock()
        actor._service.initialize.return_value = {
            "status": "ready",
            "rank": 0,
            "max_steps": 100,
            "start_step": 0,
            "steps_per_epoch": 29,
        }

        result = actor.initialize()
        actor._service.initialize.assert_called_once()
        self.assertEqual(result["max_steps"], 100)


# ===========================================================================
# TrainerActor with custom backend
# ===========================================================================


class TestTrainerActorCustomBackend(unittest.TestCase):
    """Verify that TrainerActor can use a custom TrainerService backend."""

    def test_custom_service_class(self):
        """TrainerActor can use a custom _service_class."""

        class CustomService:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self._initialized = False

            def initialize(self) -> dict:
                self._initialized = True
                return {
                    "status": "ready",
                    "rank": 0,
                    "max_steps": 10,
                    "start_step": 0,
                    "steps_per_epoch": 5,
                }

            def run_training_step(self, global_step: int) -> dict:
                return {"global_step": global_step}

            def run_training_step_with_batch(
                self, batch_data: dict, global_step: int
            ) -> dict:
                return {"global_step": global_step}

            def shutdown(self) -> None:
                pass

        class CustomTrainerActor(TrainerActor):
            _service_class = CustomService

        actor = CustomTrainerActor(
            cli_args=["dummy.py"],
            env_vars={},
            rank=0,
            world_size=1,
            master_addr="127.0.0.1",
            master_port=29500,
            generator_actor=MagicMock(),
        )

        self.assertIsInstance(actor._service, CustomService)
        result = actor.initialize()
        self.assertTrue(actor._service._initialized)
        self.assertEqual(result["max_steps"], 10)


if __name__ == "__main__":
    unittest.main()
