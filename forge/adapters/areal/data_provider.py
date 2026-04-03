"""AReaL DataProvider -- serves training data independently of PPOTrainer.

Creates a standalone dataloader from the same experiment config used by
PPOTrainer, allowing the RolloutProducer to iterate over training data
without blocking the TrainerActor.

Runs as a CPU-only ForgeActor so it can serve data to multiple consumers.
"""

from __future__ import annotations

import logging
from typing import Any

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

logger = logging.getLogger(__name__)


class AReaLDataProvider(ForgeActor):
    """Monarch actor that serves data batches from AReaL's dataset.

    Args:
        cli_args: AReaL CLI argument list (same as training script).
        env_vars: Environment variables for the data loading process.

    Usage::

        provider = await AReaLDataProvider.options(procs=1).as_actor(
            cli_args=forge_cfg.training_args,
            env_vars=forge_cfg.trainer_env,
        )
        batch = await provider.get_batch.call_one()
    """

    procs = 1
    with_gpus = False

    def __init__(
        self,
        cli_args: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        self._cli_args = cli_args or []
        self._env_vars = env_vars or {}
        self._dataloader = None
        self._iterator = None
        self._config = None
        self._batch_size = 1
        self._epoch = 0
        self._total_served = 0

    @endpoint
    def setup(self) -> dict:
        """Initialize the dataloader from AReaL config."""
        import os
        import sys

        os.environ.update(self._env_vars)

        from areal.api.cli_args import parse_cli_args

        argv = self._cli_args[1:] if self._cli_args else sys.argv[1:]
        config, _ = parse_cli_args(argv)
        self._config = config

        self._dataloader = self._create_dataloader(config)
        self._iterator = iter(self._dataloader)
        self._batch_size = config.train_dataset.batch_size

        total_batches = len(self._dataloader)
        logger.info(
            f"AReaLDataProvider ready: {total_batches} batches/epoch, "
            f"batch_size={self._batch_size}"
        )
        return {
            "batches_per_epoch": total_batches,
            "batch_size": self._batch_size,
        }

    @endpoint
    def get_batch(self) -> list[dict]:
        """Return the next batch of raw data items.

        Returns:
            List of data dicts, one per sample.
            Returns empty list if epoch is exhausted (caller should call reset).
        """
        if self._iterator is None:
            raise RuntimeError("DataProvider not initialized. Call setup() first.")

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._epoch += 1
            self._iterator = iter(self._dataloader)
            try:
                batch = next(self._iterator)
            except StopIteration:
                return []

        items = self._unpack_batch(batch)
        self._total_served += len(items)
        return items

    @endpoint
    def reset(self) -> dict:
        """Reset the iterator to the beginning."""
        if self._dataloader is not None:
            self._iterator = iter(self._dataloader)
            self._epoch += 1
        return {"epoch": self._epoch}

    @endpoint
    def get_stats(self) -> dict:
        return {
            "epoch": self._epoch,
            "total_served": self._total_served,
            "batches_per_epoch": len(self._dataloader) if self._dataloader else 0,
        }

    @staticmethod
    def _create_dataloader(config):
        """Create a standalone dataloader from AReaL config."""
        from areal.dataset import build_dataset
        from areal.utils.dataloader import create_dataloader

        dataset_config = config.train_dataset
        train_dataset = build_dataset(dataset_config)

        if train_dataset is None:
            from areal.trainer.rl_trainer import _EmptyDataLoader

            steps_per_epoch = 1
            if config.total_train_steps is not None and config.total_train_epochs > 0:
                steps_per_epoch = max(
                    1, config.total_train_steps // config.total_train_epochs
                )
            return _EmptyDataLoader(
                batch_size=dataset_config.batch_size,
                steps_per_epoch=steps_per_epoch,
            )

        return create_dataloader(
            train_dataset,
            rank=0,
            world_size=1,
            dataset_config=dataset_config,
        )

    @staticmethod
    def _unpack_batch(batch: Any) -> list[dict]:
        """Convert a dataloader batch into a list of per-sample dicts."""
        if isinstance(batch, dict):
            keys = list(batch.keys())
            if not keys:
                return []
            first = batch[keys[0]]
            if isinstance(first, (list, tuple)):
                n = len(first)
                return [{k: batch[k][i] for k in keys} for i in range(n)]
            return [batch]

        if isinstance(batch, (list, tuple)):
            if batch and isinstance(batch[0], dict):
                return list(batch)
            return [{"data": item} for item in batch]

        return [{"data": batch}]
