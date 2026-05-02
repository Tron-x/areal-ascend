"""TitanTrainerActor -- run TorchTitan's native training inside a Monarch actor.

**Conforms to:** :class:`forge.core.protocols.SPMDTrainerProtocol`
(Adapter Path; partial -- ``setup_env`` + ``run`` are present, ``teardown``
is **TODO**).  See :class:`forge.actors.msswift_trainer.MsSwiftTrainerActor`
for the canonical full conformance.

**Pure Path note:** TorchTitan is a *pure* training backend per
``.cursor/rules/framework-first-principles.mdc``, but the *current*
actor here treats it as a black-box (``run()`` calls
``torchtitan.train.Trainer.train()`` and blocks).  When the Pure Path
MVP lands, a sibling actor implementing per-step
:class:`forge.core.protocols.TrainEngine` will be added so Forge
algorithms can drive Titan step-by-step.  Both actor shapes will coexist
because the SPMD shape stays useful for pure pretrain.

This is the **B-mini** integration path for TorchTitan: a single Monarch
actor mesh whose only job is to call ``torchtitan.train.Trainer.train()``
on every rank.  No Forge GRPO orchestration, no vLLM, no weight sync --
it is the same code path you'd get from running TorchTitan's
``train.py`` under ``torchrun``, but with the rank wiring done by Monarch
instead of torchelastic so we can deploy to two bare-metal NPU hosts via
the existing ``_SSHJobFleet`` worker lifecycle.

Why this exists
---------------
TorchTitan is currently wired into ``forge launch --backend titan`` only
for use *inside* the GRPO loop.  Validating TorchTitan itself on NPU
across two hosts shouldn't require dragging in the entire RL stack;
this actor is the smallest possible bridge between Monarch's process
mesh and TorchTitan's ``Trainer``.

Once this is green, the same actor can be lifted into ``forge launch``
as a pure-pretrain ``--backend titan-pretrain`` mode (a.k.a. B-full).
The endpoint surface is intentionally small so the upgrade path is just
"call ``run()`` from a different driver".

Rank / device plumbing
----------------------
The actor inherits from Monarch's :class:`SPMDActor`, which already
populates ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` /
``MASTER_PORT`` from ``current_rank()`` on every proc.  TorchTitan's
``Trainer.__init__`` then reads ``LOCAL_RANK`` to pick the device
(``torch.device(f"{device_type}:{LOCAL_RANK}")``) and calls
``init_process_group(backend, timeout)`` with no explicit ``rank`` /
``world_size``, which is the standard ``env://`` torchelastic contract.

We do **not** manually set ``ASCEND_RT_VISIBLE_DEVICES`` per proc: the
worker procs inherit all 8 NPUs from ``worker_manager.sh``'s daemon and
TorchTitan's ``device_module.set_device(LOCAL_RANK)`` does the right
thing.  This matches how TorchTitan runs under ``torchrun`` on GPU.

What ``run()`` does
-------------------
1. ``os.chdir(cwd)`` -- TorchTitan's debug toml uses relative paths
   (``hf_assets_path = "./tests/assets/tokenizer"``,
   ``dataset = "c4_test"`` -> ``tests/assets/c4_test``), so the CWD
   has to be the TorchTitan repo root.
2. Build a ``ConfigManager`` argv from ``--job.config-file`` + extra
   overrides.  We call ``ConfigManager.parse_args(args)`` directly
   instead of monkey-patching ``sys.argv`` -- ``parse_args`` accepts
   the list explicitly.
3. ``Trainer(config).train()`` -- this is byte-for-byte the same call
   ``train.py`` makes under ``__main__``.
4. Always destroy the process group on exit so the proc can be reused
   (Monarch keeps procs alive between endpoint calls).
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time
from typing import Any

from monarch._src.spmd.actor import SPMDActor
from monarch.actor import endpoint

logger = logging.getLogger(__name__)


class TitanTrainerActor(SPMDActor):
    """Run TorchTitan ``Trainer.train()`` once per actor instance.

    Subclasses :class:`monarch._src.spmd.actor.SPMDActor` to inherit
    the ``RANK`` / ``LOCAL_RANK`` / ... env wiring + the
    ``setup_env(master_addr, master_port)`` endpoint already vetted by
    Monarch's SPMD path.  We only add a ``run()`` endpoint that does
    the TorchTitan-specific work.

    Why subclass SPMDActor (rather than re-implement)?
        SPMDActor's ``__init__`` already pulls ``current_rank()`` /
        ``current_size()`` and stashes ``rank`` / ``local_rank`` /
        ``world_size`` etc. on ``self``.  Replicating that here would
        be cargo-culted code that drifts whenever Monarch tweaks the
        contract (e.g., when the ``gpus`` dim is renamed to
        ``procs`` for non-GPU layouts).  Subclassing also means the
        existing ``setup_torch_elastic_env_async`` helper from
        ``monarch._src.spmd`` works on us if a caller ever wants the
        higher-level wiring.
    """

    @endpoint
    def run(
        self,
        toml_path: str,
        cwd: str,
        overrides: list[str] | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
    ) -> dict[str, Any]:
        """Drive one full TorchTitan training run on this rank.

        Args:
            toml_path: Absolute path to a TorchTitan ``.toml`` config
                (e.g. ``debug_model.toml``).  Passed verbatim to
                ``--job.config-file`` so any TT-supported config
                works.
            cwd: Working directory before the ``Trainer`` is built.
                Must be the TorchTitan repo root for the bundled
                debug datasets / tokenizer relative paths to resolve.
            overrides: Extra ``--<section>.<key> <value>`` pairs
                appended after ``--job.config-file``.  Same syntax
                TorchTitan's CLI accepts directly.  Useful for
                lowering ``--training.steps`` to 3 for smoke runs
                without editing the toml.
            master_addr / master_port: If provided (driver-supplied)
                set the rendezvous address explicitly via
                ``self._setup_env``.  When ``None`` the caller is
                expected to have invoked
                ``setup_torch_elastic_env_async`` (or the
                ``setup_env`` endpoint) beforehand -- in which case
                the env vars are already populated on this proc.

        Returns:
            ``{"rank": int, "host": str, "elapsed_s": float, "ok": True}``
            so the driver can sanity-check that every rank actually
            reached ``Trainer.train()`` completion.

        Notes:
            * We import ``torch`` / ``torch_npu`` / TT modules
              **inside** ``run()`` rather than at module top-level so
              the actor file stays importable on a CPU-only driver
              (the smoke driver itself can run on a non-NPU box that
              just orchestrates the remote workers).
            * ``trainer.close()`` and ``destroy_process_group()`` are
              wrapped in best-effort try/except: if an OOM in step
              0 takes the rank down, we still want to emit a
              structured error rather than mask it with a cleanup
              traceback.
        """

        if master_addr is not None and master_port is not None:
            self._setup_env(master_addr, master_port)

        os.chdir(cwd)

        import torch
        import torch_npu  # noqa: F401  -- registers the npu backend
        from torchtitan.config import ConfigManager
        from torchtitan.tools.logging import init_logger
        from torchtitan.train import Trainer

        init_logger()

        # Build TT argv exactly the same way ``train.py`` would have
        # under torchrun.  Using parse_args(args=...) keeps sys.argv
        # untouched, which matters because Monarch's bootstrap may
        # have populated sys.argv with its own internal flags.
        args: list[str] = ["--job.config-file", toml_path]
        if overrides:
            args.extend(overrides)

        config = ConfigManager().parse_args(args)

        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        world_size = int(os.environ.get("WORLD_SIZE", "-1"))
        host = socket.gethostname()
        logger.info(
            "TitanTrainerActor rank=%d local_rank=%d world_size=%d host=%s "
            "starting Trainer with config=%s",
            rank,
            local_rank,
            world_size,
            host,
            toml_path,
        )

        t0 = time.time()
        trainer: Trainer | None = None
        err: BaseException | None = None
        try:
            trainer = Trainer(config)
            trainer.train()
        except BaseException as e:  # noqa: BLE001
            err = e
            logger.exception("TitanTrainerActor rank=%d crashed", rank)
            raise
        finally:
            # Best-effort cleanup -- never let cleanup mask the real
            # error.  Trainer.close() flushes metrics + closes the
            # checkpointer; safe to skip if the trainer never finished
            # constructing.
            try:
                if trainer is not None:
                    trainer.close()
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning(
                    "TitanTrainerActor rank=%d trainer.close() raised %r",
                    rank,
                    cleanup_exc,
                )

            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning(
                    "TitanTrainerActor rank=%d destroy_process_group raised %r",
                    rank,
                    cleanup_exc,
                )

        elapsed = time.time() - t0
        # ``torch`` import above means we always have it in scope here
        # for the optional final synchronize -- keeps the device queue
        # drained before the driver moves on to teardown.
        try:
            torch.npu.synchronize()
        except Exception:  # noqa: BLE001
            pass

        # Stdout flush so the per-rank "completed" line lands in the
        # driver's interleaved log before the proc shuts down.
        sys.stdout.flush()
        sys.stderr.flush()

        return {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "host": host,
            "elapsed_s": elapsed,
            "ok": err is None,
        }
