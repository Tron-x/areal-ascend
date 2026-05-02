"""Recursive process-tree termination helper.

Used by long-running actors that own a tree of child processes (vLLM
rollout servers, dataloader workers, etc.) and need a deterministic
"clean up everything I spawned, and everything they spawned, before I
return" hook in their ``teardown`` endpoint.

Why this lives in ``forge/utils/`` rather than inside one actor
-------------------------------------------------------------
Two consumers and counting:

* :class:`forge.actors.msswift_rollout.MsSwiftRolloutActor` -- kills
  vLLM ``EngineCore`` + ``Worker_TP*`` grandchildren that hold HCCL
  ports between runs.
* :class:`forge.actors.msswift_trainer.MsSwiftTrainerActor` -- kills
  pytorch ``DataLoader`` worker children (and any process the inner
  framework spawned mid-training) so a crashed step doesn't leak
  procs into the next ``run()`` call on the same actor.

Per the framework first principles (`.cursor/rules/framework-first-principles.mdc`
"common functionality goes down to ``areal/<component>/`` or
``forge/utils/`` once ≥2 backends need it") we keep this here instead
of duplicating in each actor.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def kill_descendants(
    root_pid: int,
    *,
    term_grace_s: float = 8.0,
    kill_grace_s: float = 5.0,
) -> tuple[int, int, int]:
    """Recursively SIGTERM-then-SIGKILL every descendant of ``root_pid``.

    Walk strategy and rationale
    ---------------------------
    * Snapshot every descendant via ``psutil.Process.children(recursive=True)``
      before sending any signal: psutil enumerates ``/proc`` which is the
      only way to capture grandchildren that have ``setpgid``'d
      themselves out of the actor's process group (vLLM ``EngineCore``
      does this).  ``os.killpg`` would miss those.
    * SIGTERM first: vLLM and HuggingFace dataloader workers have
      atexit hooks that release shared memory + dump trace files on
      graceful shutdown.  Worth the ``term_grace_s`` budget.
    * SIGKILL the survivors: a process running an in-flight HCCL
      collective ignores SIGTERM until the kernel call returns,
      potentially forever.  Always escalate after the grace window.
    * Reap direct-child zombies via ``os.waitpid(-1, WNOHANG)``: the
      caller (this actor) stays running after teardown, so without an
      explicit waitpid we'd leave zombies in the proctable.  Indirect
      descendants are reparented to ``init`` after their parent dies
      and are reaped automatically.

    Zombies vs survivors
    --------------------
    ``psutil.wait_procs`` classifies any ``/proc`` entry as "alive",
    including unreaped zombies whose parent (the caller, by
    construction) hasn't called ``waitpid``.  A zombie holds no ports,
    no memory, no FDs -- just a proctable slot -- so we filter
    ``STATUS_ZOMBIE`` out of the survivor count.  Only true running
    descendants count as survivors, because the only realistic cause of
    surviving a SIGKILL is a stuck NPU/GPU driver call (kernel-mode
    uninterruptible), which is the actionable failure the caller
    cares about.

    Args:
        root_pid: PID whose descendants should be terminated.  In
            practice always ``os.getpid()`` -- the actor calling itself.
        term_grace_s: Seconds to wait after SIGTERM before escalating.
        kill_grace_s: Seconds to wait after SIGKILL before classifying
            survivors.

    Returns:
        ``(terminated_count, killed_count, survivor_count)`` where:

        * ``terminated_count`` = descendants that received SIGTERM
          (= total descendants minus those that vanished between
          snapshot and signal),
        * ``killed_count`` = descendants that needed SIGKILL escalation,
        * ``survivor_count`` = descendants still running after both
          phases (zombies excluded).  Non-zero almost always means a
          stuck kernel driver call and is logged at ERROR level.

    Returns ``(0, 0, 0)`` and logs a warning if psutil is unavailable
    or ``root_pid`` no longer exists.  Never raises.
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil unavailable; skipping descendant kill")
        return (0, 0, 0)

    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return (0, 0, 0)

    try:
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return (0, 0, 0)

    if not descendants:
        return (0, 0, 0)

    logger.info(
        "kill_descendants: root=%d descendants=%d pids=%s",
        root_pid,
        len(descendants),
        [p.pid for p in descendants],
    )

    terminated = 0
    for p in descendants:
        try:
            p.terminate()
            terminated += 1
        except psutil.NoSuchProcess:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("SIGTERM pid=%d failed: %s", p.pid, e)

    _, alive = psutil.wait_procs(descendants, timeout=term_grace_s)

    killed = 0
    for p in alive:
        try:
            p.kill()
            killed += 1
        except psutil.NoSuchProcess:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("SIGKILL pid=%d failed: %s", p.pid, e)

    if alive:
        _, still_alive = psutil.wait_procs(alive, timeout=kill_grace_s)
    else:
        still_alive = []

    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break

    real_survivors: list[int] = []
    for p in still_alive:
        try:
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                real_survivors.append(p.pid)
        except psutil.NoSuchProcess:
            pass

    if real_survivors:
        logger.error(
            "kill_descendants: %d descendants survived SIGKILL (likely "
            "stuck in kernel NPU driver call): %s",
            len(real_survivors),
            real_survivors,
        )

    return (terminated, killed, len(real_survivors))
