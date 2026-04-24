"""Monarch-native SSH worker lifecycle for bare-metal clusters.

This module is the production-grade replacement for
``forge/scripts/worker_manager.sh``: instead of shelling out to a
300-line bash script to start/stop ``run_worker_loop_forever`` on
every host, we subclass Monarch's :class:`monarch._src.job.job.SSHJob`
and inject our CANN / conda / HCCL / HiXL environment preamble.

Rationale
---------

Monarch already ships ``JobTrait`` implementations for every cluster
flavor we care about (``LocalJob``, ``SSHJob``, ``SlurmJob``,
``KubernetesJob``, ...).  ``SSHJob`` in particular is purpose-built
for "I have a list of hostnames with SSH access" -- which is exactly
the bare-metal POC environment.  Using Monarch's native job trait
gives us:

1. **Unified lifecycle API** (``apply`` / ``_state`` / ``_kill``) that
   matches Slurm/K8s when we migrate, so the launcher swap becomes a
   one-line config change.
2. **Reliable PID tracking** out of the box (``_host_to_pid``); no
   need to ``pgrep`` residuals out of remote hosts for status.
3. **Sane ``can_run`` caching**: rerunning the same spec reuses the
   existing allocation instead of spawning a second worker fleet.

What the subclass adds
----------------------

Vanilla ``SSHJob._start_host`` only runs ``python -c "startup"`` on
the remote.  Our workers can't start from a bare shell: they need
CANN sourced, conda activated, and a dozen HCCL/HiXL env vars set
before ``import torch_npu`` succeeds.  :meth:`ForgeSSHJob._start_host`
prepends the :meth:`_env_preamble` string in front of the parent's
Python invocation -- cleanly keeping parent logic, just with our
shell preamble prepended.

Vanilla ``_kill`` has a second gap that hurts on bare-metal SSH: it
only SIGKILLs the *local* ssh client PIDs.  Because Monarch's
``_start_host`` uses ``-n`` (no PTY), sshd on the remote side sees
TCP close but never forwards SIGHUP, and the remote python
``run_worker_loop_forever`` keeps running as an orphan.  We override
``_kill`` to additionally issue a best-effort remote ``pgrep | kill``
(using a self-excluding regex).  This is exactly what
``worker_manager.sh stop`` did -- just as 30 lines of Python inside
the same class that started the workers, instead of a separate shell
script.

Upstream bug notes
------------------

Two concrete quirks in vanilla ``SSHJob`` discovered during the POC,
documented here for future upstream PRs:

1. ``job._create()`` does NOT transition ``_status`` to ``"running"``.
   Callers must use the public :meth:`monarch._src.job.job.JobTrait.apply`
   wrapper -- calling ``_create`` alone leaves ``self.active == False``
   and the subsequent ``_state()`` raises ``RuntimeError('lost connection')``.
2. ``_kill`` only kills local SSH processes; remote workers need
   explicit cleanup (handled by our override above).

Typical usage
-------------

::

    job = ForgeSSHJob(
        cann_home="/usr/local/Ascend/cann-9.0.0-beta.1",
        conda_env="monarch_ascend",
        conda_bin="/root/miniconda3/bin/conda",
        areal_root="/root/AReaL",
        ssh_args=["-p", "36000", "-o", "StrictHostKeyChecking=no"],
        monarch_port=22222,
    )
    job.add_mesh("bare_metal", ["192.168.0.26", "192.168.0.23"])
    job.apply()            # starts remote workers
    host_mesh = job._state()._hosts["bare_metal"]
    await host_mesh.initialized
    # ... drive training ...
    job._kill()            # stops local AND remote worker procs

``forge launch`` wires this up automatically when the launcher YAML
has ``launcher_impl: ssh_job``.  Users rarely instantiate
``ForgeSSHJob`` directly.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence

from monarch._src.job.job import ProcessState, SSHJob

logger = logging.getLogger(__name__)


class ForgeSSHJob(SSHJob):
    """Monarch :class:`SSHJob` plus Forge's CANN/HCCL/HiXL env preamble.

    Two narrow overrides versus upstream:

    - :meth:`_start_host` -- prepend Forge-specific bash environment
      setup before the parent's ``python -c "..."`` invocation, then
      ``exec`` so PID tracking on the local side continues to map to
      the actual worker process tree on the remote (no intermediate
      bash layer to orphan).
    - :meth:`_kill` -- super()._kill() + remote pgrep/kill fallback
      (see module docstring for why vanilla ``_kill`` leaks procs on
      bare-metal SSH).

    All other behavior (``apply``, ``_state``, ``can_run``, ...) is
    inherited unchanged.

    Args:
        cann_home: CANN install root; ``{cann_home}/set_env.sh`` is
            sourced on every remote host before Python starts.
        conda_env: Name of the conda env that holds the Monarch/forge
            dependencies (``monarch``, ``torch_npu``, ``torchstore``,
            ...).
        conda_bin: Full path to the ``conda`` binary used to activate
            the env.  Worth passing explicitly: ``PATH`` is not reliable
            inside a fresh non-interactive sshd shell.
        areal_root: Repo root on the remote host.  Used to set
            ``PYTHONPATH`` and ``cd`` into the repo before starting
            the worker.
        python_exe: Python executable inside the activated env.
            Defaults to ``"python"`` which resolves correctly after
            ``conda activate``.
        ssh_args: Extra ``ssh`` flags (e.g. ``-p 36000`` for a
            non-22 port, ``-o StrictHostKeyChecking=no`` for
            unattended CI).  Forwarded verbatim to the parent class.
        monarch_port: TCP port the remote worker listens on.  Must
            match whatever the driver's :class:`BareMetalLauncher`
            is configured to attach to (default 22222).
        torchstore_pool_mb: HiXL RDMA staging pool size per worker.
            8 GB is the validated default for Qwen3-0.6B; bump up
            for larger models.
    """

    def __init__(
        self,
        *,
        cann_home: str,
        conda_env: str,
        conda_bin: str,
        areal_root: str,
        python_exe: str = "python",
        ssh_args: Sequence[str] = (),
        monarch_port: int = 22222,
        torchstore_pool_mb: int = 8192,
    ):
        super().__init__(
            python_exe=python_exe,
            ssh_args=tuple(ssh_args),
            monarch_port=monarch_port,
        )
        self._cann_home = cann_home
        self._conda_env = conda_env
        self._conda_bin = conda_bin
        self._areal_root = areal_root
        self._pool_mb = torchstore_pool_mb

    # ------------------------------------------------------------------

    def _env_preamble(self) -> str:
        """Bash one-liner that must precede the remote ``python -c ...``.

        Kept deliberately parallel to the pre-refactor
        ``worker_manager.sh::worker_script`` so a reviewer can diff
        the two side-by-side.  The preamble is identical across all
        hosts in a fleet; per-host customization happens in the
        ``python -c`` address, not here.
        """
        ascend_root = os.path.dirname(self._cann_home.rstrip("/"))
        pythonpath = ":".join(
            [
                self._areal_root,
                "/root/torchstore",
                "/root/monarch/python",
                "${PYTHONPATH:-}",
            ]
        )
        parts = [
            # CANN + ATB (Ascend Transformer Boost) env.  ``2>/dev/null``
            # silences benign "file not found" noise when CANN is not at
            # a default install path -- the missing file only matters if
            # the actual NPU runtime init later fails.
            f"source {shlex.quote(self._cann_home)}/set_env.sh 2>/dev/null",
            (
                f"[[ -f {shlex.quote(ascend_root)}/nnal/atb/set_env.sh ]] && "
                f"source {shlex.quote(ascend_root)}/nnal/atb/set_env.sh"
            ),
            # Conda activation.  Inline ``eval "$(conda shell.bash hook)"``
            # rather than relying on ``~/.bashrc`` because sshd-invoked
            # shells skip .bashrc by default.
            f'eval "$({shlex.quote(self._conda_bin)} shell.bash hook)"',
            f"conda activate {shlex.quote(self._conda_env)}",
            # PYTHONPATH pins the in-repo code paths ahead of any
            # site-packages copy, so local patches to monarch / torchstore
            # / forge take effect without reinstalling.
            f'export PYTHONPATH="{pythonpath}"',
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
            # CANN log throttling (see weight_sync.md §7.1 for why 3/INFO
            # is the right operational default -- INFO-level CANN slog
            # floods the driver at ~100 MB/min and starves the rollout).
            'export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"',
            'export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-1}"',
            'export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"',
            # Cross-node transport: HiXL requires RoCE; HCCS is
            # intra-supernode only on 910B.
            "export MONARCH_HIXL_TRANSPORT=roce",
            # Must be set BEFORE python starts -- CANN reads this once
            # at library load time and ignores later Rust-side sets.
            "export HCCL_INTRA_ROCE_ENABLE=1",
            # HCCL accepts only [120, 7200]; anything smaller bounces
            # with EI0001.
            "export HCCL_CONNECT_TIMEOUT=120",
            # Wider port range so HiXL sub-comm and HCCL process-group
            # don't collide on the default 16666 slot.  256 ports is the
            # validated minimum for FSDP-4 + HiXL on one NPU.
            'export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-60000-60255}"',
            # Clear legacy torchstore RDMA toggles so the MonarchRDMA
            # backend (HiXL) stays active.
            "unset TORCHSTORE_RDMA_ENABLED",
            # torchstore staging pool: keep NPU-resident, 2 MB aligned.
            "export TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0",
            "export TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:0",
            f'export TORCHSTORE_MONARCH_RDMA_POOL_MB="${{TORCHSTORE_MONARCH_RDMA_POOL_MB:-{self._pool_mb}}}"',
            f"cd {shlex.quote(self._areal_root)}",
        ]
        return "; ".join(parts)

    # ------------------------------------------------------------------

    def _start_host(self, host: str) -> ProcessState:
        """Start ``run_worker_loop_forever`` on ``host`` via SSH.

        Mirrors parent :meth:`SSHJob._start_host` verbatim except for
        splicing :meth:`_env_preamble` + ``exec`` before the
        ``python -c`` invocation.  The ``exec`` matters: it replaces
        the bash shell with python so the process tree on the remote
        has one level, not two, and Monarch's PID tracking on our
        local side continues to correspond to a live worker.
        """
        addr = f"tcp://{host}:{self._port}"
        startup = (
            "from monarch.actor import run_worker_loop_forever; "
            f"run_worker_loop_forever(address={addr!r}, "
            'ca="trust_all_connections")'
        )
        py_cmd = f"{shlex.quote(self._python_exe)} -c {shlex.quote(startup)}"
        full_cmd = f"{self._env_preamble()}; exec {py_cmd}"
        logger.info("ForgeSSHJob: starting worker at %s", addr)
        proc = subprocess.Popen(
            ["ssh", *self._ssh_args, host, "-n", full_cmd],
            start_new_session=True,
        )
        return ProcessState(proc.pid, addr)

    # ------------------------------------------------------------------

    def _kill(self) -> None:
        """SIGKILL local ssh clients AND pgrep-kill the remote workers.

        Vanilla ``SSHJob._kill`` only kills the *local* ssh pids.  In
        the bare-metal POC we confirmed that without a PTY (``-n``
        flag, no ``-t``) sshd does not forward SIGHUP when the client
        dies, so the remote python ``run_worker_loop_forever`` keeps
        running after ``_kill`` returns and leaks across job
        boundaries.  This override mirrors ``worker_manager.sh stop``
        on top of the parent's cleanup.

        Best-effort: per-host SSH failures log to stderr but do NOT
        raise.  Callers rely on ``_kill`` succeeding for teardown
        paths (``finally`` blocks, signal handlers) and a fatal here
        would defeat the robustness that prompted the refactor.
        """
        super()._kill()
        for host in list(self._host_to_pid):
            try:
                subprocess.run(
                    [
                        "ssh",
                        *self._ssh_args,
                        "-o",
                        "StrictHostKeyChecking=no",
                        "-o",
                        "ConnectTimeout=5",
                        "-n",
                        host,
                        # Bracket-trick so pgrep's own invoking shell
                        # (whose argv literally contains the pattern
                        # as plain text in the ssh command line) is
                        # NOT matched -- classic grep-doesn't-match-
                        # itself.  Without this the kill command
                        # would race against its own cleanup shell.
                        'pgrep -af "python.*[r]un_worker_loop_forever" '
                        '| awk "{print \\$1}" | xargs -r kill -9',
                    ],
                    check=False,
                    timeout=15,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[ForgeSSHJob] remote cleanup on {host} failed: {e!r}",
                    file=sys.stderr,
                )
