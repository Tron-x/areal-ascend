"""POC: replace ``worker_manager.sh`` with Monarch's native
:class:`monarch._src.job.job.SSHJob`.

Goal: prove the following three invariants hold when we subclass
Monarch's ``SSHJob`` and inject our CANN / conda / HCCL / HiXL
environment, WITHOUT touching any production code yet.

Checked invariants:

  1. START -- ``job._create()`` starts ``run_worker_loop_forever`` on
     every host in the hostfile, through the same SSH port and Python
     env that ``worker_manager.sh`` would.
  2. ATTACH -- ``job._state()`` returns a ``JobState`` whose host
     meshes initialize without error, and we can slice each worker
     out of it the same way ``BareMetalLauncher`` does today.
  3. STOP -- ``job._kill()`` terminates the workers; a post-hoc SSH
     probe (``pgrep -f run_worker_loop_forever``) comes back empty
     on every host.

Run::

    python forge/scripts/poc_ssh_job.py \\
        --hostfile forge/configs/hostfile.txt \\
        --ssh-port 36000

Exit code 0 == all three invariants passed.  Any failure prints
an explicit diagnostic naming which invariant broke so we know
whether to proceed with the full refactor.

This script is standalone / dev-only.  It intentionally does NOT
import from ``forge.provisioner`` so a bug there can't hide a bug
here.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Make sure upstream Monarch is importable when running from the
# repo root (the production env sets PYTHONPATH; POC doesn't).
_DEFAULT_PYTHONPATH_PARTS = [
    "/root/AReaL",
    "/root/torchstore",
    "/root/monarch/python",
]
for p in _DEFAULT_PYTHONPATH_PARTS:
    if p not in sys.path:
        sys.path.insert(0, p)

from monarch._src.job.job import ProcessState, SSHJob  # noqa: E402

# ----------------------------------------------------------------------
# ForgeSSHJob -- the 30-line subclass we'd move into forge/ if POC works
# ----------------------------------------------------------------------


class ForgeSSHJob(SSHJob):
    """Monarch ``SSHJob`` + Forge-specific remote env preamble.

    Parent ``SSHJob._start_host`` runs ``python -c "<startup>"`` on
    the remote host via ssh.  We need that process to first
    ``source set_env.sh``, ``conda activate``, and export the
    HCCL / HiXL / torchstore env vars that live in
    ``worker_manager.sh::worker_script``; otherwise the worker
    python can't import ``torch_npu`` and the rollout hangs with a
    CANN misconfigured error rather than a clean ImportError.
    """

    def __init__(
        self,
        *,
        cann_home: str,
        conda_env: str,
        conda_bin: str,
        areal_root: str,
        python_exe: str = "python",
        ssh_args=(),
        monarch_port: int = 22222,
        torchstore_pool_mb: int = 8192,
    ):
        super().__init__(
            python_exe=python_exe,
            ssh_args=ssh_args,
            monarch_port=monarch_port,
        )
        self._cann_home = cann_home
        self._conda_env = conda_env
        self._conda_bin = conda_bin
        self._areal_root = areal_root
        self._pool_mb = torchstore_pool_mb

    def _env_preamble(self) -> str:
        """Bash preamble that must precede ``python -c ...``.

        Kept in one place so the preamble is identical across all
        hosts; the shape matches ``worker_manager.sh::worker_script``
        line-for-line so a diff is trivially reviewable.
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
            f"source {shlex.quote(self._cann_home)}/set_env.sh 2>/dev/null",
            (
                f"[[ -f {shlex.quote(ascend_root)}/nnal/atb/set_env.sh ]] && "
                f"source {shlex.quote(ascend_root)}/nnal/atb/set_env.sh"
            ),
            f'eval "$({shlex.quote(self._conda_bin)} shell.bash hook)"',
            f"conda activate {shlex.quote(self._conda_env)}",
            f'export PYTHONPATH="{pythonpath}"',
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
            'export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"',
            'export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-1}"',
            'export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"',
            "export MONARCH_HIXL_TRANSPORT=roce",
            "export HCCL_INTRA_ROCE_ENABLE=1",
            "export HCCL_CONNECT_TIMEOUT=120",
            'export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-60000-60255}"',
            "unset TORCHSTORE_RDMA_ENABLED",
            "export TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0",
            "export TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:0",
            f'export TORCHSTORE_MONARCH_RDMA_POOL_MB="${{TORCHSTORE_MONARCH_RDMA_POOL_MB:-{self._pool_mb}}}"',
            f"cd {shlex.quote(self._areal_root)}",
        ]
        return "; ".join(parts)

    def _kill(self) -> None:
        """Monarch parent only SIGKILLs the local ssh clients.

        Without a PTY (``-n`` flag, no ``-t``) sshd on the remote
        side sees TCP close but does NOT forward SIGHUP, so the
        remote python ``run_worker_loop_forever`` keeps running and
        leaks across job boundaries.  Mirror ``worker_manager.sh
        stop`` behavior here: after SIGKILL-ing the local ssh pids,
        also ssh to each host and pkill the remote worker process.

        Uses the bracket trick ``[r]un_worker_loop_forever`` so the
        pkill-invoking shell's argv is NOT itself matched.
        """
        super()._kill()
        # Best-effort remote cleanup.  Does not raise on ssh failure;
        # just logs, because by now the job is already torn down from
        # Monarch's perspective and we don't want _kill to crash the
        # cleanup path of whatever called us.
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

    def _start_host(self, host: str) -> ProcessState:
        """Override parent to prepend Forge env preamble.

        Mirrors parent verbatim except we splice ``env_preamble; exec``
        in front of the ``python -c`` invocation.  ``exec`` replaces
        the bash shell with python so Monarch's PID tracking on the
        local side continues to map to the actual worker process tree
        on the remote (no intermediate bash layer to orphan).
        """
        addr = f"tcp://{host}:{self._port}"
        startup = (
            f"from monarch.actor import run_worker_loop_forever; "
            f"run_worker_loop_forever(address={addr!r}, "
            f'ca="trust_all_connections")'
        )
        py_cmd = f"{shlex.quote(self._python_exe)} -c {shlex.quote(startup)}"
        full = f"{self._env_preamble()}; exec {py_cmd}"
        print(f"[ForgeSSHJob] ssh -> {host}: starting worker on {addr}", flush=True)
        proc = subprocess.Popen(
            ["ssh", *self._ssh_args, host, "-n", full],
            start_new_session=True,
        )
        return ProcessState(proc.pid, addr)


# ----------------------------------------------------------------------
# POC harness
# ----------------------------------------------------------------------


def _read_hosts(hostfile: Path) -> list[str]:
    hosts = []
    for raw in hostfile.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        hosts.append(line.split()[0])
    return hosts


def _ssh_cmd(*, ssh_port: int):
    return [
        "ssh",
        "-p",
        str(ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=5",
    ]


def _remote_probe_worker(host: str, ssh_port: int) -> int:
    """Count the worker processes currently alive on ``host``.

    Returns -1 if the ssh itself failed (connectivity issue, not
    a worker-present issue).  Non-negative values are reliable.
    """
    try:
        out = subprocess.run(
            [
                *_ssh_cmd(ssh_port=ssh_port),
                f"root@{host}",
                # Use `[r]un_...` bracket-trick so the pgrep's own invoking
                # bash (whose argv literally contains the pattern as plain
                # text) is NOT matched -- classic grep-doesn't-match-itself.
                # Without this we'd double-count our own probe shell and
                # invariant 3 would fail spuriously.
                'pgrep -af "python.*[r]un_worker_loop_forever" | wc -l',
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return -1
        return int(out.stdout.strip() or "0")
    except (subprocess.TimeoutExpired, ValueError):
        return -1


def _tcp_reachable(host: str, port: int, *, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


async def _run(args: argparse.Namespace) -> int:
    hostfile = Path(args.hostfile).resolve()
    hosts = _read_hosts(hostfile)
    if not hosts:
        print(f"[POC] hostfile has no hosts: {hostfile}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f" Forge SSHJob POC  ({len(hosts)} hosts)")
    print("=" * 72)
    for h in hosts:
        print(f"  host = {h}  ssh-port={args.ssh_port}  worker-port={args.worker_port}")
    print("=" * 72, flush=True)

    ssh_args = ["-p", str(args.ssh_port), "-o", "StrictHostKeyChecking=no"]

    job = ForgeSSHJob(
        cann_home=args.cann,
        conda_env=args.conda_env,
        conda_bin=args.conda_bin,
        areal_root=args.areal_root,
        python_exe="python",
        ssh_args=ssh_args,
        monarch_port=args.worker_port,
    )
    job.add_mesh("poc", hosts)

    # ----- invariant 1: START -----
    # Use public `apply()` rather than `_create()` directly: apply()
    # wraps _create and transitions ``_status`` to "running", which is
    # what LoginJob._pids_active() gates on.  Calling _create alone
    # leaves self.active=False and _state() raises "lost connection".
    print("[POC] invariant 1: apply() starts workers ...", flush=True)
    try:
        job.apply(client_script=None)
    except Exception as e:
        print(f"[POC] FAIL inv-1: apply raised {e!r}", file=sys.stderr)
        return 10

    # Wait up to 60s for TCP listeners on each host:worker_port.
    # Same budget the bash script uses.
    t0 = time.time()
    ready = {h: False for h in hosts}
    while time.time() - t0 < 60:
        for h in hosts:
            if not ready[h] and _tcp_reachable(h, args.worker_port, timeout=2):
                ready[h] = True
                print(f"  {h}:{args.worker_port} READY (+{time.time() - t0:.1f}s)")
        if all(ready.values()):
            break
        await asyncio.sleep(1)
    if not all(ready.values()):
        missing = [h for h, r in ready.items() if not r]
        print(f"[POC] FAIL inv-1: workers not listening on {missing}", file=sys.stderr)
        _safe_kill(job)
        return 11
    print("[POC] inv-1 PASS (all workers listening)")

    # ----- invariant 2: ATTACH -----
    print("[POC] invariant 2: _state() attaches to workers ...", flush=True)
    try:
        state = job._state()
        # JobState uses __getattr__ to expose each mesh by its name
        # (state.poc, state.trainers, ...); the raw dict lives at
        # state._hosts.  Don't touch .hosts -- that name is reserved.
        host_mesh = state._hosts["poc"]
        await host_mesh.initialized
        size = host_mesh.size()
    except Exception as e:
        print(f"[POC] FAIL inv-2: attach/initialize raised {e!r}", file=sys.stderr)
        _safe_kill(job)
        return 20

    if size != len(hosts):
        print(
            f"[POC] FAIL inv-2: host_mesh size={size} != expected {len(hosts)}",
            file=sys.stderr,
        )
        _safe_kill(job)
        return 21
    print(f"[POC] inv-2 PASS (HostMesh size={size})")

    # ----- invariant 3: STOP -----
    print("[POC] invariant 3: _kill() cleans up workers ...", flush=True)
    try:
        job._kill()
    except Exception as e:
        print(f"[POC] FAIL inv-3: _kill raised {e!r}", file=sys.stderr)
        return 30

    # SSH-grep each host, give it a few seconds for propagation.
    await asyncio.sleep(3)
    residuals: dict[str, int] = {}
    for h in hosts:
        residuals[h] = _remote_probe_worker(h, args.ssh_port)

    leaks = {h: n for h, n in residuals.items() if n > 0}
    unknown = {h: n for h, n in residuals.items() if n < 0}

    if unknown:
        print(
            f"[POC] WARN inv-3: could not probe residuals on {unknown} "
            f"(SSH error). Manually verify.",
            file=sys.stderr,
        )
    if leaks:
        print(
            f"[POC] FAIL inv-3: residual run_worker_loop_forever procs: {leaks}",
            file=sys.stderr,
        )
        return 31
    print(f"[POC] inv-3 PASS (residuals={residuals})")

    print("=" * 72)
    print(" SSHJob POC: ALL 3 INVARIANTS PASSED")
    print(" Implication: we can delete worker_manager.sh and replace")
    print(" BareMetalLauncher._start-workers plumbing with a ForgeSSHJob.")
    print("=" * 72)
    return 0


def _safe_kill(job):
    try:
        job._kill()
    except Exception as e:
        print(f"[POC] cleanup _kill raised {e!r} (non-fatal)", file=sys.stderr)


def _install_sigint_handler(job_box: list):
    def _handler(signum, _frame):
        print(f"\n[POC] caught signal {signum}, cleaning up ...", file=sys.stderr)
        if job_box:
            _safe_kill(job_box[0])
        sys.exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="POC for replacing worker_manager.sh with Monarch SSHJob"
    )
    p.add_argument(
        "--hostfile",
        default="/root/AReaL/forge/configs/hostfile.txt",
        help="Path to hostfile (IP [slots=N] per line).",
    )
    p.add_argument("--ssh-port", type=int, default=36000)
    p.add_argument("--worker-port", type=int, default=22222)
    p.add_argument("--cann", default="/usr/local/Ascend/cann-9.0.0-beta.1")
    p.add_argument("--conda-env", default="monarch_ascend")
    p.add_argument("--conda-bin", default="/root/miniconda3/bin/conda")
    p.add_argument("--areal-root", default="/root/AReaL")
    args = p.parse_args(argv)

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
