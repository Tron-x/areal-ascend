"""``forge sync`` -- push local source trees to bare-metal worker hosts.

.. note::

   **Preferred path is now FUSE-backed code mount, not push-sync.**

   As of the ``remote_mount`` integration in ``_SSHJobFleet``
   (default ``mount_code=True``), workers and drivers under
   ``launcher_impl: ssh_job`` see the launcher's checkout
   transparently via FUSE -- no pre-launch ``tar | ssh`` step
   needed.  The mount also avoids drift entirely: edits to the
   launcher's source become visible on workers immediately.

   This subcommand and the matching ``forge launch --sync`` flag
   are kept for two scenarios:

   * ``launcher_impl: bash`` (the legacy ``worker_manager.sh``
     path) -- workers there read code from local
     ``/root/AReaL`` on each host, which still needs an
     out-of-band sync.
   * ``mount_code=False`` opt-out (rare; e.g.\\ debugging FUSE).

   New deployments should leave ``mount_code`` at its default
   and skip ``--sync`` entirely.  See ``_SSHJobFleet`` docstring
   in ``forge/cli/launch.py`` for the FUSE-backed path.

Motivation
----------
Bare-metal multi-node (``launcher == BARE_METAL``) deployments assume
the repository checkout under ``FORGE_ROOT`` is byte-identical on every
host: the driver runs ``python -m forge.apps.grpo`` (which pulls in
config schema), the workers run ``python ... run_worker_loop_forever``
(which imports the same packages).  If any host drifts, typical
failure modes range from silent behavior drift (wrong reward
computed, advantage shape mismatch) to loud blow-ups mid-train
(``ConfigKeyError: Key 'reward' not in 'GRPOConfig'`` when the
driver's ``areal/api/cli_args.py`` is stale).

Before this module we handled sync by hand with ``scp`` or
``tar | ssh`` one-liners.  Over the course of a single debugging
session that cost us two full stale-code reruns.  This subcommand
codifies the dance in one place so:

* the default set of paths is explicit and reviewed (no more "I forgot
  to ship ``areal/`` this time");
* parallelism is built in (large repos over 100 GbE saturate one
  stream; N streams * ~100 MB/s each cover it);
* the command is the same shape (``forge sync``) that new users can
  copy from the README without learning ``rsync`` flags.

Why ``tar | ssh`` instead of ``rsync``
--------------------------------------
``rsync`` is the obvious choice on paper but requires the ``rsync``
binary on BOTH sides.  Our NPU container images are minimal and
frequently ship without it (we hit this exact wall in the same
session, see ``forge/docs/launcher_impl.md``).  ``tar | ssh`` only
needs ``tar`` + ``ssh``, which are guaranteed.  The trade-off is
that we ship the full set of paths every call (no delta transfer),
which is acceptable for our scale -- the relevant source tree is a
few tens of MB, and sync is a ``forge launch`` prelude, not a hot
loop.

Safety
------
* Default path set is narrow (``forge``, ``areal``) -- the editable
  source dirs.  Everything else requires an explicit ``--path``.
* Default excludes filter out ``__pycache__``, ``*.pyc``, ``.git``
  and friends.  The remote tree never gets ``.git`` accidentally.
* We never delete remote-only files.  Users who expect rsync's
  ``--delete`` semantics should open a follow-up; this module is
  intentionally additive.
"""

from __future__ import annotations

import argparse
import concurrent.futures as _cf
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parents[2]
"""Repository root, same constant as :mod:`forge.cli.launch`."""

DEFAULT_PATHS: tuple[str, ...] = ("forge", "areal")
"""Source directories synced when the user passes no ``--path``.

Chosen for the bare-metal NPU cluster: these are the only two
editable trees.  ``examples/`` and ``docs/`` are NOT included by
default -- shipping docs to workers is wasteful -- but users can
opt in per-call with ``--path examples``.
"""

DEFAULT_EXCLUDES: tuple[str, ...] = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".git",
    ".venv",
    "node_modules",
    "*.log",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "*.egg-info",
)

DEFAULT_SSH_PORT = 36000


@dataclass
class _HostResult:
    """Per-host outcome of a single sync attempt."""

    host: str
    ok: bool
    elapsed_s: float
    bytes_sent: int
    error: str | None = None


def _read_hosts(hostfile: Path) -> list[str]:
    """Parse ``ip [slots=N]`` lines out of a hostfile.

    Duplicated from :mod:`forge.cli.launch` on purpose -- keeping the
    two modules importable in isolation avoids a circular dep when
    ``launch.py`` imports this module for ``--sync``.
    """
    hosts: list[str] = []
    for raw in hostfile.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        hosts.append(line.split()[0])
    return hosts


def _resolve_paths(paths: list[str], root: Path) -> list[Path]:
    """Resolve each user-provided path against ``root`` and validate.

    ``paths`` are joined with ``root`` if relative, kept as-is if
    absolute.  Missing paths raise ``FileNotFoundError`` so we fail
    before tearing open an SSH connection and streaming nothing.
    """
    resolved: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = root / raw
        if not p.exists():
            raise FileNotFoundError(f"sync path does not exist: {p}")
        resolved.append(p)
    return resolved


def _build_tar_argv(paths: list[Path], excludes: list[str], root: Path) -> list[str]:
    """Build the local-side ``tar`` argv for the source stream.

    Each path is passed relative to ``root`` via ``tar -C`` so the
    archive entries are rooted at the repo layout (``forge/...``,
    ``areal/...``) and extract cleanly on the remote side.

    ``--warning=no-file-changed --warning=no-file-removed`` silences
    the class of warnings that show up when Python's import system
    writes ``__pycache__`` / ``*.pyc`` files concurrently with our
    tar scan (bumps the parent directory's mtime even though the
    entries themselves are excluded).  Without these flags GNU tar
    exits with rc=1 on perfectly-valid archives, which our error
    aggregator would otherwise report as a sync failure.
    """
    argv: list[str] = [
        "tar",
        "-C",
        str(root),
        "--warning=no-file-changed",
        "--warning=no-file-removed",
        "-czf",
        "-",
    ]
    for pat in excludes:
        argv.extend(["--exclude", pat])
    for p in paths:
        try:
            rel = p.resolve().relative_to(root.resolve())
        except ValueError as e:
            raise ValueError(
                f"sync path {p} must live under root {root}; "
                f"out-of-tree paths are not supported"
            ) from e
        argv.append(str(rel))
    return argv


def _build_ssh_argv(
    host: str, ssh_port: int, remote_root: Path, extra_ssh_opts: list[str] | None = None
) -> list[str]:
    """Build the remote-side ``ssh`` argv that untars into ``remote_root``.

    ``-n`` closes stdin on the SSH client side for the control
    channel; we reopen stdin on the server side via the shell's
    ``cat`` in the remote command (actually tar reads directly from
    sshd's forwarded stdin -- the point of ``-n`` on the LOCAL client
    is to make sure no interactive escape sequence eats our tar byte
    stream).  Wait -- see docstring below: we intentionally DO NOT
    pass ``-n`` here because we need stdin to carry the tar payload
    to the remote side.
    """
    opts = [
        "-p",
        str(ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    if extra_ssh_opts:
        opts = opts + list(extra_ssh_opts)
    remote_cmd = (
        f"mkdir -p {shlex.quote(str(remote_root))} && "
        f"tar -C {shlex.quote(str(remote_root))} -xzf -"
    )
    return ["ssh", *opts, host, remote_cmd]


def _sync_one_host(
    host: str,
    *,
    tar_argv: list[str],
    ssh_port: int,
    remote_root: Path,
    dry_run: bool,
    extra_ssh_opts: list[str] | None = None,
) -> _HostResult:
    """Stream the tar archive to one host.  Swallows exceptions into
    :class:`_HostResult` so the parallel driver can report ALL hosts
    instead of crashing on the first failure."""
    ssh_argv = _build_ssh_argv(
        host, ssh_port, remote_root, extra_ssh_opts=extra_ssh_opts
    )
    if dry_run:
        print(
            f"[sync] dry-run {host}:\n"
            f"    {shlex.join(tar_argv)} | {shlex.join(ssh_argv)}",
            file=sys.stderr,
        )
        return _HostResult(host=host, ok=True, elapsed_s=0.0, bytes_sent=0)

    t0 = time.monotonic()
    try:
        tar_p = subprocess.Popen(
            tar_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert tar_p.stdout is not None
        ssh_p = subprocess.Popen(
            ssh_argv,
            stdin=tar_p.stdout,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        tar_p.stdout.close()

        ssh_out, ssh_err = ssh_p.communicate(timeout=600)
        _, tar_err = tar_p.communicate(timeout=5)
        elapsed = time.monotonic() - t0

        # tar convention: rc=0 success, rc=1 warning (file changed
        # mid-read, non-fatal -- archive still valid), rc>=2 fatal.
        # The ``--warning=no-file-changed`` flag in ``_build_tar_argv``
        # normally absorbs these into rc=0; this guard is a belt on
        # top of the suspenders for unusual hosts / tar versions.
        tar_fatal = tar_p.returncode >= 2
        if ssh_p.returncode != 0 or tar_fatal:
            msg_parts = []
            if tar_fatal:
                msg_parts.append(f"tar rc={tar_p.returncode}")
                if tar_err:
                    msg_parts.append(f"tar stderr: {tar_err.decode(errors='replace')}")
            if ssh_p.returncode != 0:
                msg_parts.append(f"ssh rc={ssh_p.returncode}")
                if ssh_err:
                    msg_parts.append(f"ssh stderr: {ssh_err.decode(errors='replace')}")
            return _HostResult(
                host=host,
                ok=False,
                elapsed_s=elapsed,
                bytes_sent=0,
                error=" | ".join(msg_parts),
            )
        return _HostResult(
            host=host,
            ok=True,
            elapsed_s=elapsed,
            bytes_sent=len(ssh_out),  # sshd echoes nothing on success; placeholder
        )
    except subprocess.TimeoutExpired as e:
        return _HostResult(
            host=host,
            ok=False,
            elapsed_s=time.monotonic() - t0,
            bytes_sent=0,
            error=f"timeout: {e!r}",
        )
    except Exception as e:  # pragma: no cover -- catch-all for unknown IO
        return _HostResult(
            host=host,
            ok=False,
            elapsed_s=time.monotonic() - t0,
            bytes_sent=0,
            error=f"{type(e).__name__}: {e}",
        )


def sync_paths_to_hosts(
    hosts: list[str],
    *,
    paths: list[str] | None = None,
    excludes: list[str] | None = None,
    ssh_port: int = DEFAULT_SSH_PORT,
    remote_root: Path | str | None = None,
    local_root: Path | str | None = None,
    parallel: int = 4,
    dry_run: bool = False,
    extra_ssh_opts: list[str] | None = None,
) -> list[_HostResult]:
    """Sync ``paths`` from ``local_root`` to ``remote_root`` on each host.

    Intended as the reusable entry point both for the ``forge sync``
    CLI and for ``forge launch --sync`` (see :mod:`forge.cli.launch`).

    Parameters
    ----------
    hosts:
        Host names / IPs (one per entry).  Duplicates are preserved
        but streamed in parallel as written.
    paths:
        List of paths to sync, each either absolute or relative to
        ``local_root``.  Defaults to :data:`DEFAULT_PATHS`.
    excludes:
        ``tar --exclude`` patterns.  Defaults to
        :data:`DEFAULT_EXCLUDES`.
    ssh_port, remote_root, local_root:
        SSH port (default :data:`DEFAULT_SSH_PORT`) and repo roots
        (both default to :data:`FORGE_ROOT`).
    parallel:
        Max concurrent SSH streams.  4 is enough to cover 2x 100 GbE
        and avoid overwhelming the driver's outbound pipe.
    dry_run:
        Print the argv instead of running it.
    extra_ssh_opts:
        Appended to the SSH argv before the host name.  Useful for
        tests and for passing ``-i <keyfile>`` / jump hosts.
    """
    if not hosts:
        raise ValueError("sync_paths_to_hosts: hosts list is empty")
    local_root_p = Path(local_root) if local_root else FORGE_ROOT
    remote_root_p = Path(remote_root) if remote_root else local_root_p
    use_paths = list(paths) if paths else list(DEFAULT_PATHS)
    use_excludes = list(excludes) if excludes else list(DEFAULT_EXCLUDES)
    resolved = _resolve_paths(use_paths, local_root_p)
    tar_argv = _build_tar_argv(resolved, use_excludes, local_root_p)

    results: list[_HostResult] = []
    n_workers = max(1, min(parallel, len(hosts)))
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _sync_one_host,
                host,
                tar_argv=tar_argv,
                ssh_port=ssh_port,
                remote_root=remote_root_p,
                dry_run=dry_run,
                extra_ssh_opts=extra_ssh_opts,
            ): host
            for host in hosts
        }
        for fut in _cf.as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: hosts.index(r.host))
    return results


def _format_summary(results: list[_HostResult]) -> str:
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    lines = [f"[sync] {len(ok)}/{len(results)} hosts green"]
    for r in results:
        tag = "ok " if r.ok else "FAIL"
        lines.append(f"  [{tag}] {r.host:<20} {r.elapsed_s:5.1f}s")
        if not r.ok and r.error:
            lines.append(f"         -> {r.error}")
    if bad:
        lines.append(f"[sync] {len(bad)} host(s) failed; see errors above")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge sync",
        description=(
            "Push local source trees (forge/, areal/, ...) to every host "
            "listed in the hostfile via `tar | ssh`.  Default paths are "
            "safe for bare-metal editable installs; everything else is "
            "opt-in via --path."
        ),
    )
    p.add_argument(
        "--hostfile",
        required=True,
        help="Path to hostfile.txt (one `IP [slots=N]` per line).",
    )
    p.add_argument(
        "--ssh-port",
        type=int,
        default=DEFAULT_SSH_PORT,
        help=f"SSH port (default {DEFAULT_SSH_PORT}).",
    )
    p.add_argument(
        "--path",
        action="append",
        default=None,
        dest="paths",
        help=(
            "Path to sync, either absolute or relative to --local-root.  "
            "Repeat to add more.  Defaults to "
            f"{list(DEFAULT_PATHS)} when omitted."
        ),
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=None,
        dest="excludes",
        help=(
            "tar --exclude pattern.  Repeat to add more.  Default excludes "
            f"filter {list(DEFAULT_EXCLUDES)}."
        ),
    )
    p.add_argument(
        "--remote-root",
        default=None,
        help="Remote FORGE_ROOT (default: same as local --local-root).",
    )
    p.add_argument(
        "--local-root",
        default=None,
        help=f"Local FORGE_ROOT (default: {FORGE_ROOT}).",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Max concurrent SSH streams (default 4).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the tar/ssh argv per host instead of running it.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    hostfile = Path(args.hostfile).expanduser()
    if not hostfile.is_file():
        print(f"[sync] hostfile not found: {hostfile}", file=sys.stderr)
        return 2
    hosts = _read_hosts(hostfile)
    if not hosts:
        print(f"[sync] hostfile has no hosts: {hostfile}", file=sys.stderr)
        return 2

    try:
        results = sync_paths_to_hosts(
            hosts,
            paths=args.paths,
            excludes=args.excludes,
            ssh_port=args.ssh_port,
            remote_root=args.remote_root,
            local_root=args.local_root,
            parallel=args.parallel,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[sync] {e}", file=sys.stderr)
        return 2

    print(_format_summary(results), file=sys.stderr)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
