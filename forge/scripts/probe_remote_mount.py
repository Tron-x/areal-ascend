"""Phase-1c probe: end-to-end Monarch ``remote_mount`` smoke test.

Goal
====

Validate that Monarch's :class:`monarch._src.job.mount_config.Mounts`
machinery -- specifically the FUSE-backed ``remote_mount`` path --
works on our 2-bare-metal-node setup, BEFORE we wire it into
:class:`forge.provisioner_ssh.ForgeSSHJob` /
:class:`forge.infra.launcher.BareMetalLauncher` for production use.

What this script does
=====================

1. Build a :class:`ForgeSSHJob` exactly the way
   ``2node_pure_training.yaml`` does (port 36000 ssh, port 22222
   monarch worker, ``pure-training`` env profile).
2. Register one ``remote_mount`` that exposes the local repo root
   (``/root/AReaL`` by default) on every worker at
   ``/mnt/forge_probe``.
3. Call ``job.state()`` -- the public method that, in addition to
   starting workers, triggers ``Mounts.ensure_open()`` and stands
   up the FUSE filesystem on each worker.
4. Spawn 1 :class:`ProbeActor` per host on the resulting host mesh.
5. Each probe actor lists the mountpoint and reads
   ``AGENTS.md`` through it, returning size + sha256.
6. Driver compares against its local copy and reports per-host.
7. Tear down via ``job.kill()`` (which both unmounts FUSE and stops
   the remote worker procs).

Pre-requisites (validated separately in Phase 1a/1b)
====================================================

* ``fuse3`` package installed on every worker container.
* ``/dev/fuse`` exposed and ``CAP_SYS_ADMIN`` granted to the
  container.

Both confirmed satisfied on 192.168.0.26 and 192.168.0.23 -- see
the Phase-1a/1b session log.

Exit codes
==========

* ``0``  -- mount works, hash matches on every host.  Phase 1c
            passes; safe to integrate ``remote_mount`` into
            ``ForgeSSHJob`` for production.
* ``2``  -- pre-flight (local marker file missing, etc.)
* ``20`` -- ``job.state()`` raised: workers didn't start, or FUSE
            mount could not be established on at least one worker.
            Likely cause: container caps / libfuse3 version /
            ``_mount_worker`` subprocess crash.
* ``30`` -- spawn or probe-actor RPC failed.
* ``40`` -- mount visible but file content mismatch on at least one
            host (FUSE block-cache or incremental-transfer bug).

Run it
======

::

    cd /root/AReaL
    python forge/scripts/probe_remote_mount.py

Cleanup is best-effort: on KeyboardInterrupt the ``finally`` block
still calls ``job.kill()`` so FUSE mounts and worker procs are torn
down even on Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import socket
import sys
import time
from pathlib import Path

_DEFAULT_PYTHONPATH_PARTS = [
    "/root/AReaL",
    "/root/torchstore",
    "/root/monarch/python",
]
for _p in _DEFAULT_PYTHONPATH_PARTS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from monarch.actor import Actor, endpoint  # noqa: E402

from forge.provisioner_ssh import ForgeSSHJob  # noqa: E402


class ProbeActor(Actor):
    """One per host.  Verifies the mount is visible and reads through it."""

    @endpoint
    def probe(self, mountpoint: str, marker_path: str) -> dict:
        # Imports are inside the endpoint so the actor's runtime env --
        # not the driver's -- is used.  socket.gethostname() is the
        # cheapest "which worker am I" identifier; we don't need
        # fully-qualified names because the driver already knows the
        # host list and we just want to disambiguate per-rank lines.
        import hashlib as _h
        import os as _os
        import socket as _socket

        out: dict = {"host": _socket.gethostname()}
        try:
            if not _os.path.isdir(mountpoint):
                out.update(
                    ok=False,
                    stage="listdir",
                    error=f"{mountpoint} is not a directory",
                )
                return out
            out["mount_listing_head"] = sorted(_os.listdir(mountpoint))[:10]

            full = _os.path.join(mountpoint, marker_path)
            if not _os.path.isfile(full):
                out.update(ok=False, stage="readfile", error=f"missing: {full}")
                return out
            with open(full, "rb") as f:
                data = f.read()
            out["size"] = len(data)
            out["sha256"] = _h.sha256(data).hexdigest()
            out["ok"] = True
            return out
        except Exception as e:  # noqa: BLE001
            out.update(ok=False, stage="exception", error=f"{type(e).__name__}: {e}")
            return out


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def _wait_workers_ready(hosts, port: int, deadline_s: float) -> bool:
    """Poll TCP listener on every host until all are up or deadline.

    Mirrors :func:`forge.scripts.poc_ssh_job._tcp_reachable` -- kept inline
    here so the probe is self-contained (it's an exit-code-only diagnostic
    we'd hate to see broken by a refactor of the POC harness).
    """
    t0 = time.time()
    ready = {h: False for h in hosts}
    while time.time() - t0 < deadline_s:
        for h in hosts:
            if not ready[h] and _tcp_reachable(h, port, timeout=2.0):
                ready[h] = True
                print(
                    f"  {h}:{port} READY (+{time.time() - t0:.1f}s)",
                    flush=True,
                )
        if all(ready.values()):
            return True
        time.sleep(1)
    missing = [h for h, r in ready.items() if not r]
    print(f"  not ready: {missing}", flush=True)
    return False


def _normalise_results(results):
    """Flatten Monarch ``.call()`` return into a plain list of payloads.

    Mirrors the helper in ``forge.apps.inference_bench`` /
    ``forge.apps.titan_pretrain``.  ``.call()`` may return either an
    iterable of ``(point, payload)`` tuples (multi-rank) or a bare
    payload (single rank) -- callers always want the flat list.
    """
    try:
        items = list(results)
    except TypeError:
        return [results]
    out = []
    for entry in items:
        if isinstance(entry, tuple) and len(entry) == 2:
            out.append(entry[1])
        else:
            out.append(entry)
    return out


async def _run(args: argparse.Namespace) -> int:
    source_dir = Path(args.source).resolve()
    if not source_dir.is_dir():
        print(f"[probe] source dir missing: {source_dir}", file=sys.stderr)
        return 2
    marker_path = args.marker
    local_marker = source_dir / marker_path
    if not local_marker.is_file():
        print(f"[probe] local marker missing: {local_marker}", file=sys.stderr)
        return 2
    local_bytes = local_marker.read_bytes()
    local_size = len(local_bytes)
    local_sha = hashlib.sha256(local_bytes).hexdigest()
    print(
        f"[probe] local marker {local_marker}: size={local_size} sha={local_sha[:12]}",
        flush=True,
    )

    hosts = args.hosts
    print("=" * 72)
    print(f" remote_mount probe  ({len(hosts)} host(s))")
    print(f"   source        = {source_dir}")
    print(f"   mountpoint    = {args.mountpoint}")
    print(f"   marker        = {marker_path}")
    print(f"   transfer_mode = {args.transfer_mode}")
    print(f"   ssh_port      = {args.ssh_port}")
    print(f"   worker_port   = {args.worker_port}")
    print("=" * 72, flush=True)

    job = ForgeSSHJob(
        cann_home=args.cann,
        conda_env=args.conda_env,
        conda_bin=args.conda_bin,
        areal_root=args.areal_root,
        python_exe="python",
        ssh_args=["-p", str(args.ssh_port), "-o", "StrictHostKeyChecking=no"],
        monarch_port=args.worker_port,
        profile="pure-training",
    )
    job.add_mesh("probe", hosts)

    # Configure the mount BEFORE state(): remote_mount() only registers
    # config in self._mounts._remote_entries.  The actual FUSE setup
    # happens inside state() -> _mounts.ensure_open() -> spawned
    # ``_mount_worker`` subprocess -> FUSEActor on each host.
    #
    # python_exe=None tells JobTrait NOT to rewrite the worker's python
    # path to ``{mountpoint}/.venv/bin/python``.  We use a conda env
    # (``monarch_ascend``) at a path completely unrelated to the source
    # tree, and the default rewrite would (a) fail validation
    # (no .venv inside the source dir) and (b) be wrong even if it
    # passed -- the worker MUST keep using the conda python that has
    # torch_npu installed.
    #
    # transfer_mode="actor" routes block transfer through Monarch's
    # actor message-passing instead of the default ``rust_tls`` path.
    # rust_tls needs Meta-internal TLS certs (cert_path or the default
    # ``fb-tls`` paths) which we obviously don't have on bare-metal
    # NPU hosts.  ``actor`` mode is described in remotemount.py as
    # "slower but works without custom TLS certs (e.g. GitHub CI,
    # local testing)" -- exactly our setup.
    job.remote_mount(
        source=str(source_dir),
        mntpoint=args.mountpoint,
        meshes=["probe"],
        python_exe=None,
        transfer_mode=args.transfer_mode,
    )

    try:
        # Two-step start: apply() launches workers asynchronously
        # (SSH commands are non-blocking; remote python takes 4-15s to
        # actually bind the monarch worker port).  We then poll TCP
        # readiness on every host BEFORE calling state(), because
        # state()'s internal SetClientConfig has a 30s timeout that
        # races CANN runtime init -- losing this race manifests as
        # exactly the "session ... last connected Xs ago, disconnected"
        # error we hit on the first probe attempt.  poc_ssh_job.py
        # uses the same wait pattern; this just reuses it inline.
        print(
            "[probe] step 1/3: job.apply() (start workers, non-blocking) ...",
            flush=True,
        )
        try:
            job.apply()
        except Exception as e:  # noqa: BLE001
            print(f"[probe] FAIL stage=apply(): {e!r}", file=sys.stderr)
            return 20

        print(
            f"[probe] step 2/3: poll TCP {args.worker_port} on all hosts (deadline {args.ready_timeout}s) ...",
            flush=True,
        )
        if not _wait_workers_ready(hosts, args.worker_port, args.ready_timeout):
            print(
                "[probe] FAIL: not all workers became reachable within "
                f"{args.ready_timeout}s",
                file=sys.stderr,
            )
            return 20

        # Now state() takes the fast path: _running is already set, so
        # _connect() skips apply() and goes straight to _state() ->
        # attach_to_workers.  The mount setup (_mounts.ensure_open) runs
        # AFTER attach completes, so the mount_worker subprocess inherits
        # a known-healthy worker mesh.
        print(
            "[probe] step 3/3: job.state() -- attach + apply remote_mount ...",
            flush=True,
        )
        try:
            state = job.state(cached_path=None)
        except Exception as e:  # noqa: BLE001
            print(f"[probe] FAIL stage=state(): {e!r}", file=sys.stderr)
            return 20

        host_mesh = state._hosts["probe"]
        await host_mesh.initialized
        print(f"[probe] host_mesh.size() = {host_mesh.size()}", flush=True)

        # 1 actor per host -- the mountpoint is a per-host filesystem
        # so we don't need multiple procs per host to test reads.
        # ``per_host={"gpus": 1}`` matches the convention in
        # forge.apps.inference_bench so proc_mesh outputs look uniform
        # across actor types.
        proc_mesh = host_mesh.spawn_procs(per_host={"gpus": 1})
        actor = proc_mesh.spawn("probe", ProbeActor)

        try:
            results = await actor.probe.call(
                mountpoint=args.mountpoint,
                marker_path=marker_path,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[probe] FAIL stage=actor-call: {e!r}", file=sys.stderr)
            return 30
        payloads = _normalise_results(results)

        print("=" * 72, flush=True)
        for p in payloads:
            print(f"  {p}", flush=True)
        print("=" * 72, flush=True)

        bad = [p for p in payloads if not isinstance(p, dict) or not p.get("ok")]
        if bad:
            print(
                f"[probe] FAIL: {len(bad)}/{len(payloads)} host(s) reported failures",
                file=sys.stderr,
            )
            for p in bad:
                if isinstance(p, dict):
                    print(
                        f"  host={p.get('host', '?')} "
                        f"stage={p.get('stage', '?')} "
                        f"error={p.get('error', '?')}",
                        file=sys.stderr,
                    )
                else:
                    print(f"  non-dict payload: {p!r}", file=sys.stderr)
            # If the only failure was at "exception"/"actor" stage -> probe
            # itself broken.  Otherwise the mount itself is wrong.
            if any(
                isinstance(p, dict) and p.get("stage") in ("exception", "actor")
                for p in bad
            ):
                return 30
            return 20

        mismatched = [
            p for p in payloads if p["size"] != local_size or p["sha256"] != local_sha
        ]
        if mismatched:
            for p in mismatched:
                print(
                    f"[probe] FAIL host={p['host']}: "
                    f"size={p['size']} (local {local_size}), "
                    f"sha={p['sha256'][:12]} (local {local_sha[:12]})",
                    file=sys.stderr,
                )
            return 40

        print(f" remote_mount probe: OK on {len(payloads)} host(s)")
        print(" Implication: FUSE backend works in worker containers.")
        print(" Phase 1c PASS -- safe to integrate remote_mount into")
        print(" provisioner_ssh.py / BareMetalLauncher.")
        print("=" * 72)
        return 0
    finally:
        # job.kill() (public) -- not _kill -- so JobTrait.kill calls
        # _mounts.ensure_stopped() before tearing down workers.  Otherwise
        # the _mount_worker subprocess survives and leaks the FUSE mount.
        try:
            print("[probe] job.kill(): unmount + stop workers ...", flush=True)
            job.kill()
        except Exception as e:  # noqa: BLE001
            print(
                f"[probe] cleanup job.kill() raised {e!r} (non-fatal)", file=sys.stderr
            )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Phase-1c probe of monarch remote_mount on ForgeSSHJob",
    )
    p.add_argument(
        "--hosts",
        nargs="+",
        default=["192.168.0.26", "192.168.0.23"],
        help="Worker hosts to mount onto.",
    )
    p.add_argument("--ssh-port", type=int, default=36000)
    p.add_argument("--worker-port", type=int, default=22222)
    p.add_argument(
        "--ready-timeout",
        type=float,
        default=60.0,
        help=(
            "How long (s) to wait for every worker's monarch port to "
            "open before giving up.  CANN runtime init alone takes "
            "~10s on idle NPUs; bump this if you see partial readiness."
        ),
    )
    p.add_argument(
        "--mountpoint",
        default="/mnt/forge_probe",
        help=(
            "Where on the worker to mount the source dir.  Avoid /tmp "
            "to make sure cross-mount-point ops aren't being shadowed; "
            "avoid the source path itself (/root/AReaL) so a stale "
            "FUSE handle can't accidentally hide the local checkout."
        ),
    )
    p.add_argument(
        "--source",
        default="/root/AReaL/forge",
        help=(
            "Local directory to expose on workers via FUSE.  Defaults "
            "to the small ``forge/`` subtree (~3MB) instead of the "
            "full repo (~2.5GB including .venv) -- this is a probe, "
            "we just need to verify the FUSE plumbing, not transfer "
            "the whole world."
        ),
    )
    p.add_argument(
        "--marker",
        default="README.md",
        help=(
            "Path RELATIVE to --source that the probe actor reads "
            "through the mount.  Driver compares its sha256 against "
            "the local file's to confirm the mount is serving the "
            "actual source content."
        ),
    )
    p.add_argument(
        "--transfer-mode",
        choices=("actor", "rust_tls"),
        default="actor",
        help=(
            "Block transfer mode.  ``actor`` uses Monarch actor "
            "message-passing (no TLS certs needed); ``rust_tls`` "
            "needs Meta-internal certs we don't have on bare-metal "
            "NPU hosts."
        ),
    )
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
