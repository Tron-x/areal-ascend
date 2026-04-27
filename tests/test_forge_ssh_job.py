"""Unit tests for :class:`forge.provisioner_ssh.ForgeSSHJob`.

Scope deliberately narrow: these tests verify the things we CAN
check without a live cluster (env-preamble shape, override
contracts, kill-invokes-super, SSH command composition).  Live
behavior on real hosts is exercised by ``forge/scripts/poc_ssh_job.py``
and by the multi-node integration suite.

The single goal of this file is: if someone refactors
``ForgeSSHJob`` and breaks the preamble in a way that keeps the
code importable but wrong (e.g. drops the ``unset
TORCHSTORE_RDMA_ENABLED`` line or changes ``HCCL_INTRA_ROCE_ENABLE``
semantics), these tests fail immediately rather than surfacing 15
minutes into the next multi-node run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.provisioner_ssh import ForgeSSHJob

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def job() -> ForgeSSHJob:
    """Default ``ForgeSSHJob`` with the production-shaped args.

    Values match the two-node NPU POC (``forge/scripts/poc_ssh_job.py``
    + ``forge/configs/hostfile.txt``) so tests guard against drift
    of either side.
    """
    return ForgeSSHJob(
        cann_home="/usr/local/Ascend/cann-9.0.0-beta.1",
        conda_env="monarch_ascend",
        conda_bin="/root/miniconda3/bin/conda",
        areal_root="/root/AReaL",
        python_exe="python",
        ssh_args=["-p", "36000", "-o", "StrictHostKeyChecking=no"],
        monarch_port=22222,
    )


# ---------------------------------------------------------------------------
# Env preamble shape -- the bash string must stay parseable and must
# preserve the invariants that the pre-refactor worker_manager.sh relied on.
# ---------------------------------------------------------------------------


class TestEnvPreamble:
    def test_preamble_is_nonempty_semi_joined(self, job: ForgeSSHJob):
        preamble = job._env_preamble()
        assert preamble
        # All parts should be ``;``-joined (not newlines, which would
        # need ``bash -c`` and break the parent's ssh-exec model).
        assert "\n" not in preamble
        # Sanity: there should be >= 10 distinct commands in the chain,
        # otherwise someone stripped out our HCCL/HiXL envs.
        assert len(preamble.split(";")) >= 15

    def test_preamble_sources_cann(self, job: ForgeSSHJob):
        assert "/usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh" in job._env_preamble()

    def test_preamble_activates_conda(self, job: ForgeSSHJob):
        p = job._env_preamble()
        assert "/root/miniconda3/bin/conda" in p
        assert "conda activate monarch_ascend" in p

    def test_preamble_cds_into_areal_root(self, job: ForgeSSHJob):
        # Last command should be the cd so the subsequent python
        # launches from the right PYTHONPATH-ahead repo.
        assert job._env_preamble().rstrip().endswith("cd /root/AReaL")

    def test_preamble_sets_hixl_transport(self, job: ForgeSSHJob):
        p = job._env_preamble()
        # Transport pinning: MUST be RoCE on bare-metal cross-node --
        # HCCS is intra-supernode only.  This is the single knob that
        # most often silently breaks cross-node HiXL if a future
        # refactor drops it.
        assert "MONARCH_HIXL_TRANSPORT=roce" in p
        # HCCL_INTRA_ROCE_ENABLE=1 must be pre-python (CANN reads once
        # at library load time); Rust-side sets are too late.
        assert "HCCL_INTRA_ROCE_ENABLE=1" in p

    def test_preamble_widens_hccl_port_range(self, job: ForgeSSHJob):
        # Avoids the HcclCommPrepare ret=0x13 collision on default 16666
        # when HiXL and HCCL coexist in the same process.
        assert "HCCL_NPU_SOCKET_PORT_RANGE" in job._env_preamble()

    def test_preamble_keeps_torchstore_rdma_live(self, job: ForgeSSHJob):
        p = job._env_preamble()
        # torchstore RDMA must NOT be opt-out-forced; the MonarchRDMA
        # transport is how we get HiXL.
        assert "unset TORCHSTORE_RDMA_ENABLED" in p
        # NPU-resident staging pool is required for 2-MB alignment.
        assert "TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0" in p
        assert "TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:0" in p

    def test_preamble_sets_pythonpath_in_order(self, job: ForgeSSHJob):
        # PYTHONPATH order matters: repo first so local patches to
        # monarch/torchstore take effect over any site-packages copy.
        p = job._env_preamble()
        assert "/root/AReaL" in p
        assert "/root/monarch/python" in p
        assert "/root/torchstore" in p
        # The AReaL entry must precede the monarch entry in the export.
        export_line = next((x for x in p.split(";") if "PYTHONPATH=" in x), None)
        assert export_line is not None
        assert export_line.index("/root/AReaL") < export_line.index(
            "/root/monarch/python"
        )

    def test_pool_mb_is_overridable(self):
        # Large-model override: user may bump the 8 GB default.
        j = ForgeSSHJob(
            cann_home="/opt/cann",
            conda_env="env",
            conda_bin="/usr/bin/conda",
            areal_root="/work",
            torchstore_pool_mb=32768,
        )
        # The exported default form is
        # ``export VAR="${VAR:-32768}"`` so caller-set envs still win.
        assert "TORCHSTORE_MONARCH_RDMA_POOL_MB" in j._env_preamble()
        assert "32768" in j._env_preamble()


# ---------------------------------------------------------------------------
# Profile-aware env block.  The two profiles must ALWAYS share the
# common block (CANN, conda, PYTHONPATH, generic HCCL) and ONLY
# diverge on the HiXL/torchstore exports -- mirroring
# ``worker_manager.sh::profile_env_block``.
# ---------------------------------------------------------------------------


class TestProfile:
    def _make(self, profile: str | None = None) -> ForgeSSHJob:
        kw: dict = {
            "cann_home": "/usr/local/Ascend/cann-9.0.0-beta.1",
            "conda_env": "monarch_ascend",
            "conda_bin": "/root/miniconda3/bin/conda",
            "areal_root": "/root/AReaL",
            "monarch_port": 22222,
        }
        if profile is not None:
            kw["profile"] = profile
        return ForgeSSHJob(**kw)

    def test_default_profile_is_hixl_coexist(self):
        # Backward-compat guard: omitting ``profile=`` must keep the
        # pre-refactor HiXL-coexist env so existing GRPO callers
        # don't break silently.
        job = self._make()
        assert job._profile == "hixl-coexist"

    def test_invalid_profile_raises(self):
        # Misspelled profile names must fail at construction time
        # rather than silently flipping to a default and breaking
        # transport selection 5 minutes into the run.
        with pytest.raises(ValueError, match="profile="):
            self._make(profile="hixl_coexist")  # underscore vs. dash

    # --- pure-training -----------------------------------------------------

    def test_pure_training_omits_hixl_transport_pin(self):
        # pure-training MUST let HCCL auto-select transport.  Any
        # MONARCH_HIXL_TRANSPORT export defeats the whole point of
        # the profile (forces RoCE intra-host => 10x bandwidth loss).
        p = self._make(profile="pure-training")._env_preamble()
        assert "MONARCH_HIXL_TRANSPORT" not in p
        assert "HCCL_INTRA_ROCE_ENABLE" not in p

    def test_pure_training_omits_hccl_port_range(self):
        # The 60000-60255 port range exists only to dodge the
        # HiXL/HCCL collision on 16666; pure-training has no HiXL,
        # so HCCL should keep the default (16666) for fastest path.
        p = self._make(profile="pure-training")._env_preamble()
        assert "HCCL_NPU_SOCKET_PORT_RANGE" not in p

    def test_pure_training_omits_torchstore_exports(self):
        p = self._make(profile="pure-training")._env_preamble()
        assert "TORCHSTORE_MONARCH_RDMA" not in p
        assert "TORCHSTORE_RDMA_ENABLED" not in p

    def test_pure_training_keeps_common_env(self):
        # Common block (CANN, conda, PYTHONPATH, HCCL_DEBUG, HF
        # offline) MUST still be present -- pure-training only
        # strips the HiXL/torchstore overlay, not the basic worker
        # bootstrap.
        p = self._make(profile="pure-training")._env_preamble()
        assert "/usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh" in p
        assert "conda activate monarch_ascend" in p
        assert "/root/AReaL" in p
        assert 'HCCL_DEBUG="${HCCL_DEBUG:-INFO}"' in p
        assert "HCCL_CONNECT_TIMEOUT=120" in p
        assert "HF_DATASETS_OFFLINE" in p

    def test_pure_training_ends_with_cd(self):
        # cd MUST be the last command so any subsequent python
        # invocation runs from the repo root regardless of the
        # profile-specific exports above it.
        p = self._make(profile="pure-training")._env_preamble().rstrip()
        assert p.endswith("cd /root/AReaL")

    # --- hixl-coexist (regression guards: re-assert the canonical
    # HiXL exports through the new code path) -------------------------------

    def test_hixl_coexist_includes_all_hixl_exports(self):
        # If a future refactor accidentally moves any of these into
        # _common_env_parts() or removes them, GRPO weight sync
        # silently breaks.  Pin the full list here.
        p = self._make(profile="hixl-coexist")._env_preamble()
        for required in (
            "MONARCH_HIXL_TRANSPORT=roce",
            "HCCL_INTRA_ROCE_ENABLE=1",
            "HCCL_NPU_SOCKET_PORT_RANGE",
            "unset TORCHSTORE_RDMA_ENABLED",
            "TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0",
            "TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:0",
            "TORCHSTORE_MONARCH_RDMA_POOL_MB",
        ):
            assert required in p, f"hixl-coexist preamble missing: {required}"

    def test_profiles_share_common_block_byte_for_byte(self):
        # The two profiles diverge ONLY on the profile-specific
        # block -- the common (CANN/conda/PYTHONPATH/...) block must
        # be identical between them.  Catch accidental drift where
        # someone "fixes" a common-block export only in one profile.
        common_pure = self._make(profile="pure-training")._common_env_parts()
        common_hixl = self._make(profile="hixl-coexist")._common_env_parts()
        assert common_pure == common_hixl


# ---------------------------------------------------------------------------
# _start_host -- SSH command composition.
# ---------------------------------------------------------------------------


class TestStartHost:
    def test_start_host_invokes_ssh_with_env_prefix(self, job: ForgeSSHJob):
        """_start_host must spawn ``ssh <ssh_args> host -n "<preamble>; exec python -c ..."``.

        The ``exec`` is load-bearing: without it the process tree on
        the remote has two levels (sshd -> bash -> python) and Monarch's
        PID tracking on the local side points at the bash, not the
        python.  A regression here wouldn't show up in unit tests
        except this one.
        """
        with patch("forge.provisioner_ssh.subprocess.Popen") as popen_mock:
            popen_mock.return_value = MagicMock(pid=12345)
            state = job._start_host("192.168.0.26")

        assert state.pid == 12345
        assert state.channel == "tcp://192.168.0.26:22222"

        argv = popen_mock.call_args[0][0]
        # argv shape: ['ssh', '-p', '36000', '-o', ..., '192.168.0.26', '-n', '<cmd>']
        assert argv[0] == "ssh"
        assert "192.168.0.26" in argv
        assert "-n" in argv
        assert any(a == "-p" for a in argv)

        cmd = argv[-1]
        # Preamble + exec + python -c
        assert "conda activate monarch_ascend" in cmd
        assert "exec " in cmd
        # The python startup must reference run_worker_loop_forever and
        # the host-specific tcp address.
        assert "run_worker_loop_forever" in cmd
        assert "tcp://192.168.0.26:22222" in cmd

    def test_start_host_uses_new_session(self, job: ForgeSSHJob):
        # start_new_session=True matches parent SSHJob behavior; keeping
        # it so our override doesn't silently change process-group semantics.
        with patch("forge.provisioner_ssh.subprocess.Popen") as popen_mock:
            popen_mock.return_value = MagicMock(pid=1)
            job._start_host("10.0.0.1")
        kwargs = popen_mock.call_args.kwargs
        assert kwargs.get("start_new_session") is True


# ---------------------------------------------------------------------------
# _kill -- calls super() + remote pgrep/kill fallback.
# ---------------------------------------------------------------------------


class TestKill:
    def test_kill_calls_super_then_remote_pgrep(self, job: ForgeSSHJob):
        """_kill must:

        1. Invoke ``super()._kill()`` (which SIGKILLs the local ssh pids).
        2. For every tracked host, issue an ssh to pgrep-kill the
           remote ``run_worker_loop_forever`` (self-excluding pattern).

        Ordering matters: if we pgrep-kill FIRST and then super()
        kills the local ssh clients, the local ssh may already have
        reaped its remote parent -- harmless but wastes a probe.
        """
        # Pretend the parent tracked two hosts.
        from monarch._src.job.job import ProcessState

        job._host_to_pid = {
            "192.168.0.26": ProcessState(11111, "tcp://192.168.0.26:22222"),
            "192.168.0.23": ProcessState(22222, "tcp://192.168.0.23:22222"),
        }

        with (
            patch("forge.provisioner_ssh.subprocess.run") as run_mock,
            patch("monarch._src.job.job.SSHJob._kill") as super_kill_mock,
        ):
            run_mock.return_value = MagicMock(returncode=0)
            job._kill()

        assert super_kill_mock.called, "super()._kill() must be called"
        # One ssh per host; no more, no fewer.
        assert run_mock.call_count == 2, run_mock.call_args_list

        hosts_contacted = set()
        for call in run_mock.call_args_list:
            argv = call.args[0]
            assert argv[0] == "ssh"
            # Self-excluding bracket-trick pattern is critical: without
            # it the pkill command kills its own invoking shell and
            # misses the actual python worker.  Guard it.
            assert "[r]un_worker_loop_forever" in argv[-1]
            hosts_contacted.add(argv[-2])  # host comes right before the command
        assert hosts_contacted == {"192.168.0.26", "192.168.0.23"}

    def test_kill_tolerates_remote_failure(self, job: ForgeSSHJob):
        """Per-host cleanup failures must NOT raise.

        ``_kill`` is called from teardown/``finally``/signal handlers.
        If it raised on a single unreachable host, we'd lose cleanup
        on every OTHER host in the fleet.
        """
        from monarch._src.job.job import ProcessState

        job._host_to_pid = {
            "unreachable.example.com": ProcessState(1, "tcp://x:1"),
        }
        with (
            patch(
                "forge.provisioner_ssh.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["ssh"], timeout=15),
            ),
            patch("monarch._src.job.job.SSHJob._kill"),
        ):
            # Must not raise; the guard is the whole point.
            job._kill()


# ---------------------------------------------------------------------------
# Launcher CLI integration -- the YAML reader.
# ---------------------------------------------------------------------------


class TestYAMLReader:
    def test_ssh_job_value_parsed(self, tmp_path: Path):
        from forge.cli.launch import _read_launcher_impl_from_yaml

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("launcher:\n  launcher_impl: ssh_job\n")
        assert _read_launcher_impl_from_yaml(cfg) == "ssh_job"

    def test_bash_default_when_absent(self, tmp_path: Path):
        from forge.cli.launch import _read_launcher_impl_from_yaml

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("launcher: {}\n")
        assert _read_launcher_impl_from_yaml(cfg) == "bash"

    def test_malformed_yaml_defaults_to_bash(self, tmp_path: Path):
        # Safety net: a broken YAML must NOT silently flip to ssh_job.
        from forge.cli.launch import _read_launcher_impl_from_yaml

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(": : invalid : :\n")
        assert _read_launcher_impl_from_yaml(cfg) == "bash"

    def test_missing_launcher_block_defaults(self, tmp_path: Path):
        from forge.cli.launch import _read_launcher_impl_from_yaml

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("something_else: 1\n")
        assert _read_launcher_impl_from_yaml(cfg) == "bash"


# ---------------------------------------------------------------------------
# LauncherConfig field presence
# ---------------------------------------------------------------------------


class TestLauncherConfigSchema:
    def test_default_is_bash(self):
        from forge.core.types import LauncherConfig

        cfg = LauncherConfig()
        assert cfg.launcher_impl == "bash"

    def test_ssh_job_roundtrips(self):
        from forge.core.types import LauncherConfig

        cfg = LauncherConfig(launcher_impl="ssh_job")
        assert cfg.launcher_impl == "ssh_job"
