"""Unit tests for :mod:`forge.cli.sync`.

These tests exercise argv composition, path validation, parallelism
scheduling, and error aggregation.  They mock :mod:`subprocess` so
nothing hits the network -- exercising real ssh is covered in the
manual 2-node smoke test described in
``forge/docs/launcher_impl.md``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.cli import sync as sync_mod
from forge.cli.sync import (
    DEFAULT_EXCLUDES,
    DEFAULT_PATHS,
    _build_ssh_argv,
    _build_tar_argv,
    _HostResult,
    _read_hosts,
    _resolve_paths,
    sync_paths_to_hosts,
)

# --- small helpers -----------------------------------------------------


def _fake_proc(
    returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""
) -> MagicMock:
    """Fabricate a subprocess.Popen-shaped mock."""
    p = MagicMock()
    p.returncode = returncode
    p.stdout = MagicMock()
    p.stdout.close = MagicMock()
    p.communicate.return_value = (stdout, stderr)
    return p


# --- hostfile parsing --------------------------------------------------


class TestReadHosts:
    def test_simple_ip_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "h.txt"
        f.write_text("192.168.0.26\n192.168.0.23\n")
        assert _read_hosts(f) == ["192.168.0.26", "192.168.0.23"]

    def test_slots_suffix_dropped(self, tmp_path: Path) -> None:
        f = tmp_path / "h.txt"
        f.write_text("192.168.0.26 slots=8\n192.168.0.23  slots=8\n")
        assert _read_hosts(f) == ["192.168.0.26", "192.168.0.23"]

    def test_comments_and_blanks_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "h.txt"
        f.write_text("# header\n\n192.168.0.26 # trainer\n   \n192.168.0.23\n")
        assert _read_hosts(f) == ["192.168.0.26", "192.168.0.23"]


# --- path resolution ---------------------------------------------------


class TestResolvePaths:
    def test_relative_paths_joined_with_root(self, tmp_path: Path) -> None:
        (tmp_path / "forge").mkdir()
        (tmp_path / "areal").mkdir()
        resolved = _resolve_paths(["forge", "areal"], tmp_path)
        assert [p.name for p in resolved] == ["forge", "areal"]
        for p in resolved:
            assert p.is_absolute()

    def test_absolute_path_passes_through(self, tmp_path: Path) -> None:
        target = tmp_path / "abs"
        target.mkdir()
        resolved = _resolve_paths([str(target)], tmp_path / "unrelated")
        assert resolved == [target]

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _resolve_paths(["nope"], tmp_path)


# --- tar argv composition ---------------------------------------------


class TestBuildTarArgv:
    def test_includes_root_change_and_stream_flag(self, tmp_path: Path) -> None:
        (tmp_path / "forge").mkdir()
        argv = _build_tar_argv(
            [tmp_path / "forge"],
            list(DEFAULT_EXCLUDES),
            tmp_path,
        )
        # Order is fixed: tar, -C, root, warning flags, -czf, -, ...
        assert argv[0] == "tar"
        assert argv[1:3] == ["-C", str(tmp_path)]
        assert "-czf" in argv
        assert argv[argv.index("-czf") + 1] == "-"

    def test_suppresses_file_changed_warnings(self, tmp_path: Path) -> None:
        """``--warning=no-file-changed`` must be present -- otherwise
        concurrent ``__pycache__`` writes bump tar's rc to 1 and our
        error aggregator reports a false-positive sync failure."""
        (tmp_path / "forge").mkdir()
        argv = _build_tar_argv([tmp_path / "forge"], [], tmp_path)
        assert "--warning=no-file-changed" in argv
        assert "--warning=no-file-removed" in argv

    def test_all_excludes_rendered(self, tmp_path: Path) -> None:
        (tmp_path / "forge").mkdir()
        argv = _build_tar_argv([tmp_path / "forge"], ["__pycache__", "*.pyc"], tmp_path)
        assert argv.count("--exclude") == 2
        idxs = [i for i, v in enumerate(argv) if v == "--exclude"]
        assert [argv[i + 1] for i in idxs] == ["__pycache__", "*.pyc"]

    def test_relative_paths_inside_root_accepted(self, tmp_path: Path) -> None:
        (tmp_path / "forge").mkdir()
        (tmp_path / "areal").mkdir()
        argv = _build_tar_argv([tmp_path / "forge", tmp_path / "areal"], [], tmp_path)
        assert argv[-2:] == ["forge", "areal"]

    def test_out_of_tree_path_rejected(self, tmp_path: Path) -> None:
        other_root = tmp_path / "other"
        other_root.mkdir()
        (other_root / "x").mkdir()
        with pytest.raises(ValueError, match="must live under root"):
            _build_tar_argv([other_root / "x"], [], tmp_path / "root")


# --- ssh argv composition ---------------------------------------------


class TestBuildSshArgv:
    def test_includes_port_and_batch_flags(self) -> None:
        argv = _build_ssh_argv("192.168.0.23", 36000, Path("/root/AReaL"))
        assert argv[0] == "ssh"
        assert "-p" in argv and "36000" in argv
        assert any("BatchMode" in a for a in argv)
        assert argv[-2] == "192.168.0.23"

    def test_remote_cmd_creates_root_and_untars(self) -> None:
        argv = _build_ssh_argv("h", 22, Path("/tmp/remote root"))
        remote_cmd = argv[-1]
        assert "mkdir -p '/tmp/remote root'" in remote_cmd
        assert "tar -C '/tmp/remote root' -xzf -" in remote_cmd

    def test_extra_ssh_opts_appended_before_host(self) -> None:
        argv = _build_ssh_argv("h", 22, Path("/r"), extra_ssh_opts=["-i", "/tmp/key"])
        assert "-i" in argv
        assert argv.index("-i") < argv.index("h")


# --- end-to-end helper -------------------------------------------------


class TestSyncPathsToHosts:
    def _make_repo(self, tmp_path: Path) -> Path:
        (tmp_path / "forge").mkdir()
        (tmp_path / "areal").mkdir()
        return tmp_path

    def test_empty_hosts_rejected(self) -> None:
        with pytest.raises(ValueError, match="hosts list is empty"):
            sync_paths_to_hosts([], local_root=".")

    def test_dry_run_emits_no_subprocess(self, tmp_path: Path, capsys) -> None:
        root = self._make_repo(tmp_path)
        with patch.object(sync_mod.subprocess, "Popen") as popen_mock:
            results = sync_paths_to_hosts(
                ["h1", "h2"],
                local_root=root,
                dry_run=True,
            )
        popen_mock.assert_not_called()
        assert all(r.ok for r in results)
        err = capsys.readouterr().err
        assert "dry-run h1" in err and "dry-run h2" in err

    def test_happy_path_all_hosts_green(self, tmp_path: Path) -> None:
        root = self._make_repo(tmp_path)

        def popen_side_effect(argv, **kw):
            # First argv is the tar command; second is ssh.
            if argv and argv[0] == "tar":
                return _fake_proc(returncode=0)
            return _fake_proc(returncode=0)

        with patch.object(sync_mod.subprocess, "Popen", side_effect=popen_side_effect):
            results = sync_paths_to_hosts(
                ["h1", "h2", "h3"], local_root=root, parallel=2
            )
        assert [r.host for r in results] == ["h1", "h2", "h3"]
        assert all(r.ok for r in results)

    def test_ssh_failure_captured_into_result(self, tmp_path: Path) -> None:
        root = self._make_repo(tmp_path)

        def popen_side_effect(argv, **kw):
            if argv and argv[0] == "tar":
                return _fake_proc(returncode=0)
            return _fake_proc(returncode=255, stderr=b"ssh: connect refused")

        with patch.object(sync_mod.subprocess, "Popen", side_effect=popen_side_effect):
            results = sync_paths_to_hosts(["h1"], local_root=root)
        assert results[0].ok is False
        assert "ssh rc=255" in results[0].error
        assert "connect refused" in results[0].error

    def test_tar_failure_captured_into_result(self, tmp_path: Path) -> None:
        root = self._make_repo(tmp_path)

        def popen_side_effect(argv, **kw):
            if argv and argv[0] == "tar":
                return _fake_proc(returncode=2, stderr=b"tar: Permission denied")
            return _fake_proc(returncode=0)

        with patch.object(sync_mod.subprocess, "Popen", side_effect=popen_side_effect):
            results = sync_paths_to_hosts(["h1"], local_root=root)
        assert results[0].ok is False
        assert "tar rc=2" in results[0].error
        assert "Permission denied" in results[0].error

    def test_partial_failures_do_not_abort_siblings(self, tmp_path: Path) -> None:
        """Failures on one host must not prevent others from completing."""
        root = self._make_repo(tmp_path)

        def popen_side_effect(argv, **kw):
            if argv and argv[0] == "tar":
                return _fake_proc(returncode=0)
            # Fail ssh only for h2.  Peek at argv to tell which host.
            if "h2" in argv:
                return _fake_proc(returncode=1, stderr=b"synthetic")
            return _fake_proc(returncode=0)

        with patch.object(sync_mod.subprocess, "Popen", side_effect=popen_side_effect):
            results = sync_paths_to_hosts(
                ["h1", "h2", "h3"], local_root=root, parallel=3
            )
        by_host = {r.host: r for r in results}
        assert by_host["h1"].ok and by_host["h3"].ok
        assert by_host["h2"].ok is False
        assert "synthetic" in by_host["h2"].error

    def test_results_ordered_like_hosts_arg(self, tmp_path: Path) -> None:
        """Even when futures complete out of order, results mirror input."""
        root = self._make_repo(tmp_path)
        with patch.object(sync_mod.subprocess, "Popen", return_value=_fake_proc(0)):
            results = sync_paths_to_hosts(
                ["zzz", "aaa", "mmm"], local_root=root, parallel=3
            )
        assert [r.host for r in results] == ["zzz", "aaa", "mmm"]

    def test_custom_paths_replace_defaults(self, tmp_path: Path) -> None:
        root = tmp_path
        (root / "examples").mkdir()
        captured_argv: list[list[str]] = []

        def popen_side_effect(argv, **kw):
            captured_argv.append(list(argv))
            return _fake_proc(returncode=0)

        with patch.object(sync_mod.subprocess, "Popen", side_effect=popen_side_effect):
            sync_paths_to_hosts(["h1"], local_root=root, paths=["examples"], parallel=1)
        tar_argv = captured_argv[0]
        assert tar_argv[0] == "tar"
        assert "examples" in tar_argv
        assert "forge" not in tar_argv and "areal" not in tar_argv


# --- formatting --------------------------------------------------------


class TestFormatSummary:
    def test_all_green_summary(self) -> None:
        from forge.cli.sync import _format_summary

        s = _format_summary(
            [
                _HostResult(host="h1", ok=True, elapsed_s=1.2, bytes_sent=0),
                _HostResult(host="h2", ok=True, elapsed_s=0.9, bytes_sent=0),
            ]
        )
        assert "2/2 hosts green" in s
        assert "FAIL" not in s

    def test_mixed_summary_surfaces_errors(self) -> None:
        from forge.cli.sync import _format_summary

        s = _format_summary(
            [
                _HostResult(host="h1", ok=True, elapsed_s=1.2, bytes_sent=0),
                _HostResult(
                    host="h2",
                    ok=False,
                    elapsed_s=0.3,
                    bytes_sent=0,
                    error="ssh rc=1",
                ),
            ]
        )
        assert "1/2 hosts green" in s
        assert "FAIL" in s and "ssh rc=1" in s


# --- defaults sanity checks -------------------------------------------


class TestDefaults:
    def test_default_paths_are_editable_dirs(self) -> None:
        assert DEFAULT_PATHS == ("forge", "areal")

    def test_default_excludes_skip_build_and_vcs_artifacts(self) -> None:
        for pat in ("__pycache__", "*.pyc", ".git", "*.egg-info"):
            assert pat in DEFAULT_EXCLUDES
