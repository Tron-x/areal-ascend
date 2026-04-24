"""Unit tests for ``ForgeActor._resolve_hosts`` (R1.5b).

Verifies the three-way decision that unifies actor placement through
the launcher's ``get_host_mesh(mesh_name)`` path:

1. caller passed an explicit ``hosts=N``  -> honor verbatim (including 0)
2. caller left ``hosts=None`` + remote launcher active -> default to 1
3. caller left ``hosts=None`` + no launcher (local run) -> stay None

No monarch runtime, no distributed; the provisioner lookup is mocked.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import patch

import pytest


def _install_monarch_stubs() -> None:
    """Register no-op stubs for the monarch imports that
    ``forge.actors.base`` does at module load so this test can run
    in environments without the real monarch wheel."""
    if "monarch" in sys.modules:
        return

    monarch = types.ModuleType("monarch")
    monarch_actor = types.ModuleType("monarch.actor")
    monarch_src = types.ModuleType("monarch._src")
    monarch_src_actor = types.ModuleType("monarch._src.actor")
    monarch_actor_mesh = types.ModuleType("monarch._src.actor.actor_mesh")

    class _FakeActor:
        def __init__(self, *args, **kwargs):
            pass

    def _current_rank():
        return types.SimpleNamespace(rank=0)

    def _current_size():
        return {"procs": 1}

    def _endpoint(fn):
        return fn

    monarch_actor.Actor = _FakeActor
    monarch_actor.current_rank = _current_rank
    monarch_actor.current_size = _current_size
    monarch_actor.endpoint = _endpoint

    class _ActorMesh:
        pass

    monarch_actor_mesh.ActorMesh = _ActorMesh

    sys.modules["monarch"] = monarch
    sys.modules["monarch.actor"] = monarch_actor
    sys.modules["monarch._src"] = monarch_src
    sys.modules["monarch._src.actor"] = monarch_src_actor
    sys.modules["monarch._src.actor.actor_mesh"] = monarch_actor_mesh


def _install_provisioner_stub() -> None:
    """Install a minimal ``forge.provisioner`` shim so importing
    ``forge.actors.base`` succeeds without pulling in the real
    provisioner (which needs monarch + vllm + torch_npu)."""
    if "forge.provisioner" in sys.modules and hasattr(
        sys.modules["forge.provisioner"], "_get_provisioner"
    ):
        return

    stub = types.ModuleType("forge.provisioner")

    async def _get_provisioner():
        raise RuntimeError(
            "test stub: patch forge.actors.base._get_provisioner in each test"
        )

    async def _register_actor(_actor):
        pass

    async def _register_service(_service):
        pass

    async def _stop_proc_mesh(_pm):
        pass

    async def _get_proc_mesh(_cfg):
        raise RuntimeError("test stub: should not be reached in these tests")

    stub._get_provisioner = _get_provisioner
    stub.get_proc_mesh = _get_proc_mesh
    stub.register_actor = _register_actor
    stub.register_service = _register_service
    stub.stop_proc_mesh = _stop_proc_mesh
    sys.modules["forge.provisioner"] = stub


_install_monarch_stubs()
_install_provisioner_stub()


# forge.types exports ProcessConfig/ServiceConfig/Launcher; importing the
# base module transitively pulls it in, so no extra stub needed.
from forge.actors.base import ForgeActor  # noqa: E402


class _MockProvisioner:
    def __init__(self, launcher):
        self.launcher = launcher


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestResolveHosts:
    def test_explicit_hosts_int_is_honored(self):
        """Caller passed ``hosts=3`` explicitly -- no provisioner
        lookup, no autodetect, just honor the number."""
        cls = ForgeActor.options(hosts=3, mesh_name="m")
        # We don't even need to mock the provisioner; the early
        # return should skip that branch entirely.
        result = _run(cls._resolve_hosts())
        assert result == 3

    def test_explicit_hosts_zero_is_honored(self):
        """``hosts=0`` is the escape hatch for "use this_host() even
        though a remote launcher is active" -- must NOT be upgraded
        to 1 by the auto-default."""
        cls = ForgeActor.options(hosts=0, mesh_name="m")

        async def _fake_prov():
            return _MockProvisioner(launcher=object())  # launcher IS active

        # ``_get_provisioner`` is lazy-imported inside ``_resolve_hosts``
        # so we patch it at the source module, not the consumer's.
        with patch("forge.provisioner._get_provisioner", _fake_prov):
            result = _run(cls._resolve_hosts())
        assert result == 0

    def test_none_hosts_with_remote_launcher_defaults_to_1(self):
        """R1.5b: ``hosts=None`` + active launcher -> 1 so the actor
        goes through the launcher's ``get_host_mesh(mesh_name)``
        path instead of falling back to ``this_host()``."""
        cls = ForgeActor.options(hosts=None, mesh_name="trainer")

        async def _fake_prov():
            return _MockProvisioner(launcher=object())

        with patch("forge.provisioner._get_provisioner", _fake_prov):
            result = _run(cls._resolve_hosts())
        assert result == 1

    def test_none_hosts_without_launcher_stays_none(self):
        """Pure local run (no launcher configured) keeps the
        historical ``this_host()`` fallback via ``hosts=None``.
        Changing this would break single-machine dev loops."""
        cls = ForgeActor.options(hosts=None, mesh_name="trainer")

        async def _fake_prov():
            return _MockProvisioner(launcher=None)

        with patch("forge.provisioner._get_provisioner", _fake_prov):
            result = _run(cls._resolve_hosts())
        assert result is None

    def test_default_options_use_none_hosts(self):
        """Sanity check: the ``options()`` classmethod default for
        ``hosts`` is ``None`` -- otherwise the R1.5b auto-default
        wouldn't kick in for callers that just say ``options(procs=1)``."""
        cls = ForgeActor.options()
        assert cls.hosts is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
