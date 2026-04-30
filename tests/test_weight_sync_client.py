"""Unit tests for ``areal.weight_sync.vllm_ext.client.WeightSyncClient``.

These tests stand up a minimal FastAPI app that mimics the contract of
``areal.weight_sync.vllm_ext.server_router`` (just the URL paths +
JSON envelope, no real vLLM workers), then exercise the client's HTTP
plumbing without a live ``torch.distributed`` group.

Run::

    cd /root/AReaL && python -m pytest tests/test_weight_sync_client.py -v
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from unittest.mock import patch

import pytest

# FastAPI / uvicorn are vLLM transitive deps -- safe to require.
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

_RESPONSE_OK = {"success": True, "message": "ok"}
_RESPONSE_FAIL = {"success": False, "message": "synthetic failure"}


class _RecordingServer:
    """A tiny FastAPI server recording every request the client made."""

    def __init__(self, *, fail_endpoint: str | None = None) -> None:
        self.app = FastAPI()
        self.calls: list[tuple[str, dict]] = []
        self._fail_endpoint = fail_endpoint
        self._setup_routes()

    def _setup_routes(self) -> None:
        for path in (
            "/areal_init_weights_update_group",
            "/areal_set_update_weight_meta",
            "/areal_update_weights_xccl",
            "/areal_update_weights",
            "/areal_pause_generation",
            "/areal_continue_generation",
        ):
            self.app.post(path)(self._make_handler(path))

    def _make_handler(self, path: str):
        async def passthrough(request: Request):
            try:
                body = await request.json()
            except Exception:
                body = {}
            self.calls.append((path, body))
            if path == self._fail_endpoint:
                return JSONResponse(_RESPONSE_FAIL, status_code=400)
            return _RESPONSE_OK

        return passthrough


@contextmanager
def _running_server(server: _RecordingServer):
    config = uvicorn.Config(server.app, host="127.0.0.1", port=0, log_level="warning")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()

    try:
        # Wait for uvicorn to bind a real port.
        for _ in range(200):  # ~10s max
            if instance.started and instance.servers:
                break
            threading.Event().wait(0.05)
        if not instance.servers:
            raise RuntimeError("uvicorn did not start in time")

        socket = instance.servers[0].sockets[0]
        host, port = socket.getsockname()[:2]
        yield f"http://{host}:{port}"
    finally:
        instance.should_exit = True
        thread.join(timeout=5)


def test_import_surface():
    """All three public symbols are exported from areal.weight_sync."""
    from areal.weight_sync import (
        WeightSyncClient,
        WeightSyncError,
        iter_named_tensors_for_broadcast,
    )

    assert WeightSyncClient.__module__ == "areal.weight_sync.vllm_ext.client"
    assert issubclass(WeightSyncError, RuntimeError)
    assert callable(iter_named_tensors_for_broadcast)


def test_pause_continue_disk_endpoints():
    """pause/continue/from-disk hit the right URLs and accept JSON envelope."""
    from areal.weight_sync import WeightSyncClient

    server = _RecordingServer()
    with _running_server(server) as url:
        client = WeightSyncClient(url, group_name="t")
        try:
            client.pause_generation()
            client.continue_generation()
            client.push_weights_from_disk("/tmp/some/model")
        finally:
            client.close()

    seen_paths = [path for path, _ in server.calls]
    assert seen_paths == [
        "/areal_pause_generation",
        "/areal_continue_generation",
        "/areal_update_weights",
    ]
    last_payload = server.calls[-1][1]
    assert last_payload == {"model_path": "/tmp/some/model"}


def test_init_communicator_fires_http_then_calls_init_pg():
    """init_communicator POSTs the rendezvous payload AND calls
    init_custom_process_group with the matching args."""
    from areal.weight_sync import WeightSyncClient

    captured: dict = {}

    def _fake_init_pg(**kwargs):
        captured.update(kwargs)
        return object()  # dummy "process group"

    server = _RecordingServer()
    with _running_server(server) as url:
        client = WeightSyncClient(url, group_name="grp1")
        with patch(
            "areal.weight_sync.vllm_ext.client.init_custom_process_group",
            side_effect=_fake_init_pg,
        ):
            client.init_communicator(
                master_addr="10.0.0.1",
                master_port=12345,
                vllm_world_size=4,
                backend="hccl",
            )
        client.close()

    # HTTP side
    init_calls = [c for c in server.calls if c[0] == "/areal_init_weights_update_group"]
    assert len(init_calls) == 1
    payload = init_calls[0][1]
    assert payload == {
        "master_address": "10.0.0.1",
        "master_port": "12345",
        "rank_offset": 1,
        "world_size": 5,
        "backend": "hccl",
        "group_name": "grp1",
    }

    # init_custom_process_group side
    assert captured["world_size"] == 5
    assert captured["rank"] == 0
    assert captured["backend"] == "hccl"
    assert captured["group_name"] == "grp1"
    assert captured["init_method"] == "tcp://10.0.0.1:12345"


def test_push_weights_xccl_buckets_by_size_and_drives_broadcast():
    """push_weights_xccl chunks by chunk_mb, sends meta+update per bucket,
    and calls dist.broadcast for each tensor in the bucket."""
    import torch

    from areal.weight_sync import WeightSyncClient

    # Three params: 4 MiB + 4 MiB + 1 MiB.
    # With chunk_mb=5 we expect 2 buckets: [a, b], [c]  (a alone is 4 MiB,
    # adding b = 8 MiB > 5 MiB so b starts a new bucket; then c joins b.)
    # Actually with our greedy "if full, flush old" rule:
    #   - bucket=[a] (4 MiB)
    #   - try add b (4 MiB), 4+4=8 > 5 → flush [a], bucket=[b]
    #   - try add c (1 MiB), 4+1=5 ≤ 5 → bucket=[b, c]
    # → 2 broadcasts of meta/update, 3 dist.broadcast calls total.
    a = torch.zeros(1024 * 1024, dtype=torch.float32)  # 4 MiB
    b = torch.zeros(1024 * 1024, dtype=torch.float32)  # 4 MiB
    c = torch.zeros(256 * 1024, dtype=torch.float32)  # 1 MiB
    named = [("a", a), ("b", b), ("c", c)]

    broadcast_calls: list = []

    def _fake_broadcast(tensor, src, group, async_op):
        broadcast_calls.append((tensor.shape, src, async_op))

    server = _RecordingServer()
    with _running_server(server) as url:
        client = WeightSyncClient(url, group_name="b")
        # Skip init_communicator -- inject a fake group directly.
        client._group = object()  # type: ignore[assignment]

        with patch(
            "areal.weight_sync.vllm_ext.client.dist.broadcast",
            side_effect=_fake_broadcast,
        ):
            client.push_weights_xccl(named, chunk_mb=5)
        client.close()

    paths = [p for p, _ in server.calls]
    # pause + 2 buckets * (meta + update) + continue
    assert paths == [
        "/areal_pause_generation",
        "/areal_set_update_weight_meta",
        "/areal_update_weights_xccl",
        "/areal_set_update_weight_meta",
        "/areal_update_weights_xccl",
        "/areal_continue_generation",
    ]
    assert len(broadcast_calls) == 3

    # First meta payload describes bucket [a]; second describes [b, c].
    meta_payloads = [
        body for path, body in server.calls if path == "/areal_set_update_weight_meta"
    ]
    assert meta_payloads[0]["names"] == ["a"]
    assert meta_payloads[1]["names"] == ["b", "c"]
    assert all(p["group_name"] == "b" for p in meta_payloads)
    assert meta_payloads[1]["dtypes"] == ["float32", "float32"]
    assert meta_payloads[1]["shapes"] == [[1024 * 1024], [256 * 1024]]


def test_push_weights_xccl_resumes_generation_on_error():
    """If broadcast or HTTP fails, continue_generation must still fire."""
    import torch

    from areal.weight_sync import WeightSyncClient

    server = _RecordingServer()
    with _running_server(server) as url:
        client = WeightSyncClient(url, group_name="r")
        client._group = object()  # type: ignore[assignment]

        def _boom(*a, **kw):
            raise RuntimeError("synthetic broadcast failure")

        with patch(
            "areal.weight_sync.vllm_ext.client.dist.broadcast", side_effect=_boom
        ):
            with pytest.raises(RuntimeError, match="synthetic"):
                client.push_weights_xccl([("a", torch.zeros(8))], chunk_mb=128)
        client.close()

    assert ("/areal_pause_generation", {}) in server.calls
    # continue must be called even though broadcast raised
    assert any(p == "/areal_continue_generation" for p, _ in server.calls)


def test_server_failure_surfaces_as_weight_sync_error():
    """Non-success HTTP responses get wrapped in WeightSyncError."""
    from areal.weight_sync import WeightSyncClient, WeightSyncError

    server = _RecordingServer(fail_endpoint="/areal_pause_generation")
    with _running_server(server) as url:
        client = WeightSyncClient(url, group_name="e")
        try:
            with pytest.raises(WeightSyncError):
                client.pause_generation()
        finally:
            client.close()


def test_dtype_str_round_trip():
    """Common torch dtypes serialise into the strings the server expects."""
    import torch

    from areal.weight_sync.vllm_ext.client import _dtype_to_str

    assert _dtype_to_str(torch.float32) == "float32"
    assert _dtype_to_str(torch.bfloat16) == "bfloat16"
    assert _dtype_to_str(torch.int64) == "int64"


def test_double_init_raises():
    """Calling init_communicator twice on the same client is a programmer error."""
    from areal.weight_sync import WeightSyncClient, WeightSyncError

    server = _RecordingServer()
    with _running_server(server) as url:
        client = WeightSyncClient(url, group_name="d")
        with patch(
            "areal.weight_sync.vllm_ext.client.init_custom_process_group",
            return_value=object(),
        ):
            client.init_communicator(
                master_addr="127.0.0.1",
                master_port=1,
                vllm_world_size=1,
                backend="gloo",
            )
            with pytest.raises(WeightSyncError, match="already initialized"):
                client.init_communicator(
                    master_addr="127.0.0.1",
                    master_port=2,
                    vllm_world_size=1,
                    backend="gloo",
                )
        client.close()


def test_push_before_init_raises():
    from areal.weight_sync import WeightSyncClient, WeightSyncError

    client = WeightSyncClient("http://127.0.0.1:1", group_name="x")
    try:
        with pytest.raises(WeightSyncError, match="init_communicator"):
            client.push_weights_xccl([], chunk_mb=1)
    finally:
        client.close()
