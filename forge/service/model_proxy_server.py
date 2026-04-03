"""ModelProxyServer -- HTTP server exposing ModelProxy as OpenAI-compatible API.

This is the key component of CLI-Native Mode (ROLL-inspired):
external agent scripts call ``/v1/chat/completions`` to generate text,
while the server transparently records every request/response as a
*trajectory* that can be consumed by the RL training loop.

Endpoints:
    POST /v1/chat/completions  -- OpenAI-compatible chat completion
    POST /v1/completions       -- Raw text completion
    GET  /health               -- Health check
    GET  /trajectory           -- Retrieve and flush recorded trajectory
    POST /trajectory/reset     -- Clear trajectory buffer

Usage::

    server = ModelProxyServer(model_proxy, host="0.0.0.0", port=8100)
    await server.start()          # non-blocking, runs in background
    ...                           # external agent calls http://host:8100/v1/chat/completions
    trajectory = server.flush()   # get recorded trajectory for training
    await server.stop()

Architecture::

    External Agent  --HTTP-->  ModelProxyServer
                                   |
                                   |  records trajectory (training metadata)
                                   v
                               ModelProxy  --Monarch RPC-->  Generator
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from forge.core.agent import GenerationResult
from forge.service.model_proxy import ModelProxy

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """One generation step in a trajectory."""

    request_id: str
    messages: list[dict[str, str]]
    prompt: str
    result: GenerationResult
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trajectory:
    """Recorded trajectory from an external agent episode.

    Contains all generation steps with their training metadata
    (token_ids, logprobs, versions) for RL training.
    """

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def all_token_ids(self) -> list[int]:
        ids = []
        for s in self.steps:
            ids.extend(s.result.token_ids)
        return ids

    @property
    def all_logprobs(self) -> list[float]:
        lp = []
        for s in self.steps:
            lp.extend(s.result.logprobs)
        return lp

    @property
    def all_versions(self) -> list[int]:
        vs = []
        for s in self.steps:
            vs.extend([s.result.version] * len(s.result.token_ids))
        return vs

    def to_training_dict(self) -> dict:
        """Convert trajectory to the format expected by AgentActor/Trainer."""
        token_ids = self.all_token_ids
        return {
            "input_ids": token_ids,
            "logprobs": self.all_logprobs,
            "loss_mask": [1] * len(token_ids),
            "versions": self.all_versions,
            "rewards": self.metadata.get("reward", 0.0),
            "seq_len": len(token_ids),
        }


class ModelProxyServer:
    """HTTP server wrapping ModelProxy with trajectory recording.

    Args:
        model_proxy: The ``ModelProxy`` instance to serve.
        host: Bind address.
        port: Bind port (0 for auto-assign).
        enable_trajectory: Whether to record trajectories.
    """

    def __init__(
        self,
        model_proxy: ModelProxy,
        host: str = "0.0.0.0",
        port: int = 8100,
        enable_trajectory: bool = True,
    ):
        self._proxy = model_proxy
        self._host = host
        self._port = port
        self._enable_trajectory = enable_trajectory

        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

        self._current_trajectory: Trajectory | None = None
        self._completed_trajectories: list[Trajectory] = []
        self._lock = asyncio.Lock()

        self._setup_routes()

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def actual_port(self) -> int:
        if self._site and self._site._server:
            sockets = self._site._server.sockets
            if sockets:
                return sockets[0].getsockname()[1]
        return self._port

    async def start(self) -> None:
        """Start the HTTP server in the background."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        actual = self.actual_port
        logger.info(f"ModelProxyServer listening on {self._host}:{actual}")

    async def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        logger.info("ModelProxyServer stopped")

    def new_trajectory(self, episode_id: str | None = None) -> Trajectory:
        """Start recording a new trajectory."""
        traj = Trajectory(episode_id=episode_id or str(uuid.uuid4()))
        self._current_trajectory = traj
        return traj

    def flush(self) -> list[Trajectory]:
        """Return and clear all completed trajectories."""
        if self._current_trajectory and self._current_trajectory.steps:
            self._completed_trajectories.append(self._current_trajectory)
            self._current_trajectory = None
        result = self._completed_trajectories
        self._completed_trajectories = []
        return result

    def _setup_routes(self) -> None:
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
        self._app.router.add_post("/v1/completions", self._handle_completions)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/trajectory", self._handle_get_trajectory)
        self._app.router.add_post("/trajectory/reset", self._handle_reset_trajectory)
        self._app.router.add_post("/trajectory/set_reward", self._handle_set_reward)

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_chat_completions(self, request: web.Request) -> web.Response:
        """OpenAI-compatible /v1/chat/completions endpoint."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        messages = body.get("messages", [])
        if not messages:
            return web.json_response({"error": "messages is required"}, status=400)

        gen_result = await self._proxy.generate(messages=messages)

        if self._enable_trajectory:
            await self._record_step(messages, gen_result)

        response_body = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "forge-generator",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": gen_result.text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(gen_result.token_ids),
                "total_tokens": len(gen_result.token_ids),
            },
        }
        return web.json_response(response_body)

    async def _handle_completions(self, request: web.Request) -> web.Response:
        """Raw /v1/completions endpoint."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        prompt = body.get("prompt", "")
        if not prompt:
            return web.json_response({"error": "prompt is required"}, status=400)

        gen_result = await self._proxy.generate(prompt=prompt)

        if self._enable_trajectory:
            await self._record_step([{"role": "user", "content": prompt}], gen_result)

        response_body = {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "forge-generator",
            "choices": [
                {
                    "text": gen_result.text,
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }
        return web.json_response(response_body)

    async def _handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_get_trajectory(self, _request: web.Request) -> web.Response:
        """Return the current trajectory as JSON."""
        traj = self._current_trajectory
        if traj is None:
            return web.json_response({"steps": [], "episode_id": None})

        steps = []
        for s in traj.steps:
            steps.append(
                {
                    "request_id": s.request_id,
                    "text": s.result.text,
                    "token_count": len(s.result.token_ids),
                    "version": s.result.version,
                    "timestamp": s.timestamp,
                }
            )
        return web.json_response(
            {
                "episode_id": traj.episode_id,
                "steps": steps,
                "step_count": len(traj.steps),
            }
        )

    async def _handle_reset_trajectory(self, _request: web.Request) -> web.Response:
        """Finalize current trajectory and start fresh."""
        async with self._lock:
            if self._current_trajectory and self._current_trajectory.steps:
                self._completed_trajectories.append(self._current_trajectory)
            self._current_trajectory = None
        return web.json_response({"status": "reset"})

    async def _handle_set_reward(self, request: web.Request) -> web.Response:
        """Set the reward for the current trajectory."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        reward = body.get("reward", 0.0)
        async with self._lock:
            if self._current_trajectory:
                self._current_trajectory.metadata["reward"] = reward
        return web.json_response({"status": "ok", "reward": reward})

    # ------------------------------------------------------------------
    # Trajectory recording
    # ------------------------------------------------------------------

    async def _record_step(
        self,
        messages: list[dict[str, str]],
        gen_result: GenerationResult,
    ) -> None:
        async with self._lock:
            if self._current_trajectory is None:
                self._current_trajectory = Trajectory()

            prompt = self._proxy._apply_template(messages)
            step = TrajectoryStep(
                request_id=str(uuid.uuid4()),
                messages=messages,
                prompt=prompt,
                result=gen_result,
            )
            self._current_trajectory.steps.append(step)
