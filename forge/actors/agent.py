"""AgentActor -- multi-turn orchestrator decoupled from agent logic.

Supports two execution modes:

**Managed Mode** (Phase 1) -- AgentActor drives the episode loop::

    AgentLogic  (what)     -- pure strategy: parse response, extract tools, feedback
    AgentActor  (how)      -- Monarch actor: drives the loop, collects training data
    ModelProxy  (bridge)   -- uniform LLM interface hiding Generator RPC details

**CLI-Native Mode** (Phase 2) -- external process drives the loop::

    External Agent  --HTTP-->  ModelProxyServer  --Monarch-->  Generator
                                   |
                               trajectory recording
                                   |
                               AgentActor collects training data

Also provides MonarchAgentWorkflow, a RolloutWorkflow drop-in that
delegates episode execution to the AgentActor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from monarch.actor import endpoint

from forge.actors.base import ForgeActor
from forge.core.agent import AgentAction, GenerationResult, ToolCall, ToolResult

if TYPE_CHECKING:
    from forge.core.agent import AgentLogic
    from forge.service.model_proxy import ModelProxy

logger = logging.getLogger(__name__)


class AgentActor(ForgeActor):
    """Multi-turn agent orchestrator for agentic RL.

    Decoupled design:

    - ``agent_logic``: an ``AgentLogic`` that defines *what* the agent does
      (parse responses, extract tools, decide when to stop).
    - ``model_proxy``: a ``ModelProxy`` that provides *how* to call the LLM
      (hides Generator RPC, applies ChatTemplate, returns GenerationResult).
    - ``reward`` / ``sandbox``: optional service handles for reward and tools.

    Backward compatibility: if ``agent_logic`` / ``model_proxy`` are not
    provided, a ``SimpleReActAgent`` and bare ``ModelProxy`` are created
    automatically from the legacy ``generator`` / ``max_turns`` / etc.

    Deploy as a service for session-based routing::

        from forge.agents import SimpleReActAgent
        from forge.service.model_proxy import ModelProxy

        proxy = ModelProxy(generator_service, chat_template)
        logic = SimpleReActAgent(max_turns=3)

        agent = await AgentActor.options(
            num_replicas=4, procs=1
        ).as_service(
            agent_logic=logic,
            model_proxy=proxy,
            reward=reward_service,
            sandbox=sandbox_service,
        )
        result = await agent.run_episode.route(data)
    """

    procs = 1
    with_gpus = False

    def __init__(
        self,
        *,
        agent_logic: AgentLogic | None = None,
        model_proxy: ModelProxy | None = None,
        reward=None,
        sandbox=None,
        # Legacy compat -- used only if agent_logic / model_proxy are None
        generator=None,
        max_turns: int = 3,
        turn_discount: float = 0.9,
        chat_template=None,
    ):
        self._logic = agent_logic
        self._proxy = model_proxy
        self._reward = reward
        self._sandbox = sandbox
        self._episode_count = 0

        # Legacy compat: auto-create logic + proxy when not provided
        if self._logic is None:
            from forge.agents.react import SimpleReActAgent

            self._logic = SimpleReActAgent(
                max_turns=max_turns, turn_discount=turn_discount
            )
        if self._proxy is None and generator is not None:
            from forge.service.model_proxy import ModelProxy as MP

            self._proxy = MP(generator, chat_template=chat_template)

    @endpoint
    async def setup(
        self,
        *,
        agent_logic: AgentLogic | None = None,
        model_proxy: ModelProxy | None = None,
        reward=None,
        sandbox=None,
        generator=None,
        chat_template=None,
        max_turns: int | None = None,
    ):
        """Configure actor references (can be set post-construction)."""
        if agent_logic is not None:
            self._logic = agent_logic
        if model_proxy is not None:
            self._proxy = model_proxy
        if reward is not None:
            self._reward = reward
        if sandbox is not None:
            self._sandbox = sandbox

        # Legacy compat
        if generator is not None and self._proxy is None:
            from forge.service.model_proxy import ModelProxy as MP

            self._proxy = MP(generator, chat_template=chat_template)
        elif chat_template is not None and self._proxy is not None:
            self._proxy.chat_template = chat_template

        if max_turns is not None and hasattr(self._logic, "_max_turns"):
            self._logic._max_turns = max_turns

    # ------------------------------------------------------------------
    # Main episode loop
    # ------------------------------------------------------------------

    @endpoint
    async def run_episode(self, data: dict) -> dict:
        """Run a multi-turn agent episode.

        Args:
            data: Dict with keys like ``messages``, ``answer``, etc.

        Returns:
            Dict with ``input_ids``, ``logprobs``, ``loss_mask``,
            ``versions``, ``rewards``, ``seq_len``.
        """
        if self._proxy is None:
            raise RuntimeError(
                "AgentActor: no ModelProxy configured. "
                "Pass model_proxy= or generator= to constructor / setup()."
            )

        self._episode_count += 1

        messages = data.get("messages", [])
        if not messages:
            prompt = data.get("prompt", "")
            messages = [{"role": "user", "content": prompt}]

        all_token_ids: list[int] = []
        all_logprobs: list[float] = []
        all_loss_mask: list[int] = []
        all_versions: list[int] = []
        reward = 0.0
        turns_used = 0

        max_turns = getattr(self._logic, "max_turns", 3)

        for turn in range(max_turns):
            turns_used += 1

            # 1. Generate via ModelProxy
            gen_result: GenerationResult = await self._proxy.generate(messages=messages)

            # 2. Collect training metadata
            all_token_ids.extend(gen_result.token_ids)
            all_logprobs.extend(gen_result.logprobs)
            all_loss_mask.extend([1] * len(gen_result.token_ids))
            all_versions.extend([gen_result.version] * len(gen_result.token_ids))

            # 3. Let agent logic process the response
            action: AgentAction = self._logic.process_response(
                gen_result.text, messages
            )

            # 4. Execute tool calls
            tool_results = await self._execute_tools(action.tool_calls)

            # 5. Compute reward
            prompt_text = self._proxy._apply_template(messages)
            reward = await self._call_reward(prompt_text, gen_result.text, data)

            # 6. Check termination
            if action.done or not self._logic.should_continue(turn, reward):
                break

            # 7. Build feedback and continue
            feedback = self._logic.format_feedback(action, tool_results, reward)
            messages.append({"role": "assistant", "content": gen_result.text})
            messages.append({"role": "user", "content": feedback})

        discount = getattr(self._logic, "compute_discount", lambda t: 1.0)(turns_used)
        reward = float(reward * discount)

        return {
            "input_ids": all_token_ids,
            "logprobs": all_logprobs,
            "loss_mask": all_loss_mask,
            "versions": all_versions,
            "rewards": reward,
            "seq_len": len(all_token_ids),
        }

    @endpoint
    async def get_stats(self) -> dict:
        proxy_stats = self._proxy.get_stats() if self._proxy else {}
        return {
            "episode_count": self._episode_count,
            "agent_logic": repr(self._logic),
            **{f"proxy_{k}": v for k, v in proxy_stats.items()},
        }

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute a list of tool calls, returning results in order."""
        results = []
        for tc in tool_calls:
            if tc.type == "code_execution":
                raw = await self._call_sandbox(tc.content)
                results.append(
                    ToolResult(
                        success=raw.get("success", False),
                        output=raw.get("result", ""),
                        error=raw.get("stderr", ""),
                        tool_call=tc,
                    )
                )
            else:
                results.append(
                    ToolResult(
                        success=False,
                        error=f"Unknown tool type: {tc.type!r}",
                        tool_call=tc,
                    )
                )
        return results

    async def _call_sandbox(self, code: str) -> dict:
        """Call sandbox via service route or actor call_one."""
        if self._sandbox is None:
            return {"success": False, "stderr": "No sandbox configured", "result": ""}

        ep = self._sandbox.execute_code
        if hasattr(ep, "route"):
            return await ep.route(code)
        return await ep.call_one(code)

    async def _call_reward(self, prompt: str, completion: str, data: dict) -> float:
        """Call reward via service route or actor call_one."""
        if self._reward is None:
            return 0.0

        ep = self._reward.compute_reward
        kwargs = {
            "prompt": prompt,
            "completion": completion,
            "task_data": {
                k: v for k, v in data.items() if k not in ("messages", "prompt")
            },
        }
        if hasattr(ep, "route"):
            return await ep.route(**kwargs)
        return await ep.call_one(**kwargs)

    # ------------------------------------------------------------------
    # CLI-Native Mode (Phase 2)
    # ------------------------------------------------------------------

    @endpoint
    async def run_episode_external(
        self,
        data: dict,
        command: list[str],
        timeout: float = 300.0,
    ) -> dict:
        """Run an episode via an external agent process (CLI-Native Mode).

        Starts a ``ModelProxyServer`` (if not already running), launches the
        external agent process, waits for completion, and collects the
        recorded trajectory for RL training.

        Args:
            data: Task data dict (passed to agent via env vars).
            command: Command to run the external agent.
            timeout: Maximum time for the agent process.

        Returns:
            Training dict (same format as ``run_episode``).
        """
        import json

        from forge.agents.external import ExternalAgentRunner
        from forge.service.model_proxy_server import ModelProxyServer

        if self._proxy is None:
            raise RuntimeError("AgentActor: no ModelProxy configured")

        self._episode_count += 1

        if not hasattr(self, "_http_server") or self._http_server is None:
            self._http_server = ModelProxyServer(self._proxy, port=0)
            await self._http_server.start()
            logger.info(
                f"Started ModelProxyServer on port {self._http_server.actual_port}"
            )

        server: ModelProxyServer = self._http_server
        server_url = f"http://127.0.0.1:{server.actual_port}"

        traj = server.new_trajectory()

        runner = ExternalAgentRunner(
            command=command,
            model_proxy_url=server_url,
        )

        task_json = json.dumps(
            {
                k: v
                for k, v in data.items()
                if isinstance(v, (str, int, float, bool, list, dict))
            },
            default=str,
        )
        result = await runner.run(
            env_extra={"FORGE_TASK_DATA": task_json},
            timeout=timeout,
        )

        if result.exit_code != 0 and not result.timed_out:
            logger.warning(
                f"External agent exited with code {result.exit_code}: "
                f"{result.stderr[:500]}"
            )

        prompt_text = data.get("prompt", "")
        full_response = " ".join(s.result.text for s in traj.steps)
        reward = await self._call_reward(prompt_text, full_response, data)
        traj.metadata["reward"] = reward

        return traj.to_training_dict()

    @endpoint
    async def stop_http_server(self) -> dict:
        """Stop the ModelProxyServer if running."""
        if hasattr(self, "_http_server") and self._http_server is not None:
            await self._http_server.stop()
            self._http_server = None
            return {"status": "stopped"}
        return {"status": "not_running"}


# ======================================================================
# MonarchAgentWorkflow -- bridge to AReaL PPOTrainer
# ======================================================================


class MonarchAgentWorkflow:
    """RolloutWorkflow drop-in that delegates to AgentActor via Monarch RPC.

    Usage::

        workflow = MonarchAgentWorkflow(agent_actor)
        result = await workflow.arun_episode(engine, data)
    """

    def __init__(self, agent_actor, chat_template=None, **kwargs):
        self._agent = agent_actor
        self._chat_template = chat_template
        self._extra_kwargs = kwargs
        self._setup_done = False

    async def setup(
        self,
        generator=None,
        reward=None,
        sandbox=None,
        chat_template=None,
    ):
        if not self._setup_done:
            tmpl = chat_template or self._chat_template
            ep = self._agent.setup
            kwargs: dict[str, Any] = dict(
                generator=generator, reward=reward, sandbox=sandbox
            )
            if tmpl is not None:
                kwargs["chat_template"] = tmpl
            if hasattr(ep, "route"):
                await ep.route(**kwargs)
            else:
                await ep.call_one(**kwargs)
            self._setup_done = True

    async def arun_episode(self, engine, data: dict[str, Any]):
        """Delegate to AgentActor and convert result to tensor dict."""
        import torch

        ep = self._agent.run_episode
        if hasattr(ep, "route"):
            res = await ep.route(data)
        else:
            res = await ep.call_one(data)

        tensor_res = {}
        for k, v in res.items():
            if isinstance(v, list):
                if k in ("logprobs", "rewards"):
                    tensor_res[k] = torch.tensor(v, dtype=torch.float32)
                elif k in ("loss_mask", "versions"):
                    tensor_res[k] = torch.tensor(v, dtype=torch.int32)
                elif k == "input_ids":
                    tensor_res[k] = torch.tensor(v, dtype=torch.int32)
                else:
                    tensor_res[k] = v
            elif isinstance(v, (int, float)):
                tensor_res[k] = torch.tensor(v, dtype=torch.float32)
            else:
                tensor_res[k] = v

        return {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in tensor_res.items()
        }
