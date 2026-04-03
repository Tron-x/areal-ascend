"""AgentActor -- multi-turn agent orchestration for agentic RL.

Drives a multi-turn loop calling:
  - Generator for text generation (via ServiceInterface.route())
  - SandboxActor for code execution (via ServiceInterface.route())
  - RewardActor for reward computation (via ServiceInterface.route())

Also provides MonarchAgentWorkflow, a RolloutWorkflow drop-in that
delegates episode execution to the AgentActor.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

logger = logging.getLogger(__name__)


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown-style fenced blocks."""
    pattern = r"```(?:python|py)\s*\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


class AgentActor(ForgeActor):
    """Multi-turn agent actor for agentic RL.

    Orchestrates generation, tool use (code execution), and reward
    computation in a multi-turn loop. Each episode retries until
    a positive reward is achieved or max_turns is reached.

    Deploy as a service for session-based routing::

        agent = await AgentActor.options(
            num_replicas=4, procs=1
        ).as_service()
        async with agent.session() as sess:
            result = await agent.run_episode.route(data)
    """

    procs = 1
    with_gpus = False

    def __init__(
        self,
        generator=None,
        reward=None,
        sandbox=None,
        max_turns: int = 3,
        turn_discount: float = 0.9,
    ):
        self._generator = generator
        self._reward = reward
        self._sandbox = sandbox
        self._max_turns = max_turns
        self._turn_discount = turn_discount
        self._episode_count = 0

    @endpoint
    async def setup(
        self,
        generator=None,
        reward=None,
        sandbox=None,
        max_turns: int | None = None,
    ):
        """Configure actor references (can be set post-construction)."""
        if generator is not None:
            self._generator = generator
        if reward is not None:
            self._reward = reward
        if sandbox is not None:
            self._sandbox = sandbox
        if max_turns is not None:
            self._max_turns = max_turns

    @endpoint
    async def run_episode(self, data: dict) -> dict:
        """Run a multi-turn agent episode.

        Args:
            data: Dict with keys like ``messages``, ``answer``, etc.

        Returns:
            Dict with ``input_ids``, ``logprobs``, ``loss_mask``,
            ``versions``, ``rewards``, ``seq_len``.
        """
        self._episode_count += 1

        messages = data.get("messages", [])
        if not messages:
            prompt = data.get("prompt", "")
            messages = [{"role": "user", "content": prompt}]

        all_token_ids = []
        all_logprobs = []
        all_loss_mask = []
        all_versions = []
        reward = 0.0
        turns_used = 0

        for turn in range(self._max_turns):
            turns_used += 1

            prompt_text = self._format_messages(messages)
            gen_result = await self._call_generator(prompt_text)

            gen_text = gen_result.get("text", "")
            gen_tokens = gen_result.get("token_ids", [])
            gen_logprobs = gen_result.get("logprobs", [])
            gen_version = gen_result.get("generator_version", -1)

            all_token_ids.extend(gen_tokens)
            all_logprobs.extend(
                gen_logprobs
                if isinstance(gen_logprobs, list)
                else [0.0] * len(gen_tokens)
            )
            all_loss_mask.extend([1] * len(gen_tokens))
            all_versions.extend([gen_version] * len(gen_tokens))

            code_blocks = _extract_code_blocks(gen_text)
            exec_output = ""
            if code_blocks and self._sandbox is not None:
                for code in code_blocks:
                    result = await self._call_sandbox(code)
                    if result.get("success"):
                        exec_output += result.get("result", "")
                    else:
                        exec_output += f"Error: {result.get('stderr', '')}"

            reward = await self._call_reward(prompt_text, gen_text, data)

            if reward > 0:
                break

            feedback = "Your answer was incorrect."
            if exec_output:
                feedback += f" Code output: {exec_output}"
            feedback += " Please try again."

            messages.append({"role": "assistant", "content": gen_text})
            messages.append({"role": "user", "content": feedback})

        discount = self._turn_discount ** max(0, turns_used - 1)
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
    def get_stats(self) -> dict:
        return {"episode_count": self._episode_count}

    async def _call_generator(self, prompt: str) -> dict:
        """Call generator via service route or actor call_one."""
        if self._generator is None:
            raise RuntimeError("Generator not configured")

        if hasattr(self._generator, "generate"):
            gen_ep = self._generator.generate
            if hasattr(gen_ep, "route"):
                results = await gen_ep.route(prompt)
            else:
                results = await gen_ep.call_one(prompt)

            if isinstance(results, list) and results:
                return results[0]
            return results if isinstance(results, dict) else {"text": str(results)}
        raise RuntimeError("Generator has no generate endpoint")

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

    def _format_messages(self, messages: list[dict]) -> str:
        """Format message list into a prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)


class MonarchAgentWorkflow:
    """RolloutWorkflow drop-in that delegates to AgentActor via Monarch RPC.

    Usage::

        workflow = MonarchAgentWorkflow(agent_actor)
        # Used by RolloutActor's prepare_batch
        result = await workflow.arun_episode(engine, data)
    """

    def __init__(self, agent_actor):
        self._agent = agent_actor
        self._setup_done = False

    async def setup(self, generator=None, reward=None, sandbox=None):
        if not self._setup_done:
            ep = self._agent.setup
            if hasattr(ep, "route"):
                await ep.route(generator=generator, reward=reward, sandbox=sandbox)
            else:
                await ep.call_one(generator=generator, reward=reward, sandbox=sandbox)
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
