"""AgentActor -- Monarch actor for multi-turn agent orchestration.

Phase 5: Demonstrates actor-to-actor communication where AgentActor
drives a multi-turn loop calling:
  - GeneratorActor for text generation (Monarch RPC)
  - SandboxActor for code execution (Monarch RPC)
  - RewardActor for reward computation (Monarch RPC)

Also provides MonarchAgentWorkflow, a RolloutWorkflow drop-in that
delegates episode execution to the AgentActor.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from monarch.actor import Actor, endpoint

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown-style fenced blocks."""
    return _CODE_BLOCK_RE.findall(text)


def _parse_token_ids(tokens: list[str]) -> list[int]:
    """Parse 'token:123' format from GeneratorActor response."""
    ids = []
    for t in tokens:
        if t.startswith("token:"):
            ids.append(int(t.split(":", 1)[1]))
        else:
            ids.append(0)
    return ids


class AgentActor(Actor):
    """Multi-turn agent that orchestrates generation, tool use, and reward.

    Lifecycle:
      __init__  -> store actor references
      setup()   -> load tokenizer, store generation config
      run_episode() -> multi-turn loop returning trajectory
      shutdown() -> cleanup

    Actor-to-actor communication:
      AgentActor ──RPC──→ GeneratorActor (text generation)
      AgentActor ──RPC──→ SandboxActor (code execution)
      AgentActor ──RPC──→ RewardActor (reward computation)
    """

    def __init__(self, generator_actor, sandbox_actor, reward_actor):
        self._generator = generator_actor
        self._sandbox = sandbox_actor
        self._reward = reward_actor
        self._tokenizer = None
        self._gconfig: dict = {}
        self._reward_fn_path: str | None = None
        self._max_turns: int = 2
        self._episode_count = 0
        self._total_turns = 0
        self._code_exec_count = 0

    @endpoint
    async def setup(
        self,
        tokenizer_path: str,
        reward_fn_path: str,
        gconfig: dict,
        max_turns: int = 2,
    ) -> dict:
        """Load tokenizer and configure the agent.

        Parameters
        ----------
        tokenizer_path : str
            HuggingFace tokenizer path.
        reward_fn_path : str
            Dotted import path for the reward function (passed to RewardActor).
        gconfig : dict
            Generation hyperparameters (max_new_tokens, temperature, top_p, etc.).
        max_turns : int
            Maximum turns per episode.
        """
        from areal.utils.hf_utils import load_hf_tokenizer

        self._tokenizer = load_hf_tokenizer(tokenizer_path)
        self._gconfig = gconfig
        self._reward_fn_path = reward_fn_path
        self._max_turns = max_turns

        await self._reward.setup.call_one(reward_fn_path)

        logger.info(
            f"[AgentActor] Setup complete: tokenizer={tokenizer_path}, "
            f"max_turns={max_turns}, reward_fn={reward_fn_path}"
        )
        return {"status": "ready", "max_turns": max_turns}

    @endpoint
    async def run_episode(self, data: dict) -> dict:
        """Run a multi-turn agent episode.

        Flow per turn:
          1. Tokenize conversation → send to GeneratorActor
          2. Parse response for code blocks
          3. If code found → execute via SandboxActor, append output
          4. Compute reward via RewardActor
          5. If reward > 0 → stop; else append retry prompt and loop

        Parameters
        ----------
        data : dict
            Dataset item with 'messages' and task-specific fields.

        Returns
        -------
        dict
            Trajectory with keys: input_ids, logprobs, loss_mask,
            versions, rewards, attention_mask (all as lists for
            serialization across Monarch RPC).
        """
        if self._tokenizer is None:
            raise RuntimeError("AgentActor not set up. Call setup() first.")

        messages = data.get("messages", [])
        tokenizer = self._tokenizer

        input_ids = list(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )

        all_token_ids: list[int] = []
        all_logprobs: list[float] = []
        all_loss_mask: list[int] = []
        all_versions: list[int] = []

        reward = 0.0
        turns_used = 0

        stop_token_ids = []
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)

        for turn in range(self._max_turns):
            turns_used = turn + 1

            payload = {
                "prompt": input_ids,
                "max_tokens": self._gconfig.get("max_new_tokens", 1024),
                "temperature": self._gconfig.get("temperature", 1.0),
                "top_p": self._gconfig.get("top_p", 1.0),
                "stop_token_ids": stop_token_ids,
                "logprobs": 1,
            }

            gen_result = await self._generator.handle_request.call_one(
                "/v1/completions", payload
            )

            choice = gen_result["choices"][0]
            gen_text = choice.get("text", "")
            gen_logprobs = choice.get("logprobs", {}).get("token_logprobs", [])
            gen_tokens_raw = choice.get("logprobs", {}).get("tokens", [])
            gen_token_ids = _parse_token_ids(gen_tokens_raw)

            input_len = len(input_ids) - len(all_token_ids)
            all_token_ids.extend(input_ids[-input_len:])
            all_logprobs.extend([0.0] * input_len)
            all_loss_mask.extend([0] * input_len)
            all_versions.extend([-1] * input_len)

            all_token_ids.extend(gen_token_ids)
            all_logprobs.extend(gen_logprobs)
            all_loss_mask.extend([1] * len(gen_token_ids))
            all_versions.extend([0] * len(gen_token_ids))

            code_blocks = _extract_code_blocks(gen_text)
            exec_output = ""
            if code_blocks and self._sandbox is not None:
                self._code_exec_count += 1
                for code in code_blocks:
                    exec_result = await self._sandbox.execute_code.call_one(
                        code, 10.0
                    )
                    if exec_result["success"]:
                        exec_output += exec_result.get("result", "")
                    else:
                        exec_output += f"Error: {exec_result.get('stderr', '')}"

            prompt_text = tokenizer.decode(input_ids)
            task_data = {k: v for k, v in data.items() if k != "messages"}
            reward = await self._reward.compute_reward.call_one(
                prompt_text, gen_text, input_ids, gen_token_ids, task_data
            )

            if reward > 0:
                break

            if turn < self._max_turns - 1:
                retry_content = (
                    "Your answer is either wrong or not parsable. "
                    "Please carefully re-read the question and try again."
                )
                if exec_output:
                    retry_content = (
                        f"Code output: {exec_output[:512]}\n\n"
                        f"The answer is still wrong. Please try again."
                    )

                next_messages = messages + [
                    {"role": "assistant", "content": gen_text},
                    {"role": "user", "content": retry_content},
                ]
                input_ids = list(
                    tokenizer.apply_chat_template(
                        next_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )

        discount = 0.9 ** max(0, turns_used - 1)
        reward = float(reward * discount)

        self._episode_count += 1
        self._total_turns += turns_used

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
        avg_turns = (
            self._total_turns / self._episode_count
            if self._episode_count > 0
            else 0
        )
        return {
            "episode_count": self._episode_count,
            "total_turns": self._total_turns,
            "avg_turns": avg_turns,
            "code_exec_count": self._code_exec_count,
        }

    @endpoint
    async def shutdown(self) -> None:
        avg = (
            self._total_turns / self._episode_count
            if self._episode_count > 0
            else 0
        )
        logger.info(
            f"[AgentActor] Shutting down. "
            f"{self._episode_count} episodes, "
            f"avg {avg:.1f} turns, "
            f"{self._code_exec_count} code executions"
        )
        self._tokenizer = None


class MonarchAgentWorkflow:
    """RolloutWorkflow drop-in that delegates to AgentActor via Monarch RPC.

    The WorkflowExecutor calls ``arun_episode(engine, data)``; this
    implementation ignores the engine (AgentActor has its own reference
    to GeneratorActor) and forwards ``data`` to the AgentActor.

    On first call, lazy-initialises the AgentActor with tokenizer,
    reward function, generation config, and max turns.
    """

    def __init__(
        self,
        agent_actor,
        tokenizer_path: str,
        reward_fn_path: str = "",
        gconfig: dict | None = None,
        max_turns: int = 2,
    ):
        self._agent = agent_actor
        self._tokenizer_path = tokenizer_path
        self._reward_fn_path = reward_fn_path
        self._gconfig = gconfig or {}
        self._max_turns = max_turns
        self._setup_done = False

    async def _ensure_setup(self):
        if not self._setup_done:
            result = await self._agent.setup.call_one(
                self._tokenizer_path,
                self._reward_fn_path,
                self._gconfig,
                self._max_turns,
            )
            logger.info(f"[MonarchAgentWorkflow] AgentActor setup: {result}")
            self._setup_done = True

    async def arun_episode(self, engine, data: dict[str, Any]):
        """Delegate to AgentActor and convert result to tensor dict."""
        import torch

        await self._ensure_setup()
        result = await self._agent.run_episode.call_one(data)

        seq = result["input_ids"]
        logprobs = result["logprobs"]
        loss_mask = result["loss_mask"]
        versions = result["versions"]
        reward = result["rewards"]

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}
