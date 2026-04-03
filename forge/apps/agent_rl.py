"""Agentic RL training app for Forge.

Extends the GRPO pipeline with multi-turn conversation, tool use
(sandbox code execution), and per-turn reward computation.

This app demonstrates Forge's unique agentic capabilities:
- Multi-turn rollout with tool calls
- Subprocess-isolated code execution
- Session-sticky routing for KV cache locality

Usage::

    python -m forge.apps.agent_rl \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml \\
        --reward-fn areal.reward.gsm8k.gsm8k_reward_fn
"""

from __future__ import annotations

import logging

from forge.api.types import Sample
from forge.core.rollout import RolloutContext

logger = logging.getLogger("forge.apps.agent_rl")


async def default_agent_rollout(
    sample: Sample,
    ctx: RolloutContext,
    *,
    max_turns: int = 5,
) -> Sample:
    """Default multi-turn agentic rollout function.

    This implements the standard agentic RL pattern:
    1. Generate a response from the model.
    2. If the response contains code, execute it in the sandbox.
    3. Append the execution result as a tool observation.
    4. Repeat until the model gives a final answer or max turns reached.
    5. Compute reward.

    Users can use this as-is or write their own rollout function
    following the same pattern.
    """
    messages = [{"role": "user", "content": sample.prompt}]
    full_response = ""
    all_response_ids: list[int] = []
    all_logprobs: list[float] = []
    all_loss_mask: list[int] = []

    for turn in range(max_turns):
        prompt_text = _format_messages(messages)
        results = await ctx.engine.generate([prompt_text], ctx.default_params)
        result = results[0]

        full_response += result.text
        all_response_ids.extend(result.token_ids)
        all_logprobs.extend(result.logprobs)
        all_loss_mask.extend([1] * len(result.token_ids))

        code = _extract_code(result.text)
        if code and "sandbox" in ctx.tools:
            exec_result = await ctx.tools.sandbox.execute(code=code, timeout=10)
            observation = _format_observation(exec_result)
            messages.append({"role": "assistant", "content": result.text})
            messages.append({"role": "tool", "content": observation})
            full_response += observation

            if ctx.tokenizer is not None:
                obs_ids = ctx.tokenizer(observation, add_special_tokens=False)[
                    "input_ids"
                ]
                all_response_ids.extend(obs_ids)
                all_logprobs.extend([0.0] * len(obs_ids))
                all_loss_mask.extend([0] * len(obs_ids))
        else:
            break

    reward = 0.0
    if ctx.reward_fn is not None:
        try:
            reward = ctx.reward_fn(
                sample.prompt,
                full_response,
                sample.prompt_ids,
                all_response_ids,
                **sample.task_data,
            )
        except Exception as e:
            logger.error("Reward computation failed: %s", e)

    return sample.with_response(
        full_response,
        float(reward),
        response_ids=all_response_ids,
        logprobs=all_logprobs,
        loss_mask=all_loss_mask,
    )


def _format_messages(messages: list[dict[str, str]]) -> str:
    """Format a list of chat messages into a single prompt string."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            parts.append(content)
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _extract_code(text: str) -> str | None:
    """Extract Python code from model output."""
    import re

    patterns = [
        r"<code>(.*?)</code>",
        r"```python\s*(.*?)\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _format_observation(exec_result: dict) -> str:
    """Format sandbox execution result as an observation string."""
    if exec_result.get("success"):
        output = exec_result.get("result", exec_result.get("stdout", ""))
        return f"\n<observation>\n{output}\n</observation>\n"
    else:
        error = exec_result.get("stderr", "Unknown error")
        return f"\n<observation>\nError: {error}\n</observation>\n"
