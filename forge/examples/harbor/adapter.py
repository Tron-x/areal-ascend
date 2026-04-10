"""Type conversion between rllm (Harbor) and Forge data structures.

Provides bidirectional conversion for the core data types that flow
between the Harbor agent layer and the Forge training pipeline::

    rllm.agents.agent.Trajectory  <-->  forge.core.types.Episode
    rllm.agents.agent.Step        <-->  forge.core.types.AgentAction
    rllm.engine.rollout.ModelOutput <--> forge.core.types.GenerationResult

The conversions are *lossy* in the direction that matters least:
Forge Episodes carry RL-specific fields (logprobs, loss_mask) that
rllm types do not have, so those are populated with defaults when
converting rllm -> Forge.  Going the other way, Harbor-specific
fields (observation, mc_return) are stored in ``metadata``.
"""

from __future__ import annotations

import uuid
from typing import Any

from forge.core.types import (
    AgentAction,
    Episode as ForgeEpisode,
    GenerationResult,
    ToolCall as ForgeToolCall,
)


def trajectory_to_forge_episode(
    trajectory: Any,
    task: dict | None = None,
    episode_id: str = "",
) -> ForgeEpisode:
    """Convert an ``rllm.agents.agent.Trajectory`` to a ``forge.core.types.Episode``.

    Each rllm Step contributes token IDs and logprobs (when available from
    the ``ModelOutput``).  The trajectory-level reward becomes the episode
    reward, and per-step rewards are preserved in ``metadata``.

    Args:
        trajectory: An ``rllm.agents.agent.Trajectory`` instance.
        task: Optional task dict for ground-truth / prompt info.
        episode_id: Override episode ID (auto-generated if empty).

    Returns:
        A Forge ``Episode`` with fields populated from the trajectory.
    """
    eid = episode_id or trajectory.uid or str(uuid.uuid4())

    all_token_ids: list[int] = []
    all_logprobs: list[float] = []
    all_loss_mask: list[int] = []
    prompt_text = ""
    response_text = ""
    step_rewards: list[float] = []

    for step in trajectory.steps:
        step_rewards.append(step.reward)

        mo = step.model_output
        if mo is not None:
            prompt_ids = getattr(mo, "prompt_ids", []) or []
            completion_ids = getattr(mo, "completion_ids", []) or []
            rollout_lp = getattr(mo, "rollout_log_probs", None) or []

            if not all_token_ids and prompt_ids:
                all_token_ids.extend(prompt_ids)
                all_loss_mask.extend([0] * len(prompt_ids))
                all_logprobs.extend([0.0] * len(prompt_ids))

            all_token_ids.extend(completion_ids)
            all_loss_mask.extend([1] * len(completion_ids))

            if rollout_lp and len(rollout_lp) == len(completion_ids):
                all_logprobs.extend(rollout_lp)
            else:
                all_logprobs.extend([0.0] * len(completion_ids))

            content = getattr(mo, "content", "") or getattr(mo, "text", "") or ""
            response_text += content

        if step.chat_completions:
            for msg in step.chat_completions:
                if msg.get("role") == "user" and not prompt_text:
                    prompt_text = msg.get("content", "")

    target = None
    if task:
        target = task.get("ground_truth") or task.get("answer")

    return ForgeEpisode(
        episode_id=eid,
        prompt=prompt_text,
        response=response_text,
        target=target,
        reward=float(trajectory.reward),
        policy_version=-1,
        token_ids=all_token_ids,
        generator_logprobs=all_logprobs,
        loss_mask=all_loss_mask,
        versions=[-1] * len(all_token_ids),
        metadata={
            "step_rewards": step_rewards,
            "trajectory_name": trajectory.name,
            "task": task,
            "rllm_info": trajectory.info,
        },
    )


def rllm_episode_to_forge_episodes(
    rllm_episode: Any,
) -> list[ForgeEpisode]:
    """Convert an ``rllm.agents.agent.Episode`` to a list of Forge ``Episode`` objects.

    An rllm Episode may contain multiple trajectories (e.g. GRPO group
    sampling).  Each trajectory becomes a separate Forge Episode.

    Args:
        rllm_episode: An ``rllm.agents.agent.Episode`` instance.

    Returns:
        List of Forge Episodes, one per trajectory.
    """
    task = rllm_episode.task if hasattr(rllm_episode, "task") else None
    task_dict = task if isinstance(task, dict) else {"task": task}
    base_id = getattr(rllm_episode, "id", "") or str(uuid.uuid4())

    episodes = []
    for i, traj in enumerate(rllm_episode.trajectories):
        eid = f"{base_id}:traj_{i}" if len(rllm_episode.trajectories) > 1 else base_id
        episodes.append(
            trajectory_to_forge_episode(traj, task=task_dict, episode_id=eid)
        )
    return episodes


def step_to_agent_action(step: Any) -> AgentAction:
    """Convert an ``rllm.agents.agent.Step`` to a Forge ``AgentAction``.

    Args:
        step: An ``rllm.agents.agent.Step`` instance.

    Returns:
        A Forge ``AgentAction``.
    """
    tool_calls = []
    action = getattr(step, "action", None)
    if action is not None:
        action_str = action.action if hasattr(action, "action") else str(action)
        if action_str:
            tool_calls.append(
                ForgeToolCall(type="harbor_action", content=action_str)
            )

    return AgentAction(
        response=step.model_response or "",
        tool_calls=tool_calls,
        done=step.done,
        metadata={
            "thought": step.thought or "",
            "observation": step.observation,
            "reward": step.reward,
            "mc_return": step.mc_return,
        },
    )


def model_output_to_generation_result(model_output: Any) -> GenerationResult:
    """Convert an ``rllm.engine.rollout.ModelOutput`` to a Forge ``GenerationResult``.

    Args:
        model_output: An ``rllm.engine.rollout.rollout_engine.ModelOutput`` instance.

    Returns:
        A Forge ``GenerationResult``.
    """
    return GenerationResult(
        text=getattr(model_output, "text", "") or "",
        token_ids=getattr(model_output, "completion_ids", []) or [],
        logprobs=getattr(model_output, "rollout_log_probs", []) or [],
        version=-1,
        raw=model_output.to_dict() if hasattr(model_output, "to_dict") else {},
    )
