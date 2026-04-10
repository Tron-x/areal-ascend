"""End-to-end validation for the Harbor-Forge adapter.

Tests the full adaptation chain without GPU:

1. HarborAgentLogic (process_response / should_continue / format_feedback)
2. Type conversion (rllm Trajectory → Forge Episode)
3. Reward bridging (rllm RewardMathFn → Forge RewardFn)
4. Data loading (GSM8K → task dicts)

Run::

    cd /root/AReaL
    PYTHONPATH=/root/harbor/harbor-verl-train:$PYTHONPATH python -m pytest forge/examples/harbor/test_adapter.py -v
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "/root/harbor/harbor-verl-train")


# ======================================================================
# 1. HarborAgentLogic tests
# ======================================================================


class TestHarborAgentLogic:
    """Test the AgentLogic adapter satisfies Forge's protocol."""

    def test_process_response_with_boxed_answer(self):
        from forge.agents.harbor import HarborAgentLogic

        logic = HarborAgentLogic(max_turns=3)
        action = logic.process_response(
            "<think>2+2=4</think>The answer is \\boxed{4}.",
            [{"role": "user", "content": "What is 2+2?"}],
        )
        assert action.done is True
        assert action.response.startswith("<think>")
        assert action.metadata["thought"] == "<think>2+2=4</think>"
        assert action.metadata["action_text"] == "The answer is \\boxed{4}."

    def test_process_response_no_answer(self):
        from forge.agents.harbor import HarborAgentLogic

        logic = HarborAgentLogic(max_turns=3)
        action = logic.process_response(
            "I need to think more about this problem.",
            [],
        )
        assert action.done is False
        assert action.tool_calls == []

    def test_process_response_with_code_block(self):
        from forge.agents.harbor import HarborAgentLogic

        logic = HarborAgentLogic(max_turns=3)
        action = logic.process_response(
            "Let me compute:\n```python\nprint(2+2)\n```\n",
            [],
        )
        assert action.done is False
        assert len(action.tool_calls) == 1
        assert action.tool_calls[0].type == "code_execution"
        assert "print(2+2)" in action.tool_calls[0].content

    def test_should_continue_positive_reward(self):
        from forge.agents.harbor import HarborAgentLogic

        logic = HarborAgentLogic(max_turns=5)
        assert logic.should_continue(0, 1.0) is False

    def test_should_continue_zero_reward_turns_left(self):
        from forge.agents.harbor import HarborAgentLogic

        logic = HarborAgentLogic(max_turns=5)
        assert logic.should_continue(0, 0.0) is True
        assert logic.should_continue(3, 0.0) is True
        assert logic.should_continue(4, 0.0) is False

    def test_format_feedback(self):
        from forge.agents.harbor import HarborAgentLogic
        from forge.core.types import AgentAction, ToolResult

        logic = HarborAgentLogic()
        fb = logic.format_feedback(
            AgentAction(response="wrong"),
            [ToolResult(success=True, output="42")],
            0.0,
        )
        assert "incorrect" in fb.lower()
        assert "42" in fb

    def test_compute_reward_with_rllm(self):
        """Test reward computation through the rllm bridge."""
        from forge.agents.harbor import HarborAgentLogic

        try:
            from rllm.rewards.reward_fn import math_reward_fn

            logic = HarborAgentLogic(reward_function=math_reward_fn)
            task = {"ground_truth": "4", "data_source": "test"}
            r = logic.compute_reward(task, "<think>calc</think>\\boxed{4}")
            assert r == 1.0

            r = logic.compute_reward(task, "<think>calc</think>\\boxed{5}")
            assert r == 0.0
        except ImportError:
            pytest.skip("rllm not available")


# ======================================================================
# 2. Type conversion tests
# ======================================================================


class TestAdapter:
    """Test rllm <-> Forge type conversions."""

    def _make_rllm_trajectory(self):
        from rllm.agents.agent import Step, Trajectory
        from rllm.engine.rollout import ModelOutput

        mo = ModelOutput(
            text="The answer is 4",
            content="The answer is \\boxed{4}",
            reasoning="2+2=4",
            tool_calls=[],
            prompt_ids=[1, 2, 3],
            completion_ids=[4, 5, 6, 7],
            prompt_length=3,
            completion_length=4,
            finish_reason="stop",
            rollout_log_probs=[-0.1, -0.2, -0.3, -0.4],
        )

        step = Step(
            chat_completions=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is \\boxed{4}"},
            ],
            thought="2+2=4",
            model_response="The answer is \\boxed{4}",
            model_output=mo,
            reward=1.0,
            done=True,
        )

        traj = Trajectory(uid="test-001", name="math", steps=[step], reward=1.0)
        return traj

    def test_trajectory_to_forge_episode(self):
        from forge.examples.harbor.adapter import trajectory_to_forge_episode

        traj = self._make_rllm_trajectory()
        task = {"question": "What is 2+2?", "ground_truth": "4"}

        episode = trajectory_to_forge_episode(traj, task=task)

        assert episode.episode_id == "test-001"
        assert episode.reward == 1.0
        assert episode.prompt == "What is 2+2?"
        assert episode.target == "4"
        assert len(episode.token_ids) == 7  # 3 prompt + 4 completion
        assert len(episode.generator_logprobs) == 7
        assert episode.loss_mask[:3] == [0, 0, 0]
        assert episode.loss_mask[3:] == [1, 1, 1, 1]
        assert episode.metadata["step_rewards"] == [1.0]

    def test_rllm_episode_to_forge_episodes(self):
        from rllm.agents.agent import Episode as RllmEpisode

        from forge.examples.harbor.adapter import rllm_episode_to_forge_episodes

        traj = self._make_rllm_trajectory()
        rllm_ep = RllmEpisode(
            id="ep-001",
            task={"question": "test"},
            trajectories=[traj, traj],
        )

        forge_eps = rllm_episode_to_forge_episodes(rllm_ep)
        assert len(forge_eps) == 2
        assert forge_eps[0].episode_id == "ep-001:traj_0"
        assert forge_eps[1].episode_id == "ep-001:traj_1"

    def test_step_to_agent_action(self):
        from rllm.agents.agent import Action, Step

        from forge.examples.harbor.adapter import step_to_agent_action

        step = Step(
            model_response="The answer is 4",
            action=Action("4"),
            done=True,
            reward=1.0,
        )
        action = step_to_agent_action(step)
        assert action.done is True
        assert action.response == "The answer is 4"
        assert len(action.tool_calls) == 1
        assert action.tool_calls[0].content == "4"

    def test_model_output_to_generation_result(self):
        from rllm.engine.rollout import ModelOutput

        from forge.examples.harbor.adapter import model_output_to_generation_result

        mo = ModelOutput(
            text="Answer",
            content="\\boxed{4}",
            reasoning="think",
            tool_calls=[],
            prompt_ids=[1, 2],
            completion_ids=[3, 4, 5],
            prompt_length=2,
            completion_length=3,
            finish_reason="stop",
            rollout_log_probs=[-0.1, -0.2, -0.3],
        )
        gen = model_output_to_generation_result(mo)
        assert gen.text == "Answer"
        assert gen.token_ids == [3, 4, 5]
        assert gen.logprobs == [-0.1, -0.2, -0.3]


# ======================================================================
# 3. Reward bridging tests
# ======================================================================


class TestReward:
    """Test Harbor reward functions bridged to Forge."""

    def test_fallback_math_reward_correct(self):
        from forge.examples.harbor.reward import _fallback_math_reward

        assert _fallback_math_reward("The answer is \\boxed{42}", "42") == 1.0

    def test_fallback_math_reward_incorrect(self):
        from forge.examples.harbor.reward import _fallback_math_reward

        assert _fallback_math_reward("The answer is \\boxed{43}", "42") == 0.0

    def test_fallback_math_reward_no_boxed(self):
        from forge.examples.harbor.reward import _fallback_math_reward

        assert _fallback_math_reward("The answer is 42", "42") == 0.0

    def test_harbor_math_reward_correct(self):
        from forge.examples.harbor.reward import harbor_math_reward

        r = harbor_math_reward(
            prompt="What is 6*7?",
            response="<think>6*7=42</think>The answer is \\boxed{42}.",
            target="42",
        )
        assert r == 1.0

    def test_harbor_math_reward_incorrect(self):
        from forge.examples.harbor.reward import harbor_math_reward

        r = harbor_math_reward(
            prompt="What is 6*7?",
            response="<think>6*7=43</think>The answer is \\boxed{43}.",
            target="42",
        )
        assert r == 0.0

    def test_wrap_rllm_reward(self):
        """Test the generic rllm reward wrapper."""
        from forge.examples.harbor.reward import wrap_rllm_reward

        try:
            from rllm.rewards.reward_fn import math_reward_fn

            forge_fn = wrap_rllm_reward(math_reward_fn)

            r = forge_fn(
                prompt="What is 2+2?",
                response="<think>calc</think>\\boxed{4}",
                target="4",
            )
            assert r == 1.0
        except ImportError:
            pytest.skip("rllm not available")


# ======================================================================
# 4. Data loading tests
# ======================================================================


class TestData:
    """Test data loaders."""

    def test_dummy_math_tasks(self):
        from forge.examples.harbor.data import _dummy_math_tasks

        tasks = _dummy_math_tasks(3)
        assert len(tasks) == 3
        for t in tasks:
            assert "question" in t
            assert "ground_truth" in t
            assert "messages" in t
            assert t["messages"][0]["role"] == "system"
            assert t["messages"][1]["role"] == "user"

    def test_load_gsm8k_fallback(self):
        """Test that GSM8K loading works (uses cache or falls back to dummy)."""
        from forge.examples.harbor.data import load_gsm8k

        tasks = load_gsm8k(split="train", max_samples=5)
        assert len(tasks) > 0
        assert "question" in tasks[0]
        assert "ground_truth" in tasks[0]
        assert "messages" in tasks[0]
        assert "data_source" in tasks[0]


# ======================================================================
# 5. Integration test: full adapter chain
# ======================================================================


class TestIntegration:
    """Test the full Harbor → Forge adaptation chain."""

    def test_full_chain_math(self):
        """Simulate a complete math episode through the adapter.

        Flow: task → HarborAgentLogic → process response → reward → Episode
        """
        from forge.agents.harbor import HarborAgentLogic
        from forge.examples.harbor.adapter import trajectory_to_forge_episode
        from forge.examples.harbor.data import _dummy_math_tasks
        from forge.examples.harbor.reward import harbor_math_reward

        logic = HarborAgentLogic(max_turns=3)
        tasks = _dummy_math_tasks(1)
        task = tasks[0]

        response = "<think>7*6=42</think>The answer is \\boxed{42}."
        action = logic.process_response(response, task["messages"])

        assert action.done is True

        reward = harbor_math_reward(
            prompt=task["question"],
            response=response,
            target=task["ground_truth"],
        )
        assert reward == 1.0

        from rllm.agents.agent import Step, Trajectory
        from rllm.engine.rollout import ModelOutput

        mo = ModelOutput(
            text=response,
            content="The answer is \\boxed{42}.",
            reasoning="7*6=42",
            tool_calls=[],
            prompt_ids=list(range(10)),
            completion_ids=list(range(10, 25)),
            prompt_length=10,
            completion_length=15,
            finish_reason="stop",
        )
        step = Step(
            chat_completions=task["messages"] + [{"role": "assistant", "content": response}],
            thought="7*6=42",
            model_response=response,
            model_output=mo,
            reward=reward,
            done=True,
        )
        traj = Trajectory(uid="integration-test", name="math", steps=[step], reward=reward)

        episode = trajectory_to_forge_episode(traj, task=task)

        assert episode.episode_id == "integration-test"
        assert episode.reward == 1.0
        assert episode.target == task["ground_truth"]
        assert len(episode.token_ids) == 25
        assert sum(episode.loss_mask) == 15
        assert episode.metadata["step_rewards"] == [1.0]

        cont = logic.should_continue(0, reward)
        assert cont is False

    def test_multi_turn_flow(self):
        """Test multi-turn interaction pattern."""
        from forge.agents.harbor import HarborAgentLogic
        from forge.core.types import ToolResult

        logic = HarborAgentLogic(max_turns=3)

        action1 = logic.process_response(
            "Let me compute:\n```python\nprint(7*6)\n```\n",
            [{"role": "user", "content": "What is 7*6?"}],
        )
        assert action1.done is False
        assert len(action1.tool_calls) == 1

        assert logic.should_continue(0, 0.0) is True

        feedback = logic.format_feedback(
            action1,
            [ToolResult(success=True, output="42")],
            0.0,
        )
        assert "42" in feedback

        action2 = logic.process_response(
            "The answer is \\boxed{42}.",
            [],
        )
        assert action2.done is True

        assert logic.should_continue(1, 1.0) is False
