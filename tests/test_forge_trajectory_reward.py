"""Tests for Phase A3: Trajectory + process reward.

Covers:

* ``forge.core.trajectory`` dataclasses roundtrip through ``to_dict`` /
  ``from_dict``.
* ``@register_reward(scope=...)`` accepts the new kwarg, defaults to
  ``"final"`` for backward compat.
* ``RewardPipeline`` correctly aggregates final + process rewards,
  writes back to trajectory, handles length mismatches.
* Built-in ``tool_call_valid`` process reward scores trajectories the
  way we expect.
"""

from __future__ import annotations

import pytest

from forge.core.trajectory import (
    Observation,
    ToolCall,
    Trajectory,
    Turn,
    TurnRole,
)

# ----------------------------------------------------------------------
# Trajectory / Turn dataclass smoke
# ----------------------------------------------------------------------


class TestTrajectoryDataclasses:
    def test_turn_roundtrip(self):
        turn = Turn(
            turn_idx=2,
            role=TurnRole.ASSISTANT,
            content="let me think",
            tool_calls=[ToolCall(name="python_sandbox", arguments={"code": "1+1"})],
            observations=[Observation(text="2", success=True, metadata={"k": 1})],
            reward=0.3,
            token_ids=[1, 2, 3],
            loss_mask=[1, 1, 0],
        )
        t2 = Turn.from_dict(turn.to_dict())
        assert t2.turn_idx == 2
        assert t2.role == TurnRole.ASSISTANT
        assert t2.content == "let me think"
        assert t2.tool_calls[0].name == "python_sandbox"
        assert t2.observations[0].text == "2"
        assert t2.reward == pytest.approx(0.3)
        assert t2.token_ids == [1, 2, 3]

    def test_trajectory_roundtrip_and_helpers(self):
        traj = Trajectory(
            trajectory_id="ep-1",
            prompt="add",
            target=42,
            turns=[
                Turn(turn_idx=0, role=TurnRole.USER, content="what is 2+2"),
                Turn(
                    turn_idx=1,
                    role=TurnRole.ASSISTANT,
                    content="4",
                    tool_calls=[],
                ),
            ],
        )
        assert traj.final_response() == "4"
        assert len(traj.assistant_turns()) == 1

        traj.turns[0].reward = 0.0
        traj.turns[1].reward = 0.5
        traj.sync_step_rewards()
        assert traj.step_rewards == [0.0, 0.5]

        t2 = Trajectory.from_dict(traj.to_dict())
        assert t2.final_response() == "4"
        assert t2.target == 42
        assert t2.step_rewards == [0.0, 0.5]

    def test_from_single_turn_factory(self):
        traj = Trajectory.from_single_turn(
            prompt="hi", response="world", reward=1.0, target="world", trajectory_id="x"
        )
        assert traj.trajectory_id == "x"
        assert traj.final_response() == "world"
        assert traj.final_reward == 1.0
        assert len(traj.turns) == 1


# ----------------------------------------------------------------------
# Registry scope extension
# ----------------------------------------------------------------------


class TestRewardScopeRegistry:
    def setup_method(self):
        from forge.reward import reset_for_tests

        reset_for_tests()

    def teardown_method(self):
        from forge.reward import reset_for_tests

        reset_for_tests()

    def test_default_scope_is_final(self):
        from forge.reward import get_reward_scope, register_reward

        @register_reward("f_default")
        def _f(prompt, response, **kw):
            return 1.0

        assert get_reward_scope("f_default") == "final"
        assert getattr(_f, "_forge_reward_scope", None) == "final"

    def test_explicit_process_scope(self):
        from forge.reward import get_reward_scope, register_reward, rewards_by_scope

        @register_reward("p_thing", scope="process")
        def _p(traj):
            return [0.0] * len(traj.turns)

        assert get_reward_scope("p_thing") == "process"
        assert "p_thing" in rewards_by_scope("process")
        assert getattr(_p, "_forge_reward_scope", None) == "process"

    def test_invalid_scope_raises_at_decoration_time(self):
        from forge.reward import register_reward

        with pytest.raises(ValueError, match="invalid scope"):

            @register_reward("bad", scope="middle")
            def _b(prompt, response):
                return 0.0

    def test_duplicate_across_scopes_still_raises(self):
        from forge.reward import register_reward

        @register_reward("same_name")
        def _first(prompt, response):
            return 0.0

        with pytest.raises(ValueError, match="already registered"):

            @register_reward("same_name", scope="process")
            def _second(traj):
                return []


# ----------------------------------------------------------------------
# RewardPipeline aggregation
# ----------------------------------------------------------------------


class TestRewardPipeline:
    def setup_method(self):
        from forge.reward import reset_for_tests

        reset_for_tests()
        # Rebuild built-ins so the pipeline finds tool_call_valid.
        import importlib

        import forge.reward.process_rewards as _pr

        importlib.reload(_pr)

    def teardown_method(self):
        from forge.reward import reset_for_tests

        reset_for_tests()

    def _build_trajectory_with_tool_call(self):
        """Single assistant turn that called a tool, received a
        successful observation."""
        return Trajectory(
            trajectory_id="ep",
            prompt="compute",
            target="4",
            turns=[
                Turn(
                    turn_idx=0,
                    role=TurnRole.ASSISTANT,
                    content="4",
                    tool_calls=[
                        ToolCall(name="python_sandbox", arguments={"code": "2+2"})
                    ],
                    observations=[Observation(text="4", success=True)],
                ),
            ],
        )

    def test_final_only_pipeline(self):
        from forge.reward import register_reward
        from forge.reward.pipeline import RewardPipeline

        @register_reward("exact_match")
        def _exact(prompt, response, **kw):
            return 1.0 if response == kw.get("target") else 0.0

        traj = self._build_trajectory_with_tool_call()
        breakdown = RewardPipeline(["exact_match"]).score(traj)
        assert breakdown.final == pytest.approx(1.0)
        assert breakdown.per_reward == {"exact_match": 1.0}
        assert breakdown.step_rewards == [0.0]
        # Writeback propagated
        assert traj.final_reward == pytest.approx(1.0)
        assert traj.step_rewards == [0.0]
        assert traj.reward_breakdown["exact_match"] == pytest.approx(1.0)

    def test_process_reward_scores_per_turn(self):
        from forge.reward.pipeline import RewardPipeline

        traj = self._build_trajectory_with_tool_call()
        breakdown = RewardPipeline(["tool_call_valid"]).score(traj)
        # One assistant turn with valid tool call and success -> 0.1
        assert breakdown.final == pytest.approx(0.1)
        assert breakdown.step_rewards == [pytest.approx(0.1)]
        assert breakdown.per_reward_steps["tool_call_valid"] == [pytest.approx(0.1)]
        # Turn.reward was written back too
        assert traj.turns[0].reward == pytest.approx(0.1)

    def test_combined_final_plus_process(self):
        from forge.reward import register_reward
        from forge.reward.pipeline import RewardPipeline

        @register_reward("exact_match")
        def _exact(prompt, response, **kw):
            return 1.0 if response == kw.get("target") else 0.0

        traj = self._build_trajectory_with_tool_call()
        breakdown = RewardPipeline(["exact_match", "tool_call_valid"]).score(traj)
        assert breakdown.final == pytest.approx(1.1)
        assert breakdown.step_rewards == [pytest.approx(0.1)]

    def test_weighted_sum(self):
        from forge.reward.pipeline import RewardPipeline

        traj = self._build_trajectory_with_tool_call()
        breakdown = RewardPipeline(
            ["tool_call_valid"],
            weights={"tool_call_valid": 5.0},
        ).score(traj)
        assert breakdown.final == pytest.approx(0.5)
        assert breakdown.step_rewards == [pytest.approx(0.5)]

    def test_process_reward_length_padding(self):
        from forge.reward import register_reward
        from forge.reward.pipeline import RewardPipeline

        @register_reward("short_vec", scope="process")
        def _short(traj):
            return [0.2]  # shorter than n_turns

        traj = Trajectory(
            turns=[
                Turn(turn_idx=0, role=TurnRole.ASSISTANT, content="a"),
                Turn(turn_idx=1, role=TurnRole.ASSISTANT, content="b"),
                Turn(turn_idx=2, role=TurnRole.ASSISTANT, content="c"),
            ],
        )
        breakdown = RewardPipeline(["short_vec"]).score(traj)
        assert breakdown.step_rewards == [pytest.approx(0.2), 0.0, 0.0]

    def test_process_reward_wrong_type_raises(self):
        from forge.reward import register_reward
        from forge.reward.pipeline import RewardPipeline

        @register_reward("bad_out", scope="process")
        def _bad(traj):
            return 0.5  # must be list, not scalar

        traj = Trajectory(turns=[Turn(turn_idx=0, role=TurnRole.ASSISTANT)])
        with pytest.raises(TypeError, match="must return list"):
            RewardPipeline(["bad_out"]).score(traj)

    def test_unknown_reward_fails_at_pipeline_construction(self):
        from forge.reward.pipeline import RewardPipeline

        with pytest.raises(ValueError, match="unknown reward"):
            RewardPipeline(["nope_never_registered"])

    def test_write_back_false_leaves_trajectory_untouched(self):
        from forge.reward.pipeline import RewardPipeline

        traj = self._build_trajectory_with_tool_call()
        original = traj.final_reward
        breakdown = RewardPipeline(["tool_call_valid"]).score(traj, write_back=False)
        assert breakdown.final == pytest.approx(0.1)
        assert traj.final_reward == original
        assert traj.step_rewards == []  # default, never populated
        assert traj.turns[0].reward == 0.0


# ----------------------------------------------------------------------
# Built-in tool_call_valid / turn_efficiency semantics
# ----------------------------------------------------------------------


class TestBuiltinProcessRewards:
    def setup_method(self):
        from forge.reward import reset_for_tests

        reset_for_tests()
        import importlib

        import forge.reward.process_rewards as _pr

        importlib.reload(_pr)

    def teardown_method(self):
        from forge.reward import reset_for_tests

        reset_for_tests()

    def test_tool_call_valid_zero_for_no_tool_call(self):
        from forge.reward.process_rewards import tool_call_valid_reward

        traj = Trajectory(
            turns=[Turn(turn_idx=0, role=TurnRole.ASSISTANT, content="just text")]
        )
        assert tool_call_valid_reward(traj) == [0.0]

    def test_tool_call_valid_zero_for_failed_observation(self):
        from forge.reward.process_rewards import tool_call_valid_reward

        traj = Trajectory(
            turns=[
                Turn(
                    turn_idx=0,
                    role=TurnRole.ASSISTANT,
                    tool_calls=[ToolCall(name="x")],
                    observations=[Observation(text="boom", success=False)],
                )
            ]
        )
        assert tool_call_valid_reward(traj) == [0.0]

    def test_tool_call_valid_skips_non_assistant(self):
        from forge.reward.process_rewards import tool_call_valid_reward

        traj = Trajectory(
            turns=[
                Turn(turn_idx=0, role=TurnRole.USER, content="go"),
                Turn(
                    turn_idx=1,
                    role=TurnRole.ASSISTANT,
                    tool_calls=[ToolCall(name="x")],
                    observations=[Observation(text="ok", success=True)],
                ),
                Turn(turn_idx=2, role=TurnRole.TOOL, content="tool_out"),
            ]
        )
        assert tool_call_valid_reward(traj) == [0.0, 0.1, 0.0]

    def test_turn_efficiency_negative_per_assistant_turn(self):
        from forge.reward.process_rewards import turn_efficiency_reward

        traj = Trajectory(
            turns=[
                Turn(turn_idx=0, role=TurnRole.ASSISTANT),
                Turn(turn_idx=1, role=TurnRole.TOOL),
                Turn(turn_idx=2, role=TurnRole.ASSISTANT),
            ]
        )
        out = turn_efficiency_reward(traj)
        assert out == [pytest.approx(-0.01), 0.0, pytest.approx(-0.01)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
