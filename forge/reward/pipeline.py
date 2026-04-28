"""RewardPipeline -- compose multiple rewards over a Trajectory.

Handles the full cross-product of (final, process) × (many registered
rewards) in a single place so training code doesn't care how any
individual reward was implemented.  Zero torch / NumPy dependencies so
the pipeline is safe to call from any process (CPU agent runner, trainer
collate callback, offline evaluator).

Usage::

    from forge.reward.pipeline import RewardPipeline

    pipeline = RewardPipeline(["gsm8k", "tool_call_valid"])
    breakdown = pipeline.score(trajectory)

    trajectory.final_reward   = breakdown.final
    trajectory.step_rewards   = breakdown.step_rewards
    trajectory.reward_breakdown = breakdown.per_reward

A reward function's scope (``final`` or ``process``) is read from the
registry; callers don't have to know or care.

Phase-A3 MVP scope:

* Scalar weights per reward (uniform sum by default).
* Process rewards add to ``turn.reward`` of every turn.
* No normalization, no clipping -- that's algorithm-side territory
  (GRPO / PPO already normalize advantages; keep rewards raw).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from forge.core.trajectory import Trajectory
from forge.reward import get_reward, get_reward_scope

logger = logging.getLogger("RewardPipeline")


@dataclass
class RewardBreakdown:
    """Structured output of :meth:`RewardPipeline.score`.

    Attributes:
        final: Weighted sum of every reward's scalar contribution.
            ``final-scope`` rewards contribute their scalar directly;
            ``process-scope`` rewards contribute the sum of their
            per-turn values.
        step_rewards: Per-turn reward vector, same length as
            ``trajectory.turns``.  Pure sum of ``process-scope``
            rewards for each turn (final-scope rewards don't
            contribute here).
        per_reward: Per-reward-name scalar contribution (pre-sum)
            for logging / diagnostics.  Final-scope entries hold
            the raw scalar; process-scope entries hold the SUM
            over turns.
        per_reward_steps: Per-reward-name per-turn vectors for
            process-scope rewards only.  Final-scope rewards are
            absent here.
    """

    final: float = 0.0
    step_rewards: list[float] = field(default_factory=list)
    per_reward: dict[str, float] = field(default_factory=dict)
    per_reward_steps: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final": self.final,
            "step_rewards": list(self.step_rewards),
            "per_reward": dict(self.per_reward),
            "per_reward_steps": {k: list(v) for k, v in self.per_reward_steps.items()},
        }


class RewardPipeline:
    """Aggregates multiple registered rewards over a :class:`Trajectory`.

    Args:
        names: Short names (as registered via ``@register_reward``)
            of rewards to include in this pipeline, in the order
            they should be applied.
        weights: Optional per-name scalar weights.  Missing entries
            default to ``1.0``.
        extra_kwargs: Extra keyword arguments forwarded to every
            final-scope reward (e.g. dataset-specific helpers).
    """

    def __init__(
        self,
        names: Iterable[str],
        *,
        weights: Mapping[str, float] | None = None,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.names: list[str] = list(names)
        self.weights: dict[str, float] = dict(weights or {})
        self.extra_kwargs: dict[str, Any] = dict(extra_kwargs or {})

        # Resolve all rewards up front so a typo blows up at
        # pipeline-construction time, not mid-rollout.  We also
        # cache (fn, scope) pairs to avoid a dict lookup per turn.
        self._resolved: list[tuple[str, Callable[..., Any], str, float]] = []
        for n in self.names:
            fn = get_reward(n)
            scope = get_reward_scope(n)
            w = float(self.weights.get(n, 1.0))
            self._resolved.append((n, fn, scope, w))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        trajectory: Trajectory,
        *,
        write_back: bool = True,
    ) -> RewardBreakdown:
        """Run every reward against ``trajectory`` and aggregate.

        Args:
            trajectory: The trajectory to score.  Must have at least
                one turn for process-scope rewards to produce any
                values; empty trajectories still work (process-scope
                contributions are simply zero-length).
            write_back: When ``True`` (default), mutate
                ``trajectory.final_reward``, ``trajectory.step_rewards``,
                ``trajectory.reward_breakdown``, and individual
                ``turn.reward`` fields in place.  Set to ``False`` for
                a pure function that only returns the breakdown.

        Returns:
            :class:`RewardBreakdown` carrying the aggregated scalar
            plus per-reward diagnostics.
        """
        n_turns = len(trajectory.turns)
        step_totals: list[float] = [0.0] * n_turns
        per_reward: dict[str, float] = {}
        per_reward_steps: dict[str, list[float]] = {}
        final_total: float = 0.0

        for name, fn, scope, weight in self._resolved:
            if scope == "final":
                scalar = self._run_final(name, fn, trajectory)
                per_reward[name] = scalar
                final_total += weight * scalar
            elif scope == "process":
                steps = self._run_process(name, fn, trajectory, n_turns)
                # Accumulate into the aggregate step vector.
                for i, v in enumerate(steps):
                    step_totals[i] += weight * v
                s = sum(steps)
                per_reward[name] = s
                per_reward_steps[name] = steps
                # Process rewards also roll into the final scalar so
                # trajectory-level advantages reflect them when the
                # trainer only consumes ``final_reward``.
                final_total += weight * s
            else:  # pragma: no cover -- guarded in register_reward
                logger.warning(
                    "RewardPipeline: skipping unknown scope %r for %r", scope, name
                )

        breakdown = RewardBreakdown(
            final=final_total,
            step_rewards=step_totals,
            per_reward=per_reward,
            per_reward_steps=per_reward_steps,
        )

        if write_back:
            trajectory.final_reward = final_total
            trajectory.step_rewards = list(step_totals)
            # Merge (don't overwrite) so caller-provided breakdowns
            # survive -- e.g. when a workflow tagged an auxiliary
            # metric pre-pipeline.
            trajectory.reward_breakdown.update(per_reward)
            # Distribute per-turn totals back onto each Turn so
            # downstream code that iterates turns sees the scalar.
            for i, v in enumerate(step_totals):
                trajectory.turns[i].reward = v

        return breakdown

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_final(
        self, name: str, fn: Callable[..., Any], trajectory: Trajectory
    ) -> float:
        """Invoke a final-scope reward with the legacy
        ``(prompt, response, **kw)`` signature.

        Extracts ``response`` from the last assistant turn so the
        existing (single-turn) reward code path keeps working even
        when handed a multi-turn trajectory.
        """
        kwargs: dict[str, Any] = dict(self.extra_kwargs)
        if trajectory.target is not None:
            kwargs.setdefault("target", trajectory.target)
            kwargs.setdefault("ground_truth", trajectory.target)
        try:
            result = fn(
                prompt=trajectory.prompt,
                response=trajectory.final_response(),
                **kwargs,
            )
        except Exception as e:  # pragma: no cover -- explicit bubble
            logger.error("final reward %r raised %s", name, e)
            raise
        return float(result)

    def _run_process(
        self,
        name: str,
        fn: Callable[..., Any],
        trajectory: Trajectory,
        n_turns: int,
    ) -> list[float]:
        """Invoke a process-scope reward; coerce output to
        ``list[float]`` of length ``n_turns``.

        Short lists are right-padded with zeros, long lists are
        truncated, so a buggy reward function can't desync the
        aggregate step vector.  Either mismatch gets logged.
        """
        try:
            raw = fn(trajectory)
        except Exception as e:  # pragma: no cover -- explicit bubble
            logger.error("process reward %r raised %s", name, e)
            raise
        if not isinstance(raw, (list, tuple)):
            raise TypeError(
                f"process reward {name!r} must return list[float] "
                f"(length == len(trajectory.turns) == {n_turns}); got "
                f"{type(raw).__name__}"
            )
        steps = [float(v) for v in raw]
        if len(steps) < n_turns:
            logger.debug(
                "process reward %r returned %d values for %d turns; padding with zeros",
                name,
                len(steps),
                n_turns,
            )
            steps = steps + [0.0] * (n_turns - len(steps))
        elif len(steps) > n_turns:
            logger.warning(
                "process reward %r returned %d values for %d turns; truncating",
                name,
                len(steps),
                n_turns,
            )
            steps = steps[:n_turns]
        return steps


__all__ = ["RewardPipeline", "RewardBreakdown"]
