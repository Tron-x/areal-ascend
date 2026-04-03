"""SimpleReActAgent -- code-execution ReAct agent extracted from the original AgentActor.

This is the default ``AgentLogic`` implementation that preserves the
behaviour of the pre-decoupling ``AgentActor.run_episode``:

1. Extract Python code blocks from model output
2. Execute them via sandbox
3. Compute reward
4. If reward <= 0 and turns remain, append feedback and retry

The agent logic is *pure* -- it has no knowledge of Monarch, Generator
actors, or distributed training.  All infrastructure calls go through
``AgentActor`` + ``ModelProxy``.
"""

from __future__ import annotations

import re

from forge.core.types import AgentAction, ToolCall, ToolResult


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown-style fenced blocks."""
    pattern = r"```(?:python|py)\s*\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


class SimpleReActAgent:
    """Code-execution ReAct loop -- the default agentic RL strategy.

    Behaviour per turn:
    - Parse the LLM response for ``python`` / ``py`` fenced code blocks.
    - Each block becomes a ``ToolCall(type="code_execution", ...)``.
    - If reward > 0, the episode ends (correct answer found).
    - Otherwise, build a feedback message and continue.

    Args:
        max_turns: Maximum turns before the episode is forcefully ended.
        turn_discount: Multiplicative discount applied per extra turn
                       (reward *= discount ** (turns_used - 1)).
    """

    def __init__(self, max_turns: int = 3, turn_discount: float = 0.9):
        self._max_turns = max_turns
        self._turn_discount = turn_discount

    # -- AgentLogic protocol --------------------------------------------------

    def process_response(
        self,
        response: str,
        messages: list[dict[str, str]],
    ) -> AgentAction:
        """Extract code blocks from the model response."""
        code_blocks = _extract_code_blocks(response)
        tool_calls = [
            ToolCall(type="code_execution", content=code) for code in code_blocks
        ]
        return AgentAction(
            response=response,
            tool_calls=tool_calls,
            done=False,
        )

    def should_continue(self, turn: int, reward: float) -> bool:
        """Continue if reward is non-positive and turns remain."""
        if reward > 0:
            return False
        return turn + 1 < self._max_turns

    def format_feedback(
        self,
        action: AgentAction,
        tool_results: list[ToolResult],
        reward: float,
    ) -> str:
        """Build feedback for the next turn."""
        feedback = "Your answer was incorrect."
        exec_output = ""
        for tr in tool_results:
            if tr.success:
                exec_output += tr.output
            else:
                exec_output += f"Error: {tr.error}"
        if exec_output:
            feedback += f" Code output: {exec_output}"
        feedback += " Please try again."
        return feedback

    # -- helpers ---------------------------------------------------------------

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def turn_discount(self) -> float:
        return self._turn_discount

    def compute_discount(self, turns_used: int) -> float:
        """Return the cumulative discount for the given number of turns."""
        return self._turn_discount ** max(0, turns_used - 1)

    def __repr__(self) -> str:
        return (
            f"SimpleReActAgent(max_turns={self._max_turns}, "
            f"turn_discount={self._turn_discount})"
        )
