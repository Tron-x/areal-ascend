"""ReToolAgent -- tool-integrated reasoning agent for agentic RL.

Implements the ``AgentLogic`` protocol with tool-calling support.
Uses ``CompositeParser`` to detect tool calls in LLM output and
``ToolRegistry`` to execute them.

Unlike ``SimpleReActAgent`` which hardcodes code block extraction,
``ReToolAgent`` handles any tool format (``<code>``, ``<tool_call>``,
``<function=>``, `` ```python ``` ``) through pluggable parsers.

Design references:
- Slime ``generate_with_retool.py``: simple for-loop + loss_mask
- AReaL ``TIRWorkflow``: answer detection + tool start/end markers
- ROLL ``ActionParser``: multi-format parsing

This agent is "pure logic" -- no Monarch, no Generator dependency.
It's used by ``AgentActor.run_episode_retool()`` which handles the
actual LLM calls and token management.
"""

from __future__ import annotations

import re

from forge.core.types import AgentAction, ToolCall, ToolResult
from forge.tools.parsers import CompositeParser
from forge.tools.protocol import ActionParser


class ReToolAgent:
    """Tool-integrated reasoning agent.

    Args:
        max_turns: Maximum tool-calling turns per episode.
        parser: Action parser for extracting tool calls. Defaults to
            ``CompositeParser`` which handles all common formats.
        answer_pattern: Regex pattern that signals the final answer.
            When matched, the episode ends immediately.
        tool_output_template: Template for formatting tool execution results.
            Use ``{output}`` as placeholder.
        invalid_action_feedback: Message sent when no tool call or answer
            is found in the LLM output.
    """

    def __init__(
        self,
        max_turns: int = 8,
        parser: ActionParser | None = None,
        answer_pattern: str = r"Answer:\s*\\boxed\{",
        tool_output_template: str = "\n<interpreter>\n{output}\n</interpreter>\n",
        invalid_action_feedback: str = (
            "\nYour previous response did not contain a valid tool call or answer. "
            "Use <code>...</code> for code execution or "
            "'Answer: \\boxed{{answer}}' for the final answer.\n"
        ),
    ):
        self._max_turns = max_turns
        self._parser = parser or CompositeParser()
        self._answer_pattern = re.compile(answer_pattern)
        self._tool_output_template = tool_output_template
        self._invalid_action_feedback = invalid_action_feedback

    @property
    def max_turns(self) -> int:
        return self._max_turns

    def process_response(
        self,
        response: str,
        messages: list[dict[str, str]],
    ) -> AgentAction:
        """Parse response for tool calls or final answer."""
        if self._answer_pattern.search(response):
            return AgentAction(response=response, done=True)

        tool_calls = self._parser.parse(response)
        forge_calls = [
            ToolCall(type=tc.name, content=tc.arguments.get("code", str(tc.arguments)))
            for tc in tool_calls
        ]

        return AgentAction(
            response=response,
            tool_calls=forge_calls,
            done=False,
        )

    def should_continue(self, turn: int, reward: float) -> bool:
        """Continue if turns remain (reward not used for ReTool)."""
        return turn + 1 < self._max_turns

    def format_feedback(
        self,
        action: AgentAction,
        tool_results: list[ToolResult],
        reward: float,
    ) -> str:
        """Format tool execution results as feedback.

        For ReTool, the feedback is the tool output wrapped in
        interpreter tags (loss_mask=0 will be set by AgentActor).
        """
        if not tool_results:
            return self._invalid_action_feedback

        parts = []
        for tr in tool_results:
            if tr.success:
                output = tr.output or "(no output)"
            else:
                output = f"Error: {tr.error}"
            parts.append(self._tool_output_template.format(output=output))

        return "".join(parts)

    def format_tool_observation(self, tool_results: list[ToolResult]) -> str:
        """Format tool results into observation text for the next turn.

        Same as ``format_feedback`` but with a clearer name for the
        ReTool loop in AgentActor.
        """
        return self.format_feedback(AgentAction(), tool_results, 0.0)

    def compute_discount(self, turns_used: int) -> float:
        """No discount for ReTool (reward is binary correctness)."""
        return 1.0

    def __repr__(self) -> str:
        return f"ReToolAgent(max_turns={self._max_turns})"
