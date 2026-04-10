"""HarborAgentLogic -- bridge rllm Workflow/Agent to Forge AgentLogic Protocol.

Adapts Harbor's ``rllm.workflows.Workflow`` abstraction to Forge's
``AgentLogic`` protocol so that any rllm agent (math, code, SWE, tool, ...)
can run inside Forge's ``AgentActor`` training loop.

Supports two tool-call extraction modes:

- **Structured** (``parser_name="qwen"`` or ``"r1"``): uses rllm's
  ``ToolParser`` to extract ``<tool_call>...</tool_call>`` or
  DeepSeek R1 special-token tool calls.
- **Regex fallback**: extracts Python code blocks from fenced markdown.

Usage::

    # With rllm tool parser (Qwen/Hermes format)
    logic = HarborAgentLogic(parser_name="qwen", max_turns=5)

    # Without rllm (regex-only fallback)
    logic = HarborAgentLogic(max_turns=3)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

from forge.core.types import AgentAction, ToolCall

logger = logging.getLogger(__name__)

_RLLM_PARSER_LOADED = None


def _try_load_rllm_parser(parser_name: str):
    """Lazy-load an rllm ToolParser by name. Returns instance or None."""
    global _RLLM_PARSER_LOADED
    try:
        if "/root/harbor/harbor-verl-train" not in sys.path:
            sys.path.insert(0, "/root/harbor/harbor-verl-train")
        from rllm.parser import get_tool_parser

        cls = get_tool_parser(parser_name)
        return cls()
    except (ImportError, AssertionError) as e:
        if _RLLM_PARSER_LOADED is None:
            logger.warning("rllm ToolParser not available (%s), using regex fallback", e)
            _RLLM_PARSER_LOADED = False
        return None


class HarborAgentLogic:
    """Bridge ``rllm.workflows.Workflow`` into the Forge ``AgentLogic`` protocol.

    This class satisfies ``forge.core.protocols.AgentLogic`` by implementing
    ``process_response``, ``should_continue``, and ``format_feedback``.

    Args:
        workflow_cls: The rllm ``Workflow`` subclass (metadata only).
        workflow_kwargs: Extra kwargs for workflow instances.
        max_turns: Maximum multi-turn interactions before forced termination.
        reward_function: An rllm ``RewardFunction`` for computing step rewards.
        thought_delimiter: Delimiter for thinking vs action (e.g. ``"</think>"``).
        parser_name: rllm tool parser name (``"qwen"``, ``"r1"``).
            If None or rllm unavailable, falls back to regex code-block extraction.
        done_pattern: Regex pattern that signals the episode is finished.
            Default detects ``\\boxed{...}``.
    """

    def __init__(
        self,
        workflow_cls: type | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        max_turns: int = 5,
        reward_function: Any = None,
        thought_delimiter: str = "</think>",
        parser_name: str | None = None,
        done_pattern: str = r"\\boxed\{",
    ):
        self._workflow_cls = workflow_cls
        self._workflow_kwargs = workflow_kwargs or {}
        self._max_turns = max_turns
        self._reward_function = reward_function
        self._thought_delimiter = thought_delimiter
        self._done_pattern = done_pattern

        self._tool_parser = None
        if parser_name:
            self._tool_parser = _try_load_rllm_parser(parser_name)
            if self._tool_parser:
                logger.info("Using rllm %s tool parser", parser_name)

    def process_response(
        self,
        response: str,
        messages: list[dict[str, str]],
    ) -> AgentAction:
        """Parse model response, extract tool calls, and detect completion.

        Extraction priority:
        1. rllm ToolParser (structured ``<tool_call>`` / R1 special tokens)
        2. Regex fallback (fenced code blocks)
        """
        thought, action_text = self._split_thought(response)

        tool_calls = self._extract_tool_calls(action_text)

        done = self._detect_done(action_text)

        if not done and not tool_calls:
            done = self._detect_done(response)

        return AgentAction(
            response=response,
            tool_calls=tool_calls,
            done=done,
            metadata={
                "thought": thought,
                "action_text": action_text,
            },
        )

    def should_continue(self, turn: int, reward: float) -> bool:
        """Continue if no correct answer found and turns remain."""
        if reward > 0:
            return False
        return turn + 1 < self._max_turns

    def format_feedback(
        self,
        action: AgentAction,
        tool_results: list,
        reward: float,
    ) -> str:
        """Build feedback message for the next turn.

        Uses ``<tool_response>`` format when tool parser is active,
        plain text otherwise.
        """
        parts = []

        for tr in tool_results:
            if not hasattr(tr, "success"):
                continue
            if self._tool_parser:
                output = tr.output if tr.success else f"Error: {tr.error}"
                parts.append(f"<tool_response>\n{output}\n</tool_response>")
            else:
                if tr.success and tr.output:
                    parts.append(f"Tool output: {tr.output}")
                elif not tr.success:
                    parts.append(f"Tool error: {tr.error}")

        if not parts and reward <= 0:
            parts.append("Your answer was incorrect. Please try again.")

        return "\n".join(parts) if parts else "Please continue."

    def compute_reward(self, task: dict, response: str) -> float:
        """Compute reward using the rllm reward function if available."""
        if self._reward_function is None:
            return 0.0

        from rllm.agents.agent import Action

        _, action_text = self._split_thought(response)
        result = self._reward_function(task, Action(action_text))
        if hasattr(result, "reward"):
            return float(result.reward)
        return float(result)

    def _split_thought(self, response: str) -> tuple[str, str]:
        """Split response into (thought, action) using the thought delimiter."""
        if self._thought_delimiter in response:
            idx = response.index(self._thought_delimiter)
            end_pos = idx + len(self._thought_delimiter)
            return response[:end_pos], response[end_pos:].strip()
        return "", response

    def _extract_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract tool calls using rllm parser (preferred) or regex fallback."""
        if self._tool_parser:
            return self._extract_with_parser(text)
        return self._extract_with_regex(text)

    def _extract_with_parser(self, text: str) -> list[ToolCall]:
        """Extract structured tool calls using rllm ToolParser."""
        try:
            rllm_calls = self._tool_parser.parse(text)
        except Exception as e:
            logger.debug("ToolParser.parse failed: %s, falling back to regex", e)
            return self._extract_with_regex(text)

        forge_calls = []
        for tc in rllm_calls:
            args_str = json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
            forge_calls.append(
                ToolCall(
                    type=tc.name,
                    content=args_str,
                    metadata={"rllm_arguments": tc.arguments},
                )
            )
        return forge_calls

    def _extract_with_regex(self, text: str) -> list[ToolCall]:
        """Fallback: extract tool calls from fenced code blocks."""
        tool_calls = []
        code_pattern = r"```(?:python|py|code)\s*\n(.*?)```"
        for code in re.findall(code_pattern, text, re.DOTALL):
            tool_calls.append(ToolCall(type="code_execution", content=code.strip()))
        return tool_calls

    def _detect_done(self, text: str) -> bool:
        """Detect if the response contains a final answer."""
        return bool(re.search(self._done_pattern, text))

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def has_tool_parser(self) -> bool:
        return self._tool_parser is not None

    def get_tool_prompt(self, tools_schema: str) -> str:
        """Get the tool-use system prompt for the current parser."""
        if self._tool_parser:
            return self._tool_parser.get_tool_prompt(tools_schema)
        return (
            "When you need to compute something, write Python code "
            "between ```python and ``` tags. The code will be executed "
            "and the result returned to you."
        )

    def __repr__(self) -> str:
        cls_name = self._workflow_cls.__name__ if self._workflow_cls else "None"
        parser = type(self._tool_parser).__name__ if self._tool_parser else "regex"
        return (
            f"HarborAgentLogic(workflow={cls_name}, max_turns={self._max_turns}, "
            f"parser={parser})"
        )
