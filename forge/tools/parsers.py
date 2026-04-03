"""Action parsers -- extract tool calls from LLM output.

Covers all major tool-calling formats:

- ``CodeBlockParser``: ``<code>...</code>`` or `` ```python ... ``` ``
  (ReTool / veRL / markdown)
- ``ToolCallParser``: ``<tool_call>{"name":...}</tool_call>``
  (Qwen3 / ChatML tool format)
- ``FunctionCallParser``: ``<function=name><parameter=k>v</parameter></function>``
  (Qwen3 Coder)
- ``CompositeParser``: chains multiple parsers, returns first match

Inspired by ROLL's ``Qwen3CoderActionParser`` and Slime's
``postprocess_predictions``.
"""

from __future__ import annotations

import json
import re

from forge.tools.protocol import ToolCall


class CodeBlockParser:
    """Parse ``<code>...</code>`` and `` ```python ... ``` `` blocks.

    Extracts code content as ``ToolCall(name="code_interpreter")``.
    """

    def __init__(
        self,
        tool_name: str = "code_interpreter",
        patterns: list[str] | None = None,
    ):
        self._tool_name = tool_name
        self._patterns = patterns or [
            r"<code>(.*?)</code>",
            r"```(?:python|py)\s*\n(.*?)```",
        ]

    def parse(self, response: str) -> list[ToolCall]:
        calls = []
        for pattern in self._patterns:
            for match in re.finditer(pattern, response, re.DOTALL):
                code = match.group(1).strip()
                if code:
                    calls.append(
                        ToolCall(
                            name=self._tool_name,
                            arguments={"code": code},
                            raw_text=match.group(0),
                        )
                    )
        return calls

    def has_tool_call(self, response: str) -> bool:
        return any(re.search(p, response, re.DOTALL) for p in self._patterns)


class ToolCallParser:
    """Parse ``<tool_call>{"name":..., "arguments":...}</tool_call>`` blocks.

    Handles Qwen3 / ChatML tool calling format.
    """

    _PATTERN = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"

    def parse(self, response: str) -> list[ToolCall]:
        calls = []
        text = response
        if "<tool_call>" in text and "</tool_call>" not in text:
            text = text + "</tool_call>"

        for match in re.finditer(self._PATTERN, text, re.DOTALL):
            try:
                raw_json = match.group(1).replace("\n", "\\n")
                data = json.loads(raw_json)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append(
                    ToolCall(
                        name=name,
                        arguments=arguments,
                        raw_text=match.group(0),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return calls

    def has_tool_call(self, response: str) -> bool:
        return "<tool_call>" in response


class FunctionCallParser:
    """Parse ``<function=name><parameter=k>v</parameter></function>`` blocks.

    Handles Qwen3 Coder format. Adapted from ROLL's
    ``Qwen3CoderActionParser``.
    """

    _FUNC_PATTERN = r"<function\s*=\s*([^>]+)>(.*?)</function>"
    _PARAM_PATTERN = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"

    def parse(self, response: str) -> list[ToolCall]:
        calls = []
        for match in re.finditer(self._FUNC_PATTERN, response, re.DOTALL):
            func_name = match.group(1).strip()
            body = match.group(2)

            params = {}
            for pm in re.finditer(self._PARAM_PATTERN, body, re.DOTALL):
                key = pm.group(1).strip()
                value = _coerce_param_value(pm.group(2).strip())
                params[key] = value

            calls.append(
                ToolCall(
                    name=func_name,
                    arguments=params,
                    raw_text=match.group(0),
                )
            )
        return calls

    def has_tool_call(self, response: str) -> bool:
        return "<function" in response


class CompositeParser:
    """Chains multiple parsers, returns all found tool calls.

    Usage::

        parser = CompositeParser([
            CodeBlockParser(),
            ToolCallParser(),
            FunctionCallParser(),
        ])
        calls = parser.parse(response)
    """

    def __init__(self, parsers: list | None = None):
        self._parsers = parsers or [
            CodeBlockParser(),
            ToolCallParser(),
            FunctionCallParser(),
        ]

    def parse(self, response: str) -> list[ToolCall]:
        all_calls = []
        for parser in self._parsers:
            all_calls.extend(parser.parse(response))
        return all_calls

    def has_tool_call(self, response: str) -> bool:
        return any(p.has_tool_call(response) for p in self._parsers)


def _coerce_param_value(v: str):
    """Coerce a string parameter value to the appropriate Python type."""
    if not v:
        return v
    if v in ("true", "false", "null"):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    if (v.startswith("[") and v.endswith("]")) or (
        v.startswith("{") and v.endswith("}")
    ):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    return v
