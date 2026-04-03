"""Chat template protocol and simple implementations.

``ChatTemplate`` defines how message lists are formatted into prompt strings.
Different models use different chat formats (ChatML, Llama-style, etc.),
so this abstraction keeps the AgentActor model-agnostic.

This module has ZERO external dependencies (no transformers, no torch).
Framework-specific implementations live in ``forge/utils/chat_template.py``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatTemplate(Protocol):
    """Structural contract for formatting chat messages into prompt strings.

    Any object with matching ``apply`` and ``apply_with_generation_prompt``
    methods satisfies this protocol -- no inheritance required.
    """

    def apply(self, messages: list[dict[str, str]]) -> str:
        """Format a list of messages into a complete prompt string.

        Args:
            messages: List of dicts with ``role`` and ``content`` keys.
                      Roles are typically ``"system"``, ``"user"``, ``"assistant"``.

        Returns:
            Formatted prompt string (WITHOUT trailing generation prompt).
        """
        ...

    def apply_with_generation_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format messages and append the generation trigger for the assistant.

        This is the primary method used during inference -- the returned
        string should end with whatever prefix the model expects before
        generating its response (e.g. ``<|assistant|>\\n``).

        Args:
            messages: List of dicts with ``role`` and ``content`` keys.

        Returns:
            Formatted prompt string with trailing generation prompt.
        """
        ...


class SimpleChatTemplate:
    """Configurable chat template using format strings.

    Supports a role-prefix / role-suffix pattern that covers most common
    formats (ChatML, Llama-style markers, plain ``<|role|>`` tags, etc.).

    Args:
        role_format: Format string with ``{role}`` and ``{content}`` placeholders.
                     Default produces ``<|im_start|>role\\ncontent<|im_end|>`` (ChatML).
        message_sep: Separator inserted between formatted messages.
        generation_prompt: String appended when ``apply_with_generation_prompt``
                          is called.  Default is the ChatML assistant prefix.
        system_prompt: Optional system message prepended to every conversation.
    """

    def __init__(
        self,
        role_format: str = "<|im_start|>{role}\n{content}<|im_end|>",
        message_sep: str = "\n",
        generation_prompt: str = "<|im_start|>assistant\n",
        system_prompt: str | None = None,
    ):
        self._role_format = role_format
        self._message_sep = message_sep
        self._generation_prompt = generation_prompt
        self._system_prompt = system_prompt

    def _prepend_system(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if self._system_prompt and (
            not messages or messages[0].get("role") != "system"
        ):
            return [{"role": "system", "content": self._system_prompt}, *messages]
        return messages

    def apply(self, messages: list[dict[str, str]]) -> str:
        messages = self._prepend_system(messages)
        parts = [
            self._role_format.format(
                role=m.get("role", "user"),
                content=m.get("content", ""),
            )
            for m in messages
        ]
        return self._message_sep.join(parts)

    def apply_with_generation_prompt(self, messages: list[dict[str, str]]) -> str:
        body = self.apply(messages)
        return body + self._message_sep + self._generation_prompt

    def __repr__(self) -> str:
        return (
            f"SimpleChatTemplate(role_format={self._role_format!r}, "
            f"generation_prompt={self._generation_prompt!r})"
        )


CHATML = SimpleChatTemplate(
    role_format="<|im_start|>{role}\n{content}<|im_end|>",
    generation_prompt="<|im_start|>assistant\n",
)

LLAMA_STYLE = SimpleChatTemplate(
    role_format="<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
    generation_prompt="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
