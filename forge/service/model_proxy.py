"""ModelProxy -- unified LLM generation interface decoupled from Monarch.

Sits between AgentLogic and the Generator actor, providing:

1. A simple ``generate(prompt | messages)`` API that returns ``GenerationResult``
2. Transparent collection of RL training metadata (token_ids, logprobs, version)
3. Chat template application so agent logic never touches raw prompt formatting

The proxy is a plain Python object (not a Monarch actor) used inside
``AgentActor``.  For Phase 2 (CLI-Native Mode), an HTTP-serving variant
can expose the same interface over ``/v1/chat/completions``.

Design reference: ROLL's ModelProxy Service -- a lightweight shim between
the Agent Framework and the Training Framework's inference backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from forge.core.types import GenerationResult

if TYPE_CHECKING:
    from forge.core.chat_template import ChatTemplate

logger = logging.getLogger(__name__)


class ModelProxy:
    """Thin proxy around a Generator service for agent-friendly generation.

    Args:
        generator: A ``ServiceInterface`` (or ActorMesh) wrapping Generator.
        chat_template: Template used to format message lists into prompts.
        default_sampling_params: Fallback sampling parameters.

    Usage inside AgentActor::

        proxy = ModelProxy(generator_service, chat_template)
        result = await proxy.generate(messages=[{"role": "user", "content": "Hi"}])
        print(result.text)          # agent-visible
        print(result.token_ids)     # training-visible
    """

    def __init__(
        self,
        generator,
        chat_template: ChatTemplate | None = None,
        default_sampling_params: dict[str, Any] | None = None,
    ):
        self._generator = generator
        self._chat_template = chat_template
        self._default_params = default_sampling_params or {}
        self._call_count = 0

    @property
    def chat_template(self) -> ChatTemplate | None:
        return self._chat_template

    @chat_template.setter
    def chat_template(self, value: ChatTemplate) -> None:
        self._chat_template = value

    async def generate(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate text and return a ``GenerationResult``.

        Exactly one of ``prompt`` or ``messages`` must be provided.
        If ``messages`` is given, the chat template is applied automatically.

        Args:
            prompt: Raw prompt string (sent directly to Generator).
            messages: Conversation in OpenAI message format.
            **kwargs: Override sampling parameters.

        Returns:
            ``GenerationResult`` with both text and training metadata.
        """
        if messages is not None and prompt is None:
            prompt = self._apply_template(messages)
        elif prompt is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        self._call_count += 1
        raw = await self._call_generator(prompt, **kwargs)
        return self._to_generation_result(raw)

    async def generate_raw(self, prompt: str, **kwargs: Any) -> dict:
        """Low-level generation returning the raw Generator response dict.

        Useful for adapters that need full control over response parsing.
        """
        return await self._call_generator(prompt, **kwargs)

    def _apply_template(self, messages: list[dict[str, str]]) -> str:
        """Format messages using the chat template."""
        if self._chat_template is None:
            from forge.core.chat_template import CHATML

            logger.warning("ModelProxy: no chat_template set, falling back to ChatML")
            self._chat_template = CHATML
        return self._chat_template.apply_with_generation_prompt(messages)

    async def _call_generator(self, prompt: str, **kwargs: Any) -> dict:
        """Dispatch to Generator via ServiceInterface.route() or ActorMesh.call_one()."""
        if self._generator is None:
            raise RuntimeError("ModelProxy: generator not configured")

        gen_ep = getattr(self._generator, "generate", None)
        if gen_ep is None:
            raise RuntimeError("ModelProxy: generator has no 'generate' endpoint")

        if hasattr(gen_ep, "route"):
            results = await gen_ep.route(prompt)
        else:
            results = await gen_ep.call_one(prompt)

        if isinstance(results, list) and results:
            return results[0]
        if isinstance(results, dict):
            return results
        return {"text": str(results)}

    @staticmethod
    def _to_generation_result(raw: dict) -> GenerationResult:
        """Convert a raw Generator response dict into a ``GenerationResult``."""
        logprobs = raw.get("logprobs", [])
        token_ids = raw.get("token_ids", [])
        if not isinstance(logprobs, list):
            logprobs = [0.0] * len(token_ids)

        return GenerationResult(
            text=raw.get("text", ""),
            token_ids=list(token_ids),
            logprobs=list(logprobs),
            version=raw.get("generator_version", -1),
            raw=raw,
        )

    def get_stats(self) -> dict:
        return {"call_count": self._call_count}
