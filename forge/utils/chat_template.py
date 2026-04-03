"""HuggingFace tokenizer-backed ChatTemplate implementation.

Loads a pretrained tokenizer and delegates to its ``apply_chat_template``
method, which handles model-specific formatting (ChatML for Qwen, Llama-style
for Llama, etc.) automatically based on the ``tokenizer_config.json``.

Requires ``transformers`` to be installed.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenizerChatTemplate:
    """Chat template backed by a HuggingFace tokenizer.

    Loads the tokenizer once on construction and reuses it for all calls.
    Thread-safe since ``apply_chat_template`` is a pure string operation.

    Args:
        model_path: HuggingFace model ID or local path
                    (e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``).
        trust_remote_code: Passed to ``AutoTokenizer.from_pretrained``.
        tokenize: If ``False`` (default), returns a string.
                  If ``True``, returns token IDs.

    Raises:
        ImportError: If ``transformers`` is not installed.
        OSError: If the tokenizer cannot be loaded from ``model_path``.
    """

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        tokenize: bool = False,
    ):
        self._model_path = model_path
        self._tokenize = tokenize
        self._tokenizer = _load_tokenizer(model_path, trust_remote_code)

        if self._tokenizer.chat_template is None:
            logger.warning(
                f"Tokenizer for {model_path!r} has no chat_template. "
                f"Falling back to default ChatML template."
            )

    @property
    def chat_template_str(self) -> str | None:
        """The raw Jinja2 chat template string from the tokenizer, if any."""
        return self._tokenizer.chat_template

    def apply(self, messages: list[dict[str, str]]) -> str:
        """Format messages WITHOUT the generation prompt."""
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=self._tokenize,
            add_generation_prompt=False,
        )

    def apply_with_generation_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format messages WITH the generation prompt appended."""
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=self._tokenize,
            add_generation_prompt=True,
        )

    def __repr__(self) -> str:
        has_template = self._tokenizer.chat_template is not None
        return (
            f"TokenizerChatTemplate(model={self._model_path!r}, "
            f"has_template={has_template})"
        )


@lru_cache(maxsize=8)
def _load_tokenizer(model_path: str, trust_remote_code: bool):
    """Load and cache a HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    resolved = model_path
    if Path(model_path).is_dir():
        resolved = str(Path(model_path).resolve())

    logger.info(f"Loading tokenizer from {resolved!r}")
    return AutoTokenizer.from_pretrained(
        resolved,
        trust_remote_code=trust_remote_code,
    )


def auto_chat_template(
    model_path: str | None = None,
    template_name: str | None = None,
    trust_remote_code: bool = True,
):
    """Factory that returns the best available ChatTemplate.

    Resolution order:
    1. If ``model_path`` is provided, try ``TokenizerChatTemplate``.
    2. If ``template_name`` is a known preset (``"chatml"``, ``"llama"``),
       return the corresponding ``SimpleChatTemplate``.
    3. Fall back to ChatML ``SimpleChatTemplate``.

    Args:
        model_path: HuggingFace model ID or local path.
        template_name: Preset name (``"chatml"``, ``"llama"``).
        trust_remote_code: Passed to tokenizer loader.

    Returns:
        A ``ChatTemplate``-compatible object.
    """
    from forge.core.chat_template import CHATML, LLAMA_STYLE

    if model_path:
        try:
            tmpl = TokenizerChatTemplate(
                model_path, trust_remote_code=trust_remote_code
            )
            logger.info(f"Using TokenizerChatTemplate for {model_path!r}")
            return tmpl
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer for {model_path!r}: {e}. "
                f"Falling back to preset template."
            )

    presets = {"chatml": CHATML, "llama": LLAMA_STYLE}
    if template_name and template_name.lower() in presets:
        return presets[template_name.lower()]

    logger.info("Using default ChatML template")
    return CHATML
