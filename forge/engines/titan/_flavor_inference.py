"""Infer ``(model_name, model_flavor)`` for TorchTitan from an HF model path.

Background
----------
TorchTitan's model factory (``torchtitan/models/<family>/__init__.py``)
keys its variants by short flavor strings ("0.6B", "1.7B", "8B", ...)
that map to a fixed ``Qwen3ModelArgs`` / ``LlamaModelArgs`` /
``DeepSeekV3ModelArgs`` etc.  The HF model directory we hand the
*inference* engine (vLLM) carries the same architecture in
``config.json`` (``hidden_size`` / ``num_hidden_layers`` / ...), but
**there is no automatic mapping back to the titan flavor string**.

When the two sides drift -- e.g. the user passes
``--model /path/to/Qwen3-0.6B`` to vLLM but the trainer-side titan
defaults to ``model_flavor="1.7B"`` -- weight sync silently produces
zero matched keys (every trainer tensor is 2x the size of the vLLM
counterpart) and the run quietly degenerates to "rollouts only, no
policy update".  See:

    /tmp/forge_wsactor_smoke/driver.log
    "[151936, 1024] vs [151936, 2048]"

This module closes that foot-gun: given the HF directory we already
hand vLLM, it reads ``config.json`` and reverses it to a titan flavor
string by exact match on the (dim, n_layers) tuple recorded in
:data:`_QWEN3_FLAVORS` etc.

API contract
------------
:func:`infer_titan_flavor_from_hf_path` returns ``(name, flavor)`` on
success and ``(None, None)`` on miss (unknown architecture, unknown
shape combination, missing config.json).  The caller is expected to
treat ``(None, None)`` as "fall back to user-supplied default" and
emit a warning -- never silently substitute, that's exactly the bug
this module exists to prevent.

Adding a new family
-------------------
Add a ``_FAMILY_FLAVORS`` table below.  Key by the tuple that uniquely
identifies the flavor in HF config; for Qwen3 / Llama / DeepSeek so far
``(hidden_size, num_hidden_layers)`` suffices; for MoE families also
include ``num_experts``.  Then register the architecture name(s) in
:data:`_ARCH_TO_FAMILY`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- flavor tables
# (dim, n_layers) -> titan flavor string.  Mirror of
# ``torchtitan/models/qwen3/__init__.py::qwen3_args``.  When titan
# adds a new flavor, mirror it here -- there's no programmatic way to
# discover them at import time without taking a hard dep on
# torchtitan in this module (which we deliberately avoid; this helper
# runs on the driver before any GPU / titan import).
_QWEN3_DENSE_FLAVORS: dict[tuple[int, int], str] = {
    (1024, 28): "0.6B",
    (2048, 28): "1.7B",
    (2560, 36): "4B",
    (4096, 36): "8B",
    (5120, 40): "14B",
    (5120, 64): "32B",
}

# MoE Qwen3 variants are keyed additionally by ``num_experts``
# because (dim, n_layers) alone collides with dense models.
# (dim, n_layers, num_experts) -> flavor.
_QWEN3_MOE_FLAVORS: dict[tuple[int, int, int], str] = {
    (2048, 48, 128): "30B-A3B",
    (4096, 94, 160): "235B-A22B",
}

# Architectures string in HF config.json -> our internal family key.
# Multiple arch strings can map to the same family (e.g. dense +
# instruct variants share the same titan model code).
_ARCH_TO_FAMILY: dict[str, str] = {
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3MoeForCausalLM": "qwen3",
    # llama3 / deepseek_v3 / llama4 entries go here as we wire those
    # families into the inference path.
}


# ---------------------------------------------------------------- public API


def infer_titan_flavor_from_hf_path(
    hf_path: str,
) -> tuple[str | None, str | None]:
    """Read ``config.json`` from ``hf_path`` and return ``(name, flavor)``.

    Args:
        hf_path: Local directory containing ``config.json`` (e.g. a
            ModelScope / HuggingFace snapshot dir).  Remote
            ``"org/repo"`` IDs are NOT resolved here -- the caller is
            expected to have downloaded the model already (the same
            directory it hands to vLLM).  Pass an absolute path or a
            relative path resolvable from CWD.

    Returns:
        ``(model_name, model_flavor)`` on success, e.g.
        ``("qwen3", "0.6B")``.  ``(None, None)`` when the path is
        missing, ``config.json`` is unreadable, the architecture is
        unknown, or the (dim, n_layers) tuple doesn't match any
        registered flavor.  Caller should warn loudly when (None,
        None) is returned and fall back to its existing default --
        never silently substitute.
    """
    if not hf_path:
        return None, None

    cfg_path = os.path.join(hf_path, "config.json")
    if not os.path.isfile(cfg_path):
        logger.debug(
            "infer_titan_flavor_from_hf_path: no config.json under %s", hf_path
        )
        return None, None

    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "infer_titan_flavor_from_hf_path: failed to read %s: %s",
            cfg_path,
            e,
        )
        return None, None

    return _infer_from_config(cfg)


def _infer_from_config(cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pure-dict variant -- exposed for unit tests + future callers.

    Splits the file IO from the inference logic so tests don't need
    to populate a temp directory just to exercise edge cases (missing
    fields, MoE vs dense disambiguation, etc.).
    """
    archs = cfg.get("architectures") or []
    if not archs:
        logger.debug("infer: no 'architectures' field in config")
        return None, None

    # Pick the first architecture we recognize; HF rarely lists more
    # than one but let's not lose to a future model that does.
    family: str | None = None
    for arch in archs:
        if arch in _ARCH_TO_FAMILY:
            family = _ARCH_TO_FAMILY[arch]
            break
    if family is None:
        logger.debug("infer: no known family for archs=%s", archs)
        return None, None

    if family == "qwen3":
        return _infer_qwen3_flavor(cfg)

    # Other families fall through here; add branches as we wire them.
    logger.debug("infer: family=%s registered but no flavor table yet", family)
    return None, None


def _infer_qwen3_flavor(cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    """Match an HF Qwen3 config to a titan flavor string."""
    dim = cfg.get("hidden_size")
    n_layers = cfg.get("num_hidden_layers")
    if dim is None or n_layers is None:
        logger.debug("infer qwen3: missing hidden_size / num_hidden_layers")
        return None, None

    # MoE variants disambiguate on num_experts; HF puts that under
    # different keys depending on the source (HF native vs ms-swift
    # repackage).  Try both.
    num_experts = cfg.get("num_experts") or cfg.get("num_local_experts")
    if cfg.get("model_type") == "qwen3_moe" or num_experts is not None:
        moe_flavor = _QWEN3_MOE_FLAVORS.get(
            (int(dim), int(n_layers), int(num_experts or 0))
        )
        if moe_flavor is not None:
            return "qwen3", moe_flavor
        logger.debug(
            "infer qwen3 MoE: no flavor for dim=%d n_layers=%d num_experts=%s",
            dim,
            n_layers,
            num_experts,
        )
        return None, None

    flavor = _QWEN3_DENSE_FLAVORS.get((int(dim), int(n_layers)))
    if flavor is not None:
        return "qwen3", flavor
    logger.debug("infer qwen3: no flavor for dim=%d n_layers=%d", dim, n_layers)
    return None, None


__all__ = ["infer_titan_flavor_from_hf_path"]
