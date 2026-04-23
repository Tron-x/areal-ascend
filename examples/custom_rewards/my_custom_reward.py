"""Example: register a custom reward function via the forge registry.

End-to-end: write this file once, reference it in YAML as
``reward: exact_match``, and forge's GRPO driver picks it up.

    # your training YAML (gsm8k_grpo_npu.yaml or a copy):
    reward: exact_match

    # forge launch:
    python -m forge launch forge/configs/launcher_bare_metal_2node.yaml \\
        --hostfile forge/configs/hostfile.txt \\
        -- reward=exact_match  [...other overrides...]

No editing any framework code.  No dotted import paths.  Just a
decorator + a YAML line.

To use this file, either import it from your main entry script or
put it under ``forge.reward.*`` / ``areal.reward.*`` where
``forge.reward.ensure_loaded()`` will auto-discover it.  For totally
external packages, add a ``-m`` / import hook in your launch path.
"""

from __future__ import annotations

from forge.reward import register_reward


@register_reward("exact_match")
def exact_match_reward(
    prompt, completions, prompt_ids=None, completion_ids=None, answer=None, **kwargs
) -> float:
    """1.0 if the model's completion contains the ground-truth answer as
    a substring, else 0.0.

    Simple baseline reward useful as a reference for verifying the
    registry + wiring.  Real rewards usually want at least a proper
    answer-extraction step (regex, last-number heuristic, boxed
    extraction, etc.) plus dataset-specific normalization.
    """
    if answer is None:
        return 0.0
    completion_str = str(completions).strip().lower()
    answer_str = str(answer).strip().lower()
    return 1.0 if answer_str and answer_str in completion_str else 0.0
