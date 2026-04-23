from forge.reward import register_reward

from areal.utils import logging

from . import get_math_verify_worker

logger = logging.getLogger("GSM8KReward")


@register_reward("gsm8k")
def gsm8k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        logger.warning("Exception in gsm8k_reward_fn", exc_info=True)
        return 0.0
