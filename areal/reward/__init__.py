import threading

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from areal.utils import logging

logger = logging.getLogger("RewardUtils")


def _patch_math_verify_for_threads():
    """Make math_verify's signal-based timeout a no-op in non-main threads.

    math_verify uses signal.alarm() in its timeout decorator (utils.timeout),
    but signal.alarm() only works in the main thread.  Monarch actors run
    in non-main threads, causing ValueError on every reward call.

    We replace the timeout decorator itself so that in non-main threads it
    returns a no-op wrapper.  This fixes both parser.parse() and grader.verify().
    """
    import math_verify.grader as _grader
    import math_verify.metric as _metric
    import math_verify.parser as _parser
    import math_verify.utils as _utils

    _original_timeout = _utils.timeout

    def _thread_safe_timeout(timeout_seconds=10):
        if threading.current_thread() is not threading.main_thread():
            def no_timeout_decorator(func):
                return func
            return no_timeout_decorator
        return _original_timeout(timeout_seconds)

    _utils.timeout = _thread_safe_timeout
    _parser.timeout = _thread_safe_timeout
    _grader.timeout = _thread_safe_timeout
    _metric.timeout = _thread_safe_timeout


_patch_math_verify_for_threads()

VALID_REWARD_FN = ["clevr_count_70k", "geometry3k"]


def get_custom_reward_fn(path: str, **kwargs):
    if "clevr_count_70k" in path:
        from .clevr_count_70k import clevr_count_70k_reward_fn

        return clevr_count_70k_reward_fn
    elif "geometry3k" in path:
        from .geometry3k import geometry3k_reward_fn

        return geometry3k_reward_fn
    else:
        raise ValueError(
            f"Reward function {path} is not supported. "
            f"Supported reward functions are: {VALID_REWARD_FN}. "
        )


class MathVerifyWorker:
    """Thin wrapper over math_verify with configurable extraction/precision.

    Args:
        try_extract_without_anchor: When False, only answers with explicit anchors
            (e.g., "answer = 1", "final answer = 1") are matched. When True,
            any numeric string in the text may be extracted.
        precision: Number of significant digits that must match.

    Notes:
        Tune these knobs based on dataset format and model output style.
    """

    def __init__(self, try_extract_without_anchor=True, precision: int = 6):
        self.verify_func = math_metric(
            gold_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            pred_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            precision=precision,
        )

    def verify(self, response: str, ground_truth: str) -> float:
        # ground_truth_parsable = "\\boxed{" + ground_truth + "}"
        try:
            ret_score, _ = self.verify_func([ground_truth], [response])
            return float(ret_score)
        except Exception:
            logger.warning(
                f"Exception in MathVerifyWorker.verify for response={response} and ground_truth={ground_truth}",
                exc_info=True,
            )
            return 0.0


_MATH_VERIFY_WORKER: MathVerifyWorker | None = None


def get_math_verify_worker() -> MathVerifyWorker:
    global _MATH_VERIFY_WORKER
    if _MATH_VERIFY_WORKER is None:
        _MATH_VERIFY_WORKER = MathVerifyWorker()
    return _MATH_VERIFY_WORKER


__all__ = [
    "VALID_REWARD_FN",
    "get_custom_reward_fn",
    "MathVerifyWorker",
    "get_math_verify_worker",
    "gsm8k_reward_fn",
    "geometry3k_reward_fn",
    "clevr_count_70k_reward_fn",
]


_LAZY_IMPORTS = {
    "gsm8k_reward_fn": "areal.reward.gsm8k",
    "geometry3k_reward_fn": "areal.reward.geometry3k",
    "clevr_count_70k_reward_fn": "areal.reward.clevr_count_70k",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
