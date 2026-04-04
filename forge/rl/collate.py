"""Batch collation for RL training.

Converts lists of ``Episode`` objects into ``TrainBatch`` format
that any ``TrainEngine`` can consume.
"""

from __future__ import annotations

from forge.core.types import Group, TrainBatch


def collate_episodes(groups: list[Group], pad_id: int = 0) -> list[TrainBatch]:
    """Collate groups of episodes into TrainBatch objects.

    Each group corresponds to one data-parallel shard.

    Args:
        groups: List of groups, each group is a list of Episodes.
        pad_id: Token ID used for padding shorter sequences.

    Returns:
        List of TrainBatch, one per group.
    """
    result = []
    for group in groups:
        if not group:
            continue

        input_ids_list = []
        logprobs_list = []
        loss_mask_list = []
        advantages_list = []
        ref_logprobs_list = []
        has_ref = all(e.ref_logprobs is not None for e in group)

        max_len = max(ep.seq_len() or len(ep.generator_logprobs) for ep in group)

        for ep in group:
            ids = ep.token_ids if ep.token_ids else []
            pad_len = max_len - len(ids)
            input_ids_list.append(ids + [pad_id] * pad_len)

            lp = ep.generator_logprobs
            logprobs_list.append(lp + [0.0] * (max_len - len(lp)))

            lm = ep.loss_mask
            loss_mask_list.append(lm + [0] * (max_len - len(lm)))

            adv = ep.advantage if ep.advantage is not None else 0.0
            advantages_list.append([adv] * max_len)

            if has_ref and ep.ref_logprobs is not None:
                rp = ep.ref_logprobs
                ref_logprobs_list.append(rp + [0.0] * (max_len - len(rp)))

        loss_inputs = {
            "generator_logprobs": logprobs_list,
            "loss_mask": loss_mask_list,
            "advantages": advantages_list,
        }
        if has_ref:
            loss_inputs["ref_logprobs"] = ref_logprobs_list

        result.append(
            TrainBatch(
                model_inputs={"input_ids": input_ids_list},
                loss_inputs=loss_inputs,
            )
        )
    return result
