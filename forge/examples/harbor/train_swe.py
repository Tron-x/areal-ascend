"""SWE-bench GRPO training loop: NPU inference + x86 Docker reward.

Implements a complete RL training cycle:
1. Sample SWE-bench tasks
2. Generate patches with vLLM on NPU
3. Evaluate patches via Docker on remote x86
4. Construct Forge Episodes with reward
5. Run GRPO training step
6. Repeat

This is a standalone training script that demonstrates the full
Harbor-Forge integration without requiring the full AReaL trainer.
It uses vLLM for both generation and logprob computation.

Usage::

    source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh

    python forge/examples/harbor/train_swe.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --x86-host 142.171.20.182 \\
        --train-steps 2 \\
        --tasks-per-step 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/harbor/harbor-verl-train")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SWE-Train")

SWE_SYSTEM_PROMPT = (
    "You are an expert software engineer. You are given a bug report. "
    "Write a minimal unified diff patch that fixes the issue. "
    "Output ONLY the patch in a ```diff ... ``` code block. "
    "Keep changes minimal."
)


def parse_args():
    p = argparse.ArgumentParser(description="SWE-bench GRPO training")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--x86-host", default="142.171.20.182")
    p.add_argument("--x86-user", default="root")
    p.add_argument("--task-dir", default="/data/harbor_swe_tasks/v0.0.2/harbor_swe_tasks")
    p.add_argument("--train-steps", type=int, default=2)
    p.add_argument("--tasks-per-step", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=2, help="Rollouts per task (GRPO group size)")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--cleanup", action="store_true")
    return p.parse_args()


def load_swe_tasks(task_dir: str, n: int) -> list[dict]:
    """Load n complete SWE-bench tasks."""
    task_path = Path(task_dir)
    tasks = []
    for d in sorted(task_path.iterdir(), key=lambda x: len(x.name)):
        if not d.is_dir():
            continue
        inst = d / "instruction.md"
        toml = d / "task.toml"
        test_sh = d / "tests" / "test.sh"
        if not all(f.exists() for f in [inst, toml, test_sh]):
            continue
        instruction = inst.read_text().strip()
        if not instruction:
            continue
        tasks.append({"task_id": d.name, "instruction": instruction})
        if len(tasks) >= n:
            break
    return tasks


def main():
    args = parse_args()

    from forge.examples.harbor.swe_reward import SWERewardFn, extract_patch

    logger.info("=" * 60)
    logger.info(" SWE-bench GRPO Training")
    logger.info(" Model:         %s", args.model)
    logger.info(" x86:           %s@%s", args.x86_user, args.x86_host)
    logger.info(" Train steps:   %d", args.train_steps)
    logger.info(" Tasks/step:    %d", args.tasks_per_step)
    logger.info(" N samples:     %d (GRPO group size)", args.n_samples)
    logger.info("=" * 60)

    reward_fn = SWERewardFn(
        host=args.x86_host,
        user=args.x86_user,
        task_data_dir=args.task_dir,
        cleanup_after=args.cleanup,
    )

    logger.info("Loading vLLM model...")
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM not available")
        return

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
    )

    gen_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.8,
        top_p=0.95,
        logprobs=1,
    )

    all_tasks = load_swe_tasks(args.task_dir, args.tasks_per_step * args.train_steps)
    logger.info("Loaded %d tasks total", len(all_tasks))

    step_stats = []

    for step in range(args.train_steps):
        t_step = time.time()
        logger.info("")
        logger.info("=" * 50)
        logger.info("  TRAINING STEP %d / %d", step + 1, args.train_steps)
        logger.info("=" * 50)

        start = step * args.tasks_per_step
        step_tasks = all_tasks[start : start + args.tasks_per_step]
        if not step_tasks:
            logger.warning("No more tasks available")
            break

        prompts = []
        for task in step_tasks:
            prompt = f"System: {SWE_SYSTEM_PROMPT}\n\nUser:\n{task['instruction']}\n\nWrite a patch:"
            for _ in range(args.n_samples):
                prompts.append(prompt)

        logger.info("Generating %d responses (%d tasks x %d samples)...",
                     len(prompts), len(step_tasks), args.n_samples)
        t_gen = time.time()
        outputs = llm.generate(prompts, gen_params)
        gen_time = time.time() - t_gen
        logger.info("Generation: %.1fs", gen_time)

        episodes = []
        step_rewards = []

        for task_idx, task in enumerate(step_tasks):
            task_id = task["task_id"]
            group_rewards = []

            for sample_idx in range(args.n_samples):
                out_idx = task_idx * args.n_samples + sample_idx
                output = outputs[out_idx]
                response = output.outputs[0].text
                token_ids = list(output.outputs[0].token_ids)
                prompt_ids = list(output.prompt_token_ids)

                logprobs_raw = output.outputs[0].logprobs or []
                logprobs = []
                for lp_dict in logprobs_raw:
                    if lp_dict and isinstance(lp_dict, dict):
                        top = max(lp_dict.values(), key=lambda x: x.logprob if hasattr(x, 'logprob') else x)
                        logprobs.append(top.logprob if hasattr(top, 'logprob') else float(top))
                    else:
                        logprobs.append(0.0)

                patch = extract_patch(response)

                logger.info("  [%s] sample %d: patch=%d chars", task_id, sample_idx, len(patch))

                if patch:
                    result = reward_fn.evaluator.evaluate(task_id, patch_text=patch)
                    reward = result.reward
                else:
                    reward = 0.0

                group_rewards.append(reward)
                step_rewards.append(reward)

                episodes.append({
                    "task_id": task_id,
                    "prompt_ids": prompt_ids,
                    "token_ids": token_ids,
                    "logprobs": logprobs[:len(token_ids)],
                    "reward": reward,
                    "response": response[:200],
                })

            mean_r = sum(group_rewards) / len(group_rewards) if group_rewards else 0
            advantages = [r - mean_r for r in group_rewards]
            for j, adv in enumerate(advantages):
                episodes[task_idx * args.n_samples + j]["advantage"] = adv

            logger.info("  [%s] rewards=%s, mean=%.2f, advantages=%s",
                        task_id,
                        [f"{r:.1f}" for r in group_rewards],
                        mean_r,
                        [f"{a:+.2f}" for a in advantages])

        avg_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0
        n_resolved = sum(1 for r in step_rewards if r > 0)
        step_time = time.time() - t_step

        pos_adv = sum(1 for e in episodes if e.get("advantage", 0) > 0)
        neg_adv = sum(1 for e in episodes if e.get("advantage", 0) < 0)
        zero_adv = sum(1 for e in episodes if e.get("advantage", 0) == 0)

        logger.info("")
        logger.info("  Step %d summary:", step + 1)
        logger.info("    Reward: avg=%.3f, resolved=%d/%d", avg_reward, n_resolved, len(step_rewards))
        logger.info("    Advantages: +%d / -%d / 0:%d", pos_adv, neg_adv, zero_adv)
        logger.info("    Time: gen=%.0fs, total=%.0fs", gen_time, step_time)

        if pos_adv > 0:
            logger.info("    GRPO would reinforce %d positive-advantage episodes", pos_adv)
        else:
            logger.info("    GRPO: all advantages zero/negative (no differentiation in group)")

        step_stats.append({
            "step": step + 1,
            "avg_reward": avg_reward,
            "resolved": n_resolved,
            "total": len(step_rewards),
            "pos_adv": pos_adv,
            "gen_time": gen_time,
            "total_time": step_time,
        })

    print("\n" + "=" * 60)
    print(" SWE-bench GRPO Training Summary")
    print("=" * 60)
    print(f" Model:       {args.model}")
    print(f" Steps:       {args.train_steps}")
    print(f" Tasks/step:  {args.tasks_per_step} x {args.n_samples} samples")
    print()

    for s in step_stats:
        print(
            f"  Step {s['step']}: reward={s['avg_reward']:.3f}, "
            f"resolved={s['resolved']}/{s['total']}, "
            f"pos_adv={s['pos_adv']}, "
            f"time={s['total_time']:.0f}s"
        )

    total_resolved = sum(s["resolved"] for s in step_stats)
    total_episodes = sum(s["total"] for s in step_stats)
    print(f"\n  Overall: {total_resolved}/{total_episodes} resolved "
          f"({100*total_resolved/total_episodes:.0f}%)")
    print("=" * 60)

    print("\nNote: This demo computes GRPO advantages but does not update")
    print("model weights (standalone mode). To run actual weight updates,")
    print("integrate with Forge's TrainerActor via forge.apps.agent_rl.")


if __name__ == "__main__":
    main()
