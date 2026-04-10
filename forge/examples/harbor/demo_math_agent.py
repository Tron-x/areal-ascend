"""Demo: Harbor MathAgent → Forge adapter → real vLLM inference on NPU.

Runs a batch of GSM8K math problems through the full chain:
1. Load tasks via harbor data loader
2. Send to vLLM for inference (real model on NPU)
3. Parse response with HarborAgentLogic
4. Compute reward with rllm math_reward_fn
5. Convert to Forge Episode via adapter

No Docker needed — pure Python + NPU inference.

Usage::

    # Set CANN environment first
    source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh

    python forge/examples/harbor/demo_math_agent.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --num-tasks 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

sys.path.insert(0, "/root/harbor/harbor-verl-train")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("HarborDemo")


def parse_args():
    p = argparse.ArgumentParser(description="Harbor MathAgent demo on NPU")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--num-tasks", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    return p.parse_args()


async def run_demo(args):
    from forge.agents.harbor import HarborAgentLogic
    from forge.examples.harbor.adapter import trajectory_to_forge_episode
    from forge.examples.harbor.data import load_gsm8k
    from forge.examples.harbor.reward import harbor_math_reward

    from rllm.agents.agent import Step, Trajectory
    from rllm.engine.rollout import ModelOutput

    logger.info("Loading model: %s", args.model)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM not available. Install with: pip install vllm")
        return

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
    )

    logger.info("Loading %d GSM8K tasks...", args.num_tasks)
    tasks = load_gsm8k(split="test", max_samples=args.num_tasks)

    logic = HarborAgentLogic(max_turns=1)

    results = {
        "total": 0,
        "correct": 0,
        "episodes": [],
    }

    prompts = []
    for task in tasks:
        messages = task["messages"]
        prompt_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        prompts.append(prompt_text)

    logger.info("Generating responses for %d tasks...", len(prompts))
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    logger.info("Generation done in %.1fs (%.1f tasks/sec)", gen_time, len(prompts) / gen_time)

    for i, (task, output) in enumerate(zip(tasks, outputs)):
        response = output.outputs[0].text
        token_ids = list(output.outputs[0].token_ids)
        logprobs_data = output.outputs[0].logprobs or []
        logprobs = []
        for lp in logprobs_data:
            if lp and isinstance(lp, dict):
                top_token = max(lp.values(), key=lambda x: x.logprob if hasattr(x, 'logprob') else x)
                logprobs.append(top_token.logprob if hasattr(top_token, 'logprob') else float(top_token))
            else:
                logprobs.append(0.0)

        action = logic.process_response(response, task["messages"])

        reward = harbor_math_reward(
            prompt=task["question"],
            response=response,
            target=task["ground_truth"],
        )

        mo = ModelOutput(
            text=response,
            content=response,
            reasoning="",
            tool_calls=[],
            prompt_ids=list(output.prompt_token_ids),
            completion_ids=token_ids,
            prompt_length=len(output.prompt_token_ids),
            completion_length=len(token_ids),
            finish_reason=output.outputs[0].finish_reason or "stop",
            rollout_log_probs=logprobs[:len(token_ids)] if logprobs else None,
        )

        step = Step(
            chat_completions=task["messages"] + [{"role": "assistant", "content": response}],
            thought="",
            model_response=response,
            model_output=mo,
            reward=reward,
            done=action.done,
        )

        traj = Trajectory(
            uid=f"gsm8k-demo-{i}",
            name="math",
            steps=[step],
            reward=reward,
        )

        episode = trajectory_to_forge_episode(traj, task=task)

        results["total"] += 1
        if reward > 0:
            results["correct"] += 1

        status = "CORRECT" if reward > 0 else "WRONG"
        logger.info(
            "[%d/%d] %s | Q: %.50s... | A: %s | Pred: %.50s...",
            i + 1, len(tasks), status,
            task["question"],
            task["ground_truth"],
            response.replace("\n", " ")[:50],
        )

        results["episodes"].append({
            "task_id": i,
            "question": task["question"][:80],
            "ground_truth": task["ground_truth"],
            "done": action.done,
            "reward": reward,
            "response_len": len(response),
            "token_count": len(token_ids),
            "episode_id": episode.episode_id,
            "forge_token_ids_len": len(episode.token_ids),
            "forge_loss_mask_sum": sum(episode.loss_mask),
        })

    accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0

    print("\n" + "=" * 60)
    print(f" Harbor MathAgent → Forge Adapter Demo Results")
    print("=" * 60)
    print(f" Model:       {args.model}")
    print(f" Tasks:       {results['total']}")
    print(f" Correct:     {results['correct']}")
    print(f" Accuracy:    {accuracy:.1%}")
    print(f" Gen time:    {gen_time:.1f}s")
    print("=" * 60)

    print("\nPer-task results:")
    for r in results["episodes"]:
        mark = "v" if r["reward"] > 0 else "x"
        print(
            f"  [{mark}] #{r['task_id']:2d} | "
            f"answer={r['ground_truth']:>6s} | "
            f"done={r['done']} | "
            f"tokens={r['token_count']:4d} | "
            f"forge_episode={r['episode_id']}"
        )

    print(f"\nForge Episode stats (sample):")
    if results["episodes"]:
        r = results["episodes"][0]
        print(f"  token_ids length:  {r['forge_token_ids_len']}")
        print(f"  loss_mask sum:     {r['forge_loss_mask_sum']} (trainable tokens)")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_demo(args))
