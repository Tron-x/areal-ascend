"""End-to-end SWE-bench pipeline: NPU inference + x86 Docker evaluation.

Demonstrates the full Harbor RL training loop:
1. Load SWE-bench task (instruction.md)
2. Generate a code patch with vLLM on NPU
3. Evaluate the patch in a Docker container on a remote x86 server
4. Collect reward and construct Forge Episode

Usage::

    source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh

    python forge/examples/harbor/demo_swe_pipeline.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --x86-host 142.171.20.182 \\
        --num-tasks 5
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
logger = logging.getLogger("SWE-Pipeline")

SWE_SYSTEM_PROMPT = """\
You are an expert software engineer. You are given a bug report for a Python project.
Your task is to write a minimal patch (in unified diff format) that fixes the described issue.

Rules:
- Output ONLY the patch in a ```diff ... ``` code block.
- The patch should be applicable with `git apply`.
- Keep changes minimal — only fix the bug, do not refactor.
- Do not add tests or documentation changes.
"""


def parse_args():
    p = argparse.ArgumentParser(description="SWE-bench NPU+x86 pipeline")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--x86-host", default="142.171.20.182")
    p.add_argument("--x86-user", default="root")
    p.add_argument("--task-dir", default="/data/harbor_swe_tasks/v0.0.2/harbor_swe_tasks")
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cleanup", action="store_true", help="Delete Docker images after evaluation")
    return p.parse_args()


def load_swe_tasks(task_dir: str, num_tasks: int) -> list[dict]:
    """Load SWE-bench tasks, preferring small projects."""
    task_path = Path(task_dir)
    if not task_path.exists():
        logger.error("Task directory not found: %s", task_dir)
        return []

    all_tasks = sorted(
        [d.name for d in task_path.iterdir() if d.is_dir()],
        key=len,
    )

    tasks = []
    for task_id in all_tasks:
        if len(tasks) >= num_tasks:
            break

        instruction_file = task_path / task_id / "instruction.md"
        toml_file = task_path / task_id / "task.toml"

        if not instruction_file.exists() or not toml_file.exists():
            continue

        instruction = instruction_file.read_text().strip()
        if not instruction:
            continue

        image = ""
        for line in toml_file.read_text().splitlines():
            if line.strip().startswith("docker_image"):
                image = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

        tasks.append({
            "task_id": task_id,
            "instruction": instruction,
            "image": image,
        })

    logger.info("Selected %d SWE-bench tasks", len(tasks))
    return tasks


def extract_patch(response: str) -> str:
    """Extract a diff/patch from the model response.

    Handles both closed code blocks (```diff ... ```) and truncated
    blocks where max_tokens was reached before the closing fence.
    """
    import re

    closed_patterns = [
        r"```diff\s*\n(.*?)```",
        r"```patch\s*\n(.*?)```",
        r"```\s*\n(diff --git.*?)```",
        r"```\s*\n(---.*?\+\+\+.*?)```",
    ]
    for pattern in closed_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    truncated_patterns = [
        r"```diff\s*\n(.*)",
        r"```patch\s*\n(.*)",
        r"```\s*\n(diff --git.*)",
    ]
    for pattern in truncated_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content and ("diff --git" in content or "---" in content):
                return content

    return ""


def main():
    args = parse_args()

    from forge.examples.harbor.swe_evaluator import SWEBenchEvaluator

    logger.info("=" * 60)
    logger.info(" SWE-bench Pipeline: NPU inference + x86 Docker eval")
    logger.info(" Model:    %s", args.model)
    logger.info(" x86:      %s@%s", args.x86_user, args.x86_host)
    logger.info(" Tasks:    %d", args.num_tasks)
    logger.info("=" * 60)

    evaluator = SWEBenchEvaluator(
        host=args.x86_host,
        user=args.x86_user,
        task_data_dir=args.task_dir,
    )

    disk_info = evaluator.get_remote_disk_usage()
    logger.info("x86 disk: %s", disk_info.split("\n")[0])

    tasks = load_swe_tasks(args.task_dir, args.num_tasks)
    if not tasks:
        logger.error("No tasks found!")
        return

    for t in tasks:
        logger.info("  Task: %s (image: %s)", t["task_id"], t["image"][:60])

    logger.info("Loading vLLM model on NPU...")
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
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
    )

    results = []

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        logger.info("")
        logger.info("=" * 50)
        logger.info("[%d/%d] Task: %s", i + 1, len(tasks), task_id)
        logger.info("=" * 50)

        prompt = (
            f"System: {SWE_SYSTEM_PROMPT}\n\n"
            f"User:\n{task['instruction']}\n\n"
            f"Please write a patch to fix this issue."
        )

        logger.info("Generating patch on NPU...")
        t0 = time.time()
        output = llm.generate([prompt], sampling_params)[0]
        gen_time = time.time() - t0
        response = output.outputs[0].text
        logger.info("Generated in %.1fs (%d tokens)", gen_time, len(output.outputs[0].token_ids))

        patch = extract_patch(response)
        if patch:
            logger.info("Extracted patch (%d chars):\n%s", len(patch), patch[:500])
        else:
            logger.warning("No patch extracted from response")
            logger.info("Response preview: %s", response[:300])

        logger.info("Evaluating on x86 Docker...")
        eval_result = evaluator.evaluate(task_id, patch_text=patch)

        status = "PASS" if eval_result.reward > 0 else "FAIL"
        logger.info(
            "Result: %s (reward=%.1f, duration=%.0fs)",
            status, eval_result.reward, eval_result.duration_sec,
        )
        if eval_result.error:
            logger.warning("Error: %s", eval_result.error)

        test_lines = eval_result.test_output.strip().split("\n")
        summary_lines = [l for l in test_lines if "PASSED" in l or "FAILED" in l or "passed" in l or "failed" in l]
        if summary_lines:
            logger.info("Test summary: %s", summary_lines[-1].strip())

        results.append({
            "task_id": task_id,
            "reward": eval_result.reward,
            "gen_time": gen_time,
            "eval_time": eval_result.duration_sec,
            "patch_len": len(patch),
            "had_patch": bool(patch),
            "error": eval_result.error,
        })

        if args.cleanup:
            evaluator.cleanup(task_id)

    print("\n" + "=" * 60)
    print(" SWE-bench Pipeline Results")
    print("=" * 60)
    total = len(results)
    correct = sum(1 for r in results if r["reward"] > 0)
    print(f" Model:       {args.model}")
    print(f" Tasks:       {total}")
    print(f" Resolved:    {correct}/{total} ({100*correct/total:.0f}%)")
    print(f" Avg gen time:  {sum(r['gen_time'] for r in results)/total:.1f}s")
    print(f" Avg eval time: {sum(r['eval_time'] for r in results)/total:.1f}s")
    print("=" * 60)

    print("\nPer-task results:")
    for r in results:
        mark = "v" if r["reward"] > 0 else "x"
        print(
            f"  [{mark}] {r['task_id'][:50]:50s} | "
            f"patch={r['patch_len']:5d}ch | "
            f"gen={r['gen_time']:5.1f}s | "
            f"eval={r['eval_time']:5.0f}s"
        )

    disk_info = evaluator.get_remote_disk_usage()
    print(f"\nx86 disk after run: {disk_info.split(chr(10))[0]}")


if __name__ == "__main__":
    main()
