"""Demo: Multi-turn tool-calling agent on NPU with Harbor adapter.

Runs GSM8K math problems using Qwen's tool-call format:
1. Model generates ``<tool_call>{"name":"python","arguments":{"code":"..."}}</tool_call>``
2. ToolExecutor runs the Python code locally
3. Result is fed back as ``<tool_response>...</tool_response>``
4. Model continues until it produces ``\\boxed{answer}``

Compares single-turn vs multi-turn accuracy.

Usage::

    source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
    python forge/examples/harbor/demo_tool_agent.py --num-tasks 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

sys.path.insert(0, "/root/harbor/harbor-verl-train")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ToolAgent")

TOOL_SCHEMA = json.dumps([{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code and return the output. Use print() to show results.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
}], indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-turn tool-calling agent demo")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--num-tasks", type=int, default=10)
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--parser", default="qwen", choices=["qwen", "r1"])
    return p.parse_args()


def main():
    args = parse_args()

    from forge.agents.harbor import HarborAgentLogic
    from forge.examples.harbor.data import load_gsm8k
    from forge.examples.harbor.reward import harbor_math_reward
    from forge.examples.harbor.tool_env import ToolExecutor

    logger.info("=" * 60)
    logger.info(" Multi-turn Tool-Calling Agent Demo")
    logger.info(" Model: %s | Parser: %s | Max turns: %d", args.model, args.parser, args.max_turns)
    logger.info("=" * 60)

    logic = HarborAgentLogic(
        parser_name=args.parser,
        max_turns=args.max_turns,
        done_pattern=r"\\boxed\{",
    )
    logger.info("Agent logic: %s", logic)

    tool_prompt = logic.get_tool_prompt(TOOL_SCHEMA)
    executor = ToolExecutor(timeout=15, allowed_tools={"python", "code_execution"})

    logger.info("Loading vLLM...")
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

    tasks = load_gsm8k(split="test", max_samples=args.num_tasks)
    logger.info("Loaded %d tasks", len(tasks))

    single_correct = 0
    multi_correct = 0
    total = len(tasks)

    for i, task in enumerate(tasks):
        question = task["question"]
        answer = task["ground_truth"]
        logger.info("")
        logger.info("[%d/%d] Q: %s", i + 1, total, question[:80])

        system_msg = (
            "You are a math assistant. Solve the problem step by step. "
            "You can use the python tool to compute. "
            "Put your final answer in \\boxed{answer}.\n\n"
            f"{tool_prompt}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ]

        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        output = llm.generate([prompt], sampling_params)[0]
        single_response = output.outputs[0].text

        single_reward = harbor_math_reward(prompt=question, response=single_response, target=answer)
        if single_reward > 0:
            single_correct += 1

        full_response = single_response
        multi_reward = single_reward

        if multi_reward <= 0:
            conversation = list(messages)
            current_response = single_response

            for turn in range(args.max_turns):
                action = logic.process_response(current_response, conversation)

                if action.done:
                    multi_reward = harbor_math_reward(prompt=question, response=full_response, target=answer)
                    break

                if action.tool_calls:
                    tool_results = executor.execute(action.tool_calls)
                    feedback = logic.format_feedback(action, tool_results, 0.0)
                    logger.info("  Turn %d: %d tool calls, feedback: %s", turn + 1, len(action.tool_calls), feedback[:80])
                else:
                    feedback = logic.format_feedback(action, [], 0.0)

                conversation.append({"role": "assistant", "content": current_response})
                conversation.append({"role": "user", "content": feedback})

                next_prompt = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
                output = llm.generate([next_prompt], sampling_params)[0]
                current_response = output.outputs[0].text
                full_response += "\n" + current_response

                multi_reward = harbor_math_reward(prompt=question, response=full_response, target=answer)
                if multi_reward > 0 or not logic.should_continue(turn, multi_reward):
                    break

        if multi_reward > 0:
            multi_correct += 1

        s_mark = "v" if single_reward > 0 else "x"
        m_mark = "v" if multi_reward > 0 else "x"
        logger.info("  Answer: %s | Single: [%s] | Multi: [%s]", answer, s_mark, m_mark)

    print("\n" + "=" * 60)
    print(" Tool-Calling Agent Results")
    print("=" * 60)
    print(f" Model:       {args.model}")
    print(f" Parser:      {args.parser}")
    print(f" Tasks:       {total}")
    print(f" Single-turn: {single_correct}/{total} ({100*single_correct/total:.0f}%)")
    print(f" Multi-turn:  {multi_correct}/{total} ({100*multi_correct/total:.0f}%)")
    delta = multi_correct - single_correct
    print(f" Improvement: {'+' if delta >= 0 else ''}{delta} tasks")
    print("=" * 60)


if __name__ == "__main__":
    main()
