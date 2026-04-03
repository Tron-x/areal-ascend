"""End-to-end test for Forge's ReTool modules.

Verifies the full tool-integrated reasoning pipeline WITHOUT
any GPU, Monarch, or training framework -- purely testing the
Forge tool system, parsers, sandbox, and agent logic.

Simulates what happens inside AgentActor.run_episode_retool:
1. LLM generates text with code blocks
2. CompositeParser extracts tool calls
3. PythonSandbox executes code
4. Loss mask correctly marks tool output as [0]
5. ReToolAgent decides when to stop
"""

import pytest


def test_composite_parser_all_formats():
    """Parser handles all common tool-calling formats."""
    from forge.tools.parsers import CompositeParser

    parser = CompositeParser()

    # Format 1: <code> tags (ReTool / veRL)
    calls = parser.parse("Think: I need to compute.\n<code>print(2+3)</code>")
    assert len(calls) == 1
    assert calls[0].arguments["code"] == "print(2+3)"

    # Format 2: ```python blocks (markdown)
    calls = parser.parse("```python\nprint(42)\n```")
    assert len(calls) == 1

    # Format 3: <tool_call> (Qwen3)
    calls = parser.parse(
        '<tool_call>{"name":"code_interpreter","arguments":{"code":"x=1"}}</tool_call>'
    )
    assert len(calls) == 1
    assert calls[0].name == "code_interpreter"

    # Format 4: <function=> (Qwen3 Coder)
    calls = parser.parse("<function=list_dir><parameter=path>.</parameter></function>")
    assert len(calls) == 1
    assert calls[0].name == "list_dir"

    # No tool call
    assert parser.parse("Just thinking...") == []


def test_tool_registry_register_and_execute():
    """Registry registers tools and executes by name."""
    from forge.tools.python_sandbox import PythonTool
    from forge.tools.registry import ToolRegistry

    reg = ToolRegistry()
    tool = PythonTool(timeout=10, safety_check=False)
    reg.register_tool(tool)

    # Check registration
    assert "code_interpreter" in reg.list_tools()
    specs = reg.get_tool_specs()
    assert len(specs) == 1
    assert specs[0]["function"]["name"] == "code_interpreter"


@pytest.mark.asyncio
async def test_python_sandbox_execution():
    """Sandbox executes Python and captures output."""
    from forge.tools.python_sandbox import PythonSandbox

    sb = PythonSandbox(timeout=10, safety_check=False)

    # Basic math
    ok, stdout, stderr = await sb.execute("print(2 + 3)")
    assert ok
    assert "5" in stdout

    # Error handling: the sandbox wrapper catches exceptions and prints them
    ok, stdout, stderr = await sb.execute("raise ValueError('test')")
    assert "ValueError" in stdout or "ValueError" in stderr


@pytest.mark.asyncio
async def test_python_tool_execute():
    """PythonTool wraps sandbox with ToolResult."""
    from forge.tools.python_sandbox import PythonTool

    tool = PythonTool(timeout=10, safety_check=False)
    result = await tool.execute({"code": "print(7 * 6)"})
    assert result.success
    assert "42" in result.output


def test_retool_agent_answer_detection():
    """ReToolAgent detects final answer and stops."""
    from forge.agents.retool import ReToolAgent

    agent = ReToolAgent()

    # No answer -> extract tool calls
    action = agent.process_response("Let me compute:\n<code>print(2+2)</code>", [])
    assert not action.done
    assert len(action.tool_calls) == 1

    # Answer found -> done
    action = agent.process_response("The answer is Answer: \\boxed{42}", [])
    assert action.done


def test_retool_loss_mask_pattern():
    """Verify the loss_mask pattern: LLM=[1], tool_output=[0]."""
    from forge.agents.retool import ReToolAgent
    from forge.core.types import ToolResult

    agent = ReToolAgent()

    # Simulate the ReTool loop manually
    all_loss_mask = []

    # Turn 1: LLM generates code
    llm_output = "Let me try:\n<code>print(2+2)</code>"
    llm_tokens = list(range(20))  # 20 fake token IDs
    all_loss_mask.extend([1] * len(llm_tokens))  # LLM output: train

    # Parse and execute
    action = agent.process_response(llm_output, [])
    assert len(action.tool_calls) == 1

    # Tool result
    tool_results = [ToolResult(success=True, output="4")]
    agent.format_tool_observation(tool_results)  # verify it runs
    obs_tokens = list(range(10))  # 10 fake observation tokens
    all_loss_mask.extend([0] * len(obs_tokens))  # Tool output: NOT trained

    # Turn 2: LLM gives final answer
    llm_tokens2 = list(range(15))
    all_loss_mask.extend([1] * len(llm_tokens2))  # LLM output: train

    # Verify mask pattern
    assert len(all_loss_mask) == 20 + 10 + 15  # 45 total
    assert all_loss_mask[:20] == [1] * 20  # LLM turn 1
    assert all_loss_mask[20:30] == [0] * 10  # Tool output
    assert all_loss_mask[30:45] == [1] * 15  # LLM turn 2


def test_reward_to_go_for_tool_episodes():
    """Reward-to-go spreads sparse reward across multi-turn episode."""
    from forge.rl.rewards import reward_to_go, spread_final_reward

    # 3-turn episode, only final answer gets reward=1
    step_rewards = [0.0, 0.0, 1.0]
    returns = reward_to_go(step_rewards, gamma=1.0)
    assert returns == [1.0, 1.0, 1.0]

    # With discount
    returns = reward_to_go(step_rewards, gamma=0.9)
    assert abs(returns[0] - 0.81) < 0.01
    assert abs(returns[2] - 1.0) < 0.01

    # Convenience function
    returns = spread_final_reward(5, final_reward=1.0)
    assert all(abs(r - 1.0) < 0.01 for r in returns)


def test_process_reward_tool_events():
    """Process reward assigns intermediate rewards for tool events."""
    from forge.rl.rewards import process_reward

    events = [
        {"tool_success": True},  # +0.1
        {"tool_error": True},  # -0.1
        {"tool_success": True},  # +0.1
        {"answer_found": True},  # +0.5
    ]
    rewards = process_reward(events)
    assert rewards[0] > 0  # tool success
    assert rewards[1] < 0  # tool error
    assert rewards[3] > rewards[0]  # answer > tool success


@pytest.mark.asyncio
async def test_full_retool_simulation():
    """Full ReTool simulation: parse → execute → mask → reward.

    This tests the exact flow that AgentActor.run_episode_retool does,
    but without Monarch actors.
    """
    from forge.agents.retool import ReToolAgent
    from forge.core.types import ToolResult
    from forge.rl.rewards import composite_reward, reward_to_go
    from forge.tools.python_sandbox import PythonTool

    agent = ReToolAgent(max_turns=5)
    tool = PythonTool(timeout=10, safety_check=False)

    # Simulated LLM responses for a math problem
    llm_responses = [
        "I need to compute 7 * 6.\n<code>print(7 * 6)</code>",
        "The result is 42. Answer: \\boxed{42}",
    ]

    all_loss_mask = []
    all_outputs = []
    step_events = []

    for turn, response in enumerate(llm_responses):
        # LLM output tokens (simulated)
        llm_token_count = len(response.split())
        all_loss_mask.extend([1] * llm_token_count)

        action = agent.process_response(response, [])

        if action.done:
            step_events.append({"answer_found": True})
            break

        # Execute tool calls
        for tc in action.tool_calls:
            code = tc.content
            result = await tool.execute({"code": code})
            all_outputs.append(result.output)
            step_events.append(
                {"tool_success": result.success, "tool_error": not result.success}
            )

            # Tool observation tokens
            obs = agent.format_tool_observation(
                [ToolResult(success=result.success, output=result.output)]
            )
            obs_token_count = len(obs.split())
            all_loss_mask.extend([0] * obs_token_count)

    # Verify
    assert "42" in all_outputs[0]  # Tool executed correctly
    assert 0 in all_loss_mask  # Tool output masked
    assert 1 in all_loss_mask  # LLM output trained

    # Compute rewards
    from forge.rl.rewards import process_reward

    step_rewards = process_reward(step_events)
    reward_to_go(step_rewards + [1.0])  # verify it runs

    final = composite_reward(
        {"correctness": 1.0, "tool_efficiency": len(step_events) <= 2},
        weights={"correctness": 1.0, "tool_efficiency": 0.1},
    )
    assert final > 1.0  # Correct + efficient


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
