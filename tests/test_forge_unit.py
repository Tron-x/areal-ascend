"""Pure CPU unit tests for Forge components.

No GPU, no Monarch, no distributed runtime required.
Tests ReplayBuffer, ForgeConfig, SimpleReActAgent, and ChatTemplate
in isolation by directly calling methods (bypassing @endpoint decorator).

Monarch is mocked out so the tests can run in any Python environment.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock the entire monarch package tree so forge.actors can be imported
# without the Rust bindings.
# ---------------------------------------------------------------------------


class _AutoMockModule(types.ModuleType):
    """Module that returns MagicMock for any attribute not explicitly set."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return MagicMock()


def _endpoint_decorator(fn):
    """No-op @endpoint decorator for unit tests."""
    return fn


def _install_monarch_mocks():
    prefixes = [
        "monarch",
        "monarch.actor",
        "monarch._src",
        "monarch._src.actor",
        "monarch._src.actor.actor_mesh",
        "monarch._src.actor.endpoint",
        "monarch._rust_bindings",
        "monarch._rust_bindings.monarch_hyperactor",
        "monarch._rust_bindings.monarch_hyperactor.channel",
    ]
    for p in prefixes:
        if p not in sys.modules:
            sys.modules[p] = _AutoMockModule(p)

    actor_mod = sys.modules["monarch.actor"]
    actor_mod.endpoint = _endpoint_decorator
    actor_mod.Actor = type("Actor", (), {"__init__": lambda self, *a, **k: None})
    actor_mod.current_rank = MagicMock(return_value=MagicMock(rank=0))
    actor_mod.current_size = MagicMock(return_value={"procs": 1})

    mesh_mod = sys.modules["monarch._src.actor.actor_mesh"]
    mesh_mod.current_rank = actor_mod.current_rank
    mesh_mod.context = MagicMock()

    ep_mod = sys.modules["monarch._src.actor.endpoint"]
    ep_mod.EndpointProperty = type("EndpointProperty", (), {})


_install_monarch_mocks()

# ---------------------------------------------------------------------------


def _make_buffer(**kwargs):
    """Create a ReplayBuffer instance without Monarch actor scaffolding."""
    from forge.actors.replay_buffer import EvictionPolicy, ReplayBuffer

    buf = object.__new__(ReplayBuffer)
    from collections import deque

    defaults = {
        "max_size": 100,
        "eviction_policy": "count",
        "max_age_steps": 2,
        "max_sample_count": 4,
    }
    defaults.update(kwargs)
    buf._buffer = deque()
    buf._max_size = defaults["max_size"]
    buf._eviction_policy = EvictionPolicy(defaults["eviction_policy"])
    buf._max_age_steps = defaults["max_age_steps"]
    buf._max_sample_count = defaults["max_sample_count"]
    buf._total_added = 0
    buf._total_sampled = 0
    buf._total_evicted = 0
    return buf


class TestReplayBufferAdd:
    def test_add_single(self):
        buf = _make_buffer()
        result = buf.add({"x": 1}, version=0, step=0)
        assert result["buffer_size"] == 1
        assert result["total_added"] == 1

    def test_add_batch(self):
        buf = _make_buffer()
        items = [{"x": i} for i in range(5)]
        result = buf.add_batch(items, version=1, step=0)
        assert result["buffer_size"] == 5
        assert result["total_added"] == 5
        assert result["batch_added"] == 5

    def test_add_batch_empty(self):
        buf = _make_buffer()
        result = buf.add_batch([], version=0, step=0)
        assert result["buffer_size"] == 0
        assert result["batch_added"] == 0


class TestReplayBufferSample:
    def test_sample_empty_returns_none(self):
        buf = _make_buffer()
        assert buf.sample(batch_size=1) is None

    def test_sample_returns_data(self):
        buf = _make_buffer()
        buf.add({"val": 42}, version=0, step=0)
        result = buf.sample(batch_size=1)
        assert result is not None
        assert len(result) == 1
        assert result[0]["val"] == 42

    def test_sample_respects_batch_size(self):
        buf = _make_buffer()
        for i in range(10):
            buf.add({"i": i}, version=0, step=0)
        result = buf.sample(batch_size=3)
        assert len(result) == 3

    def test_sample_clamps_to_buffer_size(self):
        buf = _make_buffer()
        buf.add({"x": 1}, version=0, step=0)
        result = buf.sample(batch_size=100)
        assert len(result) == 1


class TestReplayBufferVersionFiltering:
    def test_min_version_filters_old_data(self):
        buf = _make_buffer()
        buf.add({"old": True}, version=0, step=0)
        buf.add({"new": True}, version=5, step=5)
        result = buf.sample(batch_size=10, min_version=3)
        assert result is not None
        assert len(result) == 1
        assert result[0]["new"] is True

    def test_min_version_all_filtered_returns_none(self):
        buf = _make_buffer()
        buf.add({"x": 1}, version=0, step=0)
        buf.add({"x": 2}, version=1, step=1)
        result = buf.sample(batch_size=1, min_version=10)
        assert result is None

    def test_min_version_negative_disables_filter(self):
        buf = _make_buffer()
        buf.add({"x": 1}, version=0, step=0)
        result = buf.sample(batch_size=1, min_version=-1)
        assert result is not None
        assert len(result) == 1


class TestReplayBufferEviction:
    def test_count_eviction(self):
        buf = _make_buffer(max_size=3, eviction_policy="count")
        for i in range(5):
            buf.add({"i": i}, version=0, step=0)
        assert buf.buffer_size() == 3
        stats = buf.get_stats()
        assert stats["total_evicted"] == 2

    def test_age_eviction(self):
        buf = _make_buffer(eviction_policy="age", max_age_steps=2)
        buf.add({"old": True}, version=0, step=0)
        buf.add({"mid": True}, version=1, step=2)
        buf.add({"new": True}, version=2, step=4)
        result = buf.sample(batch_size=10, current_step=4)
        assert result is not None
        for item in result:
            assert "old" not in item

    def test_sample_count_eviction(self):
        buf = _make_buffer(max_sample_count=2)
        buf.add({"x": 1}, version=0, step=0)
        buf.sample(batch_size=1)
        buf.sample(batch_size=1)
        assert buf.buffer_size() == 0

    def test_no_eviction_policy(self):
        buf = _make_buffer(max_size=2, eviction_policy="none")
        for i in range(5):
            buf.add({"i": i}, version=0, step=0)
        assert buf.buffer_size() == 5


class TestReplayBufferWaitAndSample:
    def test_wait_and_sample_empty_returns_none(self):
        buf = _make_buffer()
        assert buf.wait_and_sample(batch_size=1) is None

    def test_wait_and_sample_with_data(self):
        buf = _make_buffer()
        buf.add({"x": 1}, version=0, step=0)
        result = buf.wait_and_sample(batch_size=1)
        assert result is not None
        assert len(result) == 1

    def test_wait_and_sample_with_version_filter(self):
        buf = _make_buffer()
        buf.add({"old": True}, version=0, step=0)
        assert buf.wait_and_sample(batch_size=1, min_version=5) is None
        buf.add({"new": True}, version=5, step=5)
        result = buf.wait_and_sample(batch_size=1, min_version=5)
        assert result is not None


class TestReplayBufferStats:
    def test_stats_version_tracking(self):
        buf = _make_buffer()
        buf.add({"a": 1}, version=2, step=0)
        buf.add({"b": 2}, version=5, step=1)
        buf.add({"c": 3}, version=3, step=2)
        stats = buf.get_stats()
        assert stats["min_version"] == 2
        assert stats["max_version"] == 5
        assert stats["buffer_size"] == 3

    def test_stats_empty_buffer(self):
        buf = _make_buffer()
        stats = buf.get_stats()
        assert stats["min_version"] == -1
        assert stats["max_version"] == -1
        assert stats["buffer_size"] == 0


# ======================================================================
# ForgeConfig
# ======================================================================


class TestForgeConfig:
    def test_default_values(self):
        from forge.core.config import ForgeConfig

        cfg = ForgeConfig()
        assert cfg.async_pipeline is False
        assert cfg.replay_buffer_size == 4096
        assert cfg.max_staleness_steps == 2
        assert cfg.agent_mode == "managed"
        assert cfg.model_path == ""
        assert cfg.backend_type == "areal"

    def test_async_pipeline_config(self):
        from forge.core.config import ForgeConfig

        cfg = ForgeConfig(
            async_pipeline=True,
            replay_buffer_size=2048,
            max_staleness_steps=3,
        )
        assert cfg.async_pipeline is True
        assert cfg.replay_buffer_size == 2048
        assert cfg.max_staleness_steps == 3

    def test_resolve_log_dir_default(self):
        from forge.core.config import ForgeConfig

        cfg = ForgeConfig(
            experiment_name="exp1",
            trial_name="trial1",
            fileroot="/tmp/test",
        )
        log_dir = cfg.resolve_log_dir()
        assert "exp1" in log_dir
        assert "trial1" in log_dir

    def test_resolve_log_dir_explicit(self):
        from forge.core.config import ForgeConfig

        cfg = ForgeConfig(log_dir="/custom/logs")
        assert cfg.resolve_log_dir() == "/custom/logs"


# ======================================================================
# SimpleReActAgent (AgentLogic protocol)
# ======================================================================


class TestSimpleReActAgent:
    def test_process_response_extracts_code(self):
        from forge.agents.react import SimpleReActAgent

        agent = SimpleReActAgent(max_turns=3)
        response = "Here is code:\n```python\nprint('hello')\n```\nDone."
        action = agent.process_response(response, [])
        assert len(action.tool_calls) == 1
        assert action.tool_calls[0].type == "code_execution"
        assert "print('hello')" in action.tool_calls[0].content
        assert action.done is False

    def test_process_response_no_code(self):
        from forge.agents.react import SimpleReActAgent

        agent = SimpleReActAgent()
        action = agent.process_response("No code here.", [])
        assert len(action.tool_calls) == 0
        assert action.response == "No code here."

    def test_process_response_multiple_blocks(self):
        from forge.agents.react import SimpleReActAgent

        agent = SimpleReActAgent()
        response = "```python\nx = 1\n```\nand\n```py\ny = 2\n```"
        action = agent.process_response(response, [])
        assert len(action.tool_calls) == 2

    def test_should_continue_positive_reward_stops(self):
        from forge.agents.react import SimpleReActAgent

        agent = SimpleReActAgent(max_turns=5)
        assert agent.should_continue(turn=0, reward=1.0) is False

    def test_should_continue_zero_reward_continues(self):
        from forge.agents.react import SimpleReActAgent

        agent = SimpleReActAgent(max_turns=5)
        assert agent.should_continue(turn=0, reward=0.0) is True
        assert agent.should_continue(turn=3, reward=0.0) is True

    def test_should_continue_last_turn_stops(self):
        from forge.agents.react import SimpleReActAgent

        agent = SimpleReActAgent(max_turns=3)
        assert agent.should_continue(turn=2, reward=0.0) is False

    def test_format_feedback_success(self):
        from forge.agents.react import SimpleReActAgent
        from forge.core.types import AgentAction, ToolResult

        agent = SimpleReActAgent()
        action = AgentAction(response="test")
        results = [ToolResult(success=True, output="42")]
        feedback = agent.format_feedback(action, results, reward=0.0)
        assert "42" in feedback
        assert "incorrect" in feedback.lower()

    def test_format_feedback_error(self):
        from forge.agents.react import SimpleReActAgent
        from forge.core.types import AgentAction, ToolResult

        agent = SimpleReActAgent()
        action = AgentAction(response="test")
        results = [ToolResult(success=False, error="NameError: x")]
        feedback = agent.format_feedback(action, results, reward=0.0)
        assert "Error: NameError: x" in feedback

    def test_compute_discount(self):
        from forge.agents.react import SimpleReActAgent

        agent = SimpleReActAgent(turn_discount=0.9)
        assert agent.compute_discount(1) == 1.0
        assert abs(agent.compute_discount(2) - 0.9) < 1e-6
        assert abs(agent.compute_discount(3) - 0.81) < 1e-6

    def test_protocol_compliance(self):
        from forge.agents.react import SimpleReActAgent
        from forge.core.protocols import AgentLogic

        agent = SimpleReActAgent()
        assert isinstance(agent, AgentLogic)


# ======================================================================
# ChatTemplate
# ======================================================================


class TestChatTemplate:
    def test_simple_chatml(self):
        from forge.core.chat_template import CHATML

        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = CHATML.apply_with_generation_prompt(messages)
        assert "<|im_start|>user" in result
        assert "Hello" in result
        assert "<|im_start|>assistant" in result

    def test_simple_chatml_no_gen_prompt(self):
        from forge.core.chat_template import CHATML

        messages = [{"role": "user", "content": "Hi"}]
        result = CHATML.apply(messages)
        assert "<|im_start|>assistant" not in result

    def test_llama_style(self):
        from forge.core.chat_template import LLAMA_STYLE

        messages = [{"role": "user", "content": "Test"}]
        result = LLAMA_STYLE.apply_with_generation_prompt(messages)
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_simple_template_system_prompt(self):
        from forge.core.chat_template import SimpleChatTemplate

        tmpl = SimpleChatTemplate(system_prompt="You are helpful.")
        messages = [{"role": "user", "content": "Hi"}]
        result = tmpl.apply(messages)
        assert "You are helpful." in result

    def test_simple_template_no_duplicate_system(self):
        from forge.core.chat_template import SimpleChatTemplate

        tmpl = SimpleChatTemplate(system_prompt="System msg")
        messages = [
            {"role": "system", "content": "Already here"},
            {"role": "user", "content": "Hi"},
        ]
        result = tmpl.apply(messages)
        assert result.count("System msg") == 0
        assert "Already here" in result

    def test_protocol_compliance(self):
        from forge.core.chat_template import CHATML, ChatTemplate

        assert isinstance(CHATML, ChatTemplate)


# ======================================================================
# Core data classes
# ======================================================================


class TestCoreDataClasses:
    def test_generation_result_defaults(self):
        from forge.core.types import GenerationResult

        r = GenerationResult()
        assert r.text == ""
        assert r.token_ids == []
        assert r.logprobs == []
        assert r.version == -1

    def test_tool_call(self):
        from forge.core.types import ToolCall

        tc = ToolCall(type="code_execution", content="print(1)")
        assert tc.type == "code_execution"
        assert tc.metadata == {}

    def test_agent_action_defaults(self):
        from forge.core.types import AgentAction

        a = AgentAction()
        assert a.response == ""
        assert a.tool_calls == []
        assert a.done is False

    def test_tool_result(self):
        from forge.core.types import ToolResult

        tr = ToolResult(success=True, output="ok")
        assert tr.success is True
        assert tr.tool_call is None


# ======================================================================
# DataProvider protocol
# ======================================================================


class TestDataProviderProtocol:
    def test_protocol_structural(self):
        from forge.core.protocols import DataProvider

        class MyProvider:
            def get_batch(self):
                return [{"prompt": "hi"}]

            def reset(self):
                pass

            def __len__(self):
                return 10

        p = MyProvider()
        assert isinstance(p, DataProvider)

    def test_protocol_missing_method(self):
        from forge.core.protocols import DataProvider

        class Incomplete:
            def get_batch(self):
                return []

        assert not isinstance(Incomplete(), DataProvider)


# ======================================================================
# RL Loss functions (pure math, CPU tensors, no GPU needed)
# ======================================================================


class TestRLLossOps:
    def test_masked_mean(self):
        import torch

        from forge.rl.loss.ops import masked_mean

        values = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        result = masked_mean(values, mask)
        assert abs(result.item() - 1.5) < 1e-6

    def test_masked_mean_with_scale(self):
        import torch

        from forge.rl.loss.ops import masked_mean

        values = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        result = masked_mean(values, mask, loss_scale=torch.tensor(4.0))
        assert abs(result.item() - 0.75) < 1e-6

    def test_create_shifted_targets(self):
        import torch

        from forge.rl.loss.ops import create_shifted_targets

        ids = torch.tensor([[10, 20, 30, 40]])
        targets = create_shifted_targets(ids)
        assert targets[0, 0].item() == 20
        assert targets[0, 1].item() == 30
        assert targets[0, 2].item() == 40
        assert targets[0, 3].item() == -100

    def test_create_shifted_targets_with_mask(self):
        import torch

        from forge.rl.loss.ops import create_shifted_targets

        ids = torch.tensor([[10, 20, 30, 40]])
        mask = torch.tensor([[0, 1, 1, 0]])
        targets = create_shifted_targets(ids, loss_mask=mask)
        assert targets[0, 0].item() == -100
        assert targets[0, 1].item() == 30
        assert targets[0, 3].item() == -100

    def test_compute_ratio_on_policy(self):
        import torch

        from forge.rl.loss.ops import compute_ratio

        logprobs = torch.tensor([[- 1.0, -2.0, -1.5]])
        gen_logprobs = torch.tensor([[-1.0, -2.0, -1.5]])
        mask = torch.ones(1, 3)
        ratio, log_ratio, metrics = compute_ratio(logprobs, gen_logprobs, mask)
        assert torch.allclose(ratio, torch.ones(1, 3), atol=1e-6)
        assert torch.allclose(log_ratio, torch.zeros(1, 3), atol=1e-6)

    def test_compute_kl_k3_zero_when_same(self):
        import torch

        from forge.rl.loss.ops import compute_kl

        lp = torch.tensor([[-1.0, -2.0]])
        kl, metrics = compute_kl(lp, lp, torch.ones(1, 2), kl_type="k3")
        assert torch.allclose(kl, torch.zeros(1, 2), atol=1e-6)

    def test_pg_ppo_clip_no_clip_when_on_policy(self):
        import torch

        from forge.rl.loss.ops import pg_ppo_clip

        ratio = torch.ones(1, 3)
        advantages = torch.tensor([[0.5, -0.3, 0.1]])
        mask = torch.ones(1, 3)
        loss, metrics = pg_ppo_clip(ratio, advantages, mask)
        expected = -ratio * advantages
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_aggregate_fixed_horizon(self):
        import torch

        from forge.rl.loss.ops import aggregate

        loss = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        result, _ = aggregate(loss, mask, agg_type="fixed_horizon")
        assert abs(result.item() - 3.0 / 4.0) < 1e-6

    def test_aggregate_token_mean(self):
        import torch

        from forge.rl.loss.ops import aggregate

        loss = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        result, _ = aggregate(loss, mask, agg_type="token_mean")
        assert abs(result.item() - 1.5) < 1e-6


class TestGRPOLoss:
    def test_grpo_runs(self):
        import torch

        from forge.rl.loss.grpo import GRPOLoss

        B, S, V = 2, 8, 32
        logits = torch.randn(B, S, V, requires_grad=True)
        target_ids = torch.randint(0, V, (B, S))
        advantages = torch.randn(B, S)
        gen_logprobs = torch.randn(B, S)
        loss_mask = torch.ones(B, S)

        loss_fn = GRPOLoss(beta=0.0)
        output = loss_fn(logits, target_ids, advantages, gen_logprobs, loss_mask)
        assert output.loss.shape == ()
        assert output.loss.requires_grad
        assert len(output.metrics) > 0

    def test_grpo_with_kl(self):
        import torch

        from forge.rl.loss.grpo import GRPOLoss

        B, S, V = 2, 8, 32
        logits = torch.randn(B, S, V)
        target_ids = torch.randint(0, V, (B, S))
        advantages = torch.randn(B, S)
        gen_logprobs = torch.randn(B, S)
        ref_logprobs = torch.randn(B, S)
        loss_mask = torch.ones(B, S)

        loss_fn = GRPOLoss(beta=0.1)
        output = loss_fn(
            logits, target_ids, advantages, gen_logprobs, loss_mask,
            ref_logprobs=ref_logprobs,
        )
        assert output.loss.shape == ()
        assert any("kl_ref" in m.key for m in output.metrics)

    def test_grpo_beta_zero_no_ref_needed(self):
        import torch

        from forge.rl.loss.grpo import GRPOLoss

        loss_fn = GRPOLoss(beta=0.0)
        output = loss_fn(
            torch.randn(1, 4, 16),
            torch.randint(0, 16, (1, 4)),
            torch.randn(1, 4),
            torch.randn(1, 4),
            torch.ones(1, 4),
        )
        assert output.loss.shape == ()


class TestDAPOLoss:
    def test_dapo_runs(self):
        import torch

        from forge.rl.loss.dapo import DAPOLoss

        B, S, V = 2, 8, 32
        logits = torch.randn(B, S, V, requires_grad=True)
        target_ids = torch.randint(0, V, (B, S))
        advantages = torch.randn(B, S)
        gen_logprobs = torch.randn(B, S)
        loss_mask = torch.ones(B, S)

        loss_fn = DAPOLoss()
        output = loss_fn(logits, target_ids, advantages, gen_logprobs, loss_mask)
        assert output.loss.shape == ()
        assert output.loss.requires_grad
        assert any("dual_clip" in m.key for m in output.metrics)
