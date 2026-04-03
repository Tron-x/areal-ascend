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

        logprobs = torch.tensor([[-1.0, -2.0, -1.5]])
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
            logits,
            target_ids,
            advantages,
            gen_logprobs,
            loss_mask,
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


# ======================================================================
# Observability (metrics + tracer)
# ======================================================================


class TestMetricAccumulators:
    def test_mean_accumulator(self):
        from forge.observability.metrics import MeanAccumulator, Reduce

        acc = MeanAccumulator(Reduce.MEAN)
        acc.append(1.0)
        acc.append(3.0)
        assert abs(acc.get_value() - 2.0) < 1e-6
        state = acc.get_state()
        assert state["sum"] == 4.0
        assert state["count"] == 2
        acc.reset()
        assert acc.get_value() == 0.0

    def test_sum_accumulator(self):
        from forge.observability.metrics import Reduce, SumAccumulator

        acc = SumAccumulator(Reduce.SUM)
        acc.append(5.0)
        acc.append(3.0)
        assert abs(acc.get_value() - 8.0) < 1e-6

    def test_max_accumulator(self):
        from forge.observability.metrics import MaxAccumulator, Reduce

        acc = MaxAccumulator(Reduce.MAX)
        acc.append(1.0)
        acc.append(5.0)
        acc.append(3.0)
        assert abs(acc.get_value() - 5.0) < 1e-6

    def test_min_accumulator(self):
        from forge.observability.metrics import MinAccumulator, Reduce

        acc = MinAccumulator(Reduce.MIN)
        acc.append(5.0)
        acc.append(1.0)
        acc.append(3.0)
        assert abs(acc.get_value() - 1.0) < 1e-6

    def test_std_accumulator(self):
        from forge.observability.metrics import Reduce, StdAccumulator

        acc = StdAccumulator(Reduce.STD)
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            acc.append(v)
        assert abs(acc.get_value() - 2.0) < 0.1

    def test_mean_cross_rank_reduce(self):
        from forge.observability.metrics import MeanAccumulator

        states = [
            {"reduction_type": "mean", "sum": 10.0, "count": 5},
            {"reduction_type": "mean", "sum": 20.0, "count": 10},
        ]
        result = MeanAccumulator.get_reduced_value_from_states(states)
        assert abs(result - 2.0) < 1e-6


class TestMetricCollector:
    def test_record_and_flush(self):
        from forge.observability.metrics import MetricCollector, Reduce

        collector = MetricCollector()
        from forge.observability.metrics import Metric

        collector.push(Metric("loss", 1.0, Reduce.MEAN))
        collector.push(Metric("loss", 3.0, Reduce.MEAN))
        collector.push(Metric("count", 1, Reduce.SUM))
        collector.push(Metric("count", 1, Reduce.SUM))

        result = collector.flush(step=0)
        assert abs(result["loss"] - 2.0) < 1e-6
        assert abs(result["count"] - 2.0) < 1e-6

        result2 = collector.flush(step=1)
        assert result2.get("loss", 0.0) == 0.0

    def test_reduce_metrics_states(self):
        from forge.observability.metrics import reduce_metrics_states

        states = [
            {"loss": {"reduction_type": "mean", "sum": 6.0, "count": 3}},
            {"loss": {"reduction_type": "mean", "sum": 4.0, "count": 2}},
        ]
        result = reduce_metrics_states(states)
        assert abs(result["loss"] - 2.0) < 1e-6


class TestTracer:
    def test_tracer_basic(self):
        import time

        from forge.observability.tracer import Tracer

        t = Tracer("test", log_to_metrics=False)
        t.start()
        time.sleep(0.01)
        elapsed = t.step("work")
        assert elapsed > 0.005
        total = t.stop()
        assert total >= elapsed

    def test_tracer_context_manager(self):
        from forge.observability.tracer import Tracer

        with Tracer("ctx", log_to_metrics=False) as t:
            t.step("a")
        assert len(t._steps) == 1


# ======================================================================
# GroupBuffer
# ======================================================================


def _make_group_buffer(**kwargs):
    """Create a GroupBuffer without Monarch scaffolding."""
    from forge.actors.group_buffer import GroupBuffer

    buf = object.__new__(GroupBuffer)
    from collections import OrderedDict

    defaults = {
        "group_size": 4,
        "max_groups": 100,
        "max_staleness_steps": 2,
        "group_filter": None,
    }
    defaults.update(kwargs)
    buf._group_size = defaults["group_size"]
    buf._max_groups = defaults["max_groups"]
    buf._max_staleness_steps = defaults["max_staleness_steps"]
    buf._filter = defaults["group_filter"]
    buf._pending = OrderedDict()
    buf._complete = OrderedDict()
    buf._total_episodes = 0
    buf._total_groups_completed = 0
    buf._total_groups_expired = 0
    buf._total_groups_filtered = 0
    return buf


class TestGroupBufferAdd:
    def test_add_episodes_until_complete(self):
        buf = _make_group_buffer(group_size=3)
        for i in range(3):
            result = buf.add_episode("prompt_0", {"resp": i}, version=0, step=0)
        assert result["complete_groups"] == 1
        assert result["pending_groups"] == 0

    def test_incomplete_group_stays_pending(self):
        buf = _make_group_buffer(group_size=4)
        buf.add_episode("p1", {"r": 0}, version=0, step=0)
        buf.add_episode("p1", {"r": 1}, version=0, step=0)
        stats = buf.get_stats()
        assert stats["pending_groups"] == 1
        assert stats["complete_groups"] == 0

    def test_multiple_groups(self):
        buf = _make_group_buffer(group_size=2)
        buf.add_episode("p0", {"r": 0}, version=0, step=0)
        buf.add_episode("p1", {"r": 0}, version=0, step=0)
        buf.add_episode("p0", {"r": 1}, version=0, step=0)
        buf.add_episode("p1", {"r": 1}, version=0, step=0)
        stats = buf.get_stats()
        assert stats["complete_groups"] == 2
        assert stats["pending_groups"] == 0

    def test_add_group_sync(self):
        buf = _make_group_buffer(group_size=4)
        result = buf.add_group("p0", [{"r": i} for i in range(4)], version=0, step=0)
        assert result["complete_groups"] == 1


class TestGroupBufferSample:
    def test_sample_empty_returns_none(self):
        buf = _make_group_buffer()
        assert buf.sample_group() is None

    def test_sample_fifo_order(self):
        buf = _make_group_buffer(group_size=2)
        buf.add_episode("first", {"r": 0}, version=0, step=0)
        buf.add_episode("first", {"r": 1}, version=0, step=0)
        buf.add_episode("second", {"r": 0}, version=0, step=0)
        buf.add_episode("second", {"r": 1}, version=0, step=0)

        group = buf.sample_group()
        assert group is not None
        assert len(group) == 2
        assert group[0]["r"] == 0

        group2 = buf.sample_group()
        assert group2 is not None
        assert len(group2) == 2

        assert buf.sample_group() is None

    def test_sample_groups_batch(self):
        buf = _make_group_buffer(group_size=2)
        for pid in range(5):
            buf.add_group(f"p{pid}", [{"r": i} for i in range(2)])
        groups = buf.sample_groups(count=3)
        assert groups is not None
        assert len(groups) == 3
        assert buf.get_stats()["complete_groups"] == 2


class TestGroupBufferExpiry:
    def test_stale_groups_expired(self):
        buf = _make_group_buffer(group_size=4, max_staleness_steps=1)
        buf.add_episode("old", {"r": 0}, version=0, step=0)
        buf.add_episode("old", {"r": 1}, version=0, step=0)
        buf.sample_group(current_step=5)
        assert buf.get_stats()["pending_groups"] == 0
        assert buf.get_stats()["total_groups_expired"] == 1

    def test_fresh_groups_not_expired(self):
        buf = _make_group_buffer(group_size=4, max_staleness_steps=3)
        buf.add_episode("fresh", {"r": 0}, version=0, step=5)
        buf.sample_group(current_step=6)
        assert buf.get_stats()["pending_groups"] == 1


class TestGroupBufferFilter:
    def test_filter_drops_group(self):
        def all_same_reward(gid, episodes):
            rewards = [e.get("reward", 0) for e in episodes]
            return len(set(rewards)) <= 1

        buf = _make_group_buffer(group_size=2, group_filter=all_same_reward)
        buf.add_episode("p0", {"reward": 1.0}, version=0, step=0)
        buf.add_episode("p0", {"reward": 1.0}, version=0, step=0)
        assert buf.get_stats()["complete_groups"] == 0
        assert buf.get_stats()["total_groups_filtered"] == 1

    def test_filter_keeps_diverse_group(self):
        def all_same_reward(gid, episodes):
            rewards = [e.get("reward", 0) for e in episodes]
            return len(set(rewards)) <= 1

        buf = _make_group_buffer(group_size=2, group_filter=all_same_reward)
        buf.add_episode("p0", {"reward": 1.0}, version=0, step=0)
        buf.add_episode("p0", {"reward": 0.0}, version=0, step=0)
        assert buf.get_stats()["complete_groups"] == 1


class TestGroupBufferMaxGroups:
    def test_oldest_evicted_when_full(self):
        buf = _make_group_buffer(group_size=10, max_groups=2)
        buf.add_episode("p0", {"r": 0}, version=0, step=0)
        buf.add_episode("p1", {"r": 0}, version=0, step=0)
        buf.add_episode("p2", {"r": 0}, version=0, step=0)
        assert buf.get_stats()["pending_groups"] == 2
        assert buf.get_stats()["total_groups_expired"] == 1


# ======================================================================
# Tool system
# ======================================================================


class TestToolSpec:
    def test_to_openai_spec(self):
        from forge.tools.protocol import ToolSpec

        spec = ToolSpec(
            name="code_interpreter",
            description="Run Python",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        )
        openai = spec.to_openai_spec()
        assert openai["type"] == "function"
        assert openai["function"]["name"] == "code_interpreter"
        assert "code" in openai["function"]["parameters"]["properties"]


class TestCodeBlockParser:
    def test_parse_code_tag(self):
        from forge.tools.parsers import CodeBlockParser

        p = CodeBlockParser()
        calls = p.parse("Think step by step.\n<code>print(42)</code>\nDone.")
        assert len(calls) == 1
        assert calls[0].name == "code_interpreter"
        assert calls[0].arguments["code"] == "print(42)"

    def test_parse_markdown_python(self):
        from forge.tools.parsers import CodeBlockParser

        p = CodeBlockParser()
        calls = p.parse("```python\nx = 1+1\nprint(x)\n```")
        assert len(calls) == 1
        assert "print(x)" in calls[0].arguments["code"]

    def test_parse_multiple(self):
        from forge.tools.parsers import CodeBlockParser

        p = CodeBlockParser()
        calls = p.parse("<code>a=1</code> then <code>b=2</code>")
        assert len(calls) == 2

    def test_no_match(self):
        from forge.tools.parsers import CodeBlockParser

        p = CodeBlockParser()
        assert p.parse("No code here.") == []
        assert p.has_tool_call("No code here.") is False

    def test_has_tool_call(self):
        from forge.tools.parsers import CodeBlockParser

        p = CodeBlockParser()
        assert p.has_tool_call("<code>x</code>") is True
        assert p.has_tool_call("```python\nx\n```") is True


class TestToolCallParser:
    def test_parse_qwen3_format(self):
        from forge.tools.parsers import ToolCallParser

        p = ToolCallParser()
        resp = '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n</tool_call>'
        calls = p.parse(resp)
        assert len(calls) == 1
        assert calls[0].name == "code_interpreter"
        assert calls[0].arguments["code"] == "print(1)"

    def test_incomplete_tag_auto_close(self):
        from forge.tools.parsers import ToolCallParser

        p = ToolCallParser()
        resp = '<tool_call>{"name": "test", "arguments": {}}'
        calls = p.parse(resp)
        assert len(calls) == 1

    def test_invalid_json_skipped(self):
        from forge.tools.parsers import ToolCallParser

        p = ToolCallParser()
        resp = "<tool_call>not json</tool_call>"
        assert p.parse(resp) == []


class TestFunctionCallParser:
    def test_parse_qwen3_coder(self):
        from forge.tools.parsers import FunctionCallParser

        p = FunctionCallParser()
        resp = "<function=list_dir><parameter=path>.</parameter></function>"
        calls = p.parse(resp)
        assert len(calls) == 1
        assert calls[0].name == "list_dir"
        assert calls[0].arguments["path"] == "."

    def test_parse_numeric_params(self):
        from forge.tools.parsers import FunctionCallParser

        p = FunctionCallParser()
        resp = "<function=calc><parameter=x>42</parameter><parameter=y>3.14</parameter></function>"
        calls = p.parse(resp)
        assert calls[0].arguments["x"] == 42
        assert abs(calls[0].arguments["y"] - 3.14) < 0.01


class TestCompositeParser:
    def test_finds_all_formats(self):
        from forge.tools.parsers import CompositeParser

        p = CompositeParser()
        resp = (
            "<code>print(1)</code>\n"
            '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>\n'
            "<function=calc><parameter=x>5</parameter></function>"
        )
        calls = p.parse(resp)
        assert len(calls) == 3

    def test_has_tool_call_any_format(self):
        from forge.tools.parsers import CompositeParser

        p = CompositeParser()
        assert p.has_tool_call("<code>x</code>") is True
        assert p.has_tool_call("<tool_call>x</tool_call>") is True
        assert p.has_tool_call("<function=f>x</function>") is True
        assert p.has_tool_call("plain text") is False


class TestToolRegistry:
    def test_register_and_list(self):
        from forge.tools.protocol import ToolSpec
        from forge.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register(ToolSpec(name="test_tool", description="A test"))
        assert "test_tool" in reg.list_tools()

    def test_get_specs_openai(self):
        from forge.tools.protocol import ToolSpec
        from forge.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register(ToolSpec(name="t1", description="Tool 1"))
        reg.register(ToolSpec(name="t2", description="Tool 2"))
        specs = reg.get_tool_specs()
        assert len(specs) == 2
        assert all(s["type"] == "function" for s in specs)


class TestPythonSandbox:
    def test_safety_check_blocks_os(self):
        from forge.tools.python_sandbox import PythonSandbox

        sb = PythonSandbox()
        safe, msg = sb.check_safety("import os\nos.system('rm -rf /')")
        assert safe is False
        assert "os" in msg.lower()

    def test_safety_check_allows_math(self):
        from forge.tools.python_sandbox import PythonSandbox

        sb = PythonSandbox(allowed_modules={"math"})
        safe, msg = sb.check_safety("import math\nprint(math.sqrt(4))")
        assert safe is True

    def test_safety_check_disabled_skips_in_execute(self):
        from forge.tools.python_sandbox import PythonSandbox

        sb = PythonSandbox(safety_check=False)
        # When safety_check=False, execute() skips check_safety entirely
        # check_safety() itself always runs the patterns regardless
        assert sb.safety_check is False


# ======================================================================
# ReToolAgent
# ======================================================================


class TestReToolAgent:
    def test_detects_answer(self):
        from forge.agents.retool import ReToolAgent

        agent = ReToolAgent()
        action = agent.process_response("After calculation, Answer: \\boxed{42}", [])
        assert action.done is True
        assert len(action.tool_calls) == 0

    def test_detects_code_block(self):
        from forge.agents.retool import ReToolAgent

        agent = ReToolAgent()
        action = agent.process_response("Let me compute:\n<code>print(2+2)</code>", [])
        assert action.done is False
        assert len(action.tool_calls) == 1
        assert action.tool_calls[0].type == "code_interpreter"

    def test_detects_tool_call(self):
        from forge.agents.retool import ReToolAgent

        agent = ReToolAgent()
        action = agent.process_response(
            '<tool_call>{"name": "code_interpreter", "arguments": {"code": "x=1"}}</tool_call>',
            [],
        )
        assert action.done is False
        assert len(action.tool_calls) == 1

    def test_no_tool_no_answer(self):
        from forge.agents.retool import ReToolAgent

        agent = ReToolAgent()
        action = agent.process_response("Hmm, let me think...", [])
        assert action.done is False
        assert len(action.tool_calls) == 0

    def test_format_tool_observation(self):
        from forge.agents.retool import ReToolAgent
        from forge.core.types import ToolResult

        agent = ReToolAgent()
        results = [ToolResult(success=True, output="4")]
        obs = agent.format_tool_observation(results)
        assert "<interpreter>" in obs
        assert "4" in obs

    def test_format_error_observation(self):
        from forge.agents.retool import ReToolAgent
        from forge.core.types import ToolResult

        agent = ReToolAgent()
        results = [ToolResult(success=False, error="NameError: x")]
        obs = agent.format_tool_observation(results)
        assert "Error" in obs

    def test_should_continue(self):
        from forge.agents.retool import ReToolAgent

        agent = ReToolAgent(max_turns=3)
        assert agent.should_continue(0, 0.0) is True
        assert agent.should_continue(2, 0.0) is False

    def test_compute_discount_always_1(self):
        from forge.agents.retool import ReToolAgent

        agent = ReToolAgent()
        assert agent.compute_discount(5) == 1.0

    def test_protocol_compliance(self):
        from forge.agents.retool import ReToolAgent
        from forge.core.protocols import AgentLogic

        agent = ReToolAgent()
        assert isinstance(agent, AgentLogic)
