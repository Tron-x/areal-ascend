"""RolloutProducer -- CPU-only actor that drives rollout and feeds ReplayBuffer.

This actor is the "producer" half of the async rollout/train pipeline.
It runs on CPU (no GPU needed) and orchestrates:

1. Fetch data from DataProvider
2. Run generation via AgentActor (multi-turn) or ModelProxy/Generator (single-turn)
3. Compute rewards via RewardActor
4. Push completed rollout batches to ReplayBuffer

The "consumer" half is the TrainerActor calling ``train_on_buffered_batch``.
"""

from __future__ import annotations

import logging
import time

from monarch.actor import endpoint

from forge.actors.base import ForgeActor

logger = logging.getLogger(__name__)


class RolloutProducer(ForgeActor):
    """Drives rollout generation and pushes results to ReplayBuffer.

    Runs on CPU -- all GPU work happens on Generator (via AgentActor
    or ModelProxy) and RewardActor replicas.

    Args:
        generator: Generator ServiceInterface or ActorMesh.
        reward: RewardActor ServiceInterface or ActorMesh.
        agent: AgentActor ServiceInterface (for multi-turn episodes).
        replay_buffer: ReplayBuffer ActorMesh.
        n_samples: Number of completions per prompt (for group generation).

    Usage::

        producer = await RolloutProducer.options(procs=1).as_actor(
            generator=gen_service,
            reward=reward_service,
            agent=agent_service,
            replay_buffer=buffer,
        )
        await producer.produce_step.call_one(data_items, step=0)
    """

    procs = 1
    with_gpus = False

    def __init__(
        self,
        generator=None,
        reward=None,
        agent=None,
        replay_buffer=None,
        n_samples: int = 1,
    ):
        self._generator = generator
        self._reward = reward
        self._agent = agent
        self._buffer = replay_buffer
        self._n_samples = n_samples
        self._total_produced = 0
        self._total_time = 0.0

    @endpoint
    async def produce_step(
        self,
        data_items: list[dict],
        step: int = -1,
        version: int = -1,
    ) -> dict:
        """Run rollout on a batch of data items and push to ReplayBuffer.

        Args:
            data_items: List of raw data dicts from DataProvider.
            step: Current global training step (for buffer tagging).
            version: Current policy version (for staleness tracking).

        Returns:
            Production statistics.
        """
        t0 = time.monotonic()
        results = []

        for item in data_items:
            for _ in range(self._n_samples):
                result = await self._generate_episode(item)
                if result is not None:
                    results.append(result)

        if results and self._buffer is not None:
            await self._push_to_buffer(results, version=version, step=step)

        elapsed = time.monotonic() - t0
        self._total_produced += len(results)
        self._total_time += elapsed

        return {
            "produced": len(results),
            "step": step,
            "elapsed": elapsed,
            "total_produced": self._total_produced,
        }

    @endpoint
    def get_stats(self) -> dict:
        avg = (
            (self._total_time / self._total_produced) if self._total_produced > 0 else 0
        )
        return {
            "total_produced": self._total_produced,
            "total_time": self._total_time,
            "avg_time_per_episode": avg,
        }

    async def _generate_episode(self, data: dict) -> dict | None:
        """Generate a single rollout episode.

        Uses AgentActor (multi-turn) if available, otherwise falls back
        to direct Generator call (single-turn).
        """
        try:
            if self._agent is not None:
                return await self._generate_via_agent(data)
            return await self._generate_via_generator(data)
        except Exception as e:
            logger.warning(f"Episode generation failed: {e}")
            return None

    async def _generate_via_agent(self, data: dict) -> dict:
        """Route through AgentActor for multi-turn episodes."""
        ep = self._agent.run_episode
        if hasattr(ep, "route"):
            return await ep.route(data)
        return await ep.call_one(data)

    async def _generate_via_generator(self, data: dict) -> dict | None:
        """Direct single-turn generation + reward computation."""
        prompt = data.get("prompt", "")
        if not prompt:
            messages = data.get("messages", [])
            if messages:
                prompt = messages[-1].get("content", "")
        if not prompt:
            return None

        gen_ep = self._generator.generate
        if hasattr(gen_ep, "route"):
            gen_results = await gen_ep.route(prompt)
        else:
            gen_results = await gen_ep.call_one(prompt)

        if isinstance(gen_results, list) and gen_results:
            gen_result = gen_results[0]
        elif isinstance(gen_results, dict):
            gen_result = gen_results
        else:
            return None

        text = gen_result.get("text", "")
        token_ids = gen_result.get("token_ids", [])
        logprobs = gen_result.get("logprobs", [])
        version = gen_result.get("generator_version", -1)

        if not isinstance(logprobs, list):
            logprobs = [0.0] * len(token_ids)

        reward = 0.0
        if self._reward is not None:
            reward = await self._compute_reward(prompt, text, data)

        return {
            "packed_input_ids": token_ids,
            "logprobs": logprobs,
            "loss_mask": [1] * len(token_ids),
            "versions": [version] * len(token_ids),
            "rewards": reward,
            "seq_len": len(token_ids),
            "prompt": prompt,
            "completion": text,
        }

    async def _compute_reward(self, prompt: str, completion: str, data: dict) -> float:
        ep = self._reward.compute_reward
        kwargs = {
            "prompt": prompt,
            "completion": completion,
            "task_data": {
                k: v for k, v in data.items() if k not in ("messages", "prompt")
            },
        }
        try:
            if hasattr(ep, "route"):
                return await ep.route(**kwargs)
            return await ep.call_one(**kwargs)
        except Exception as e:
            logger.warning(f"Reward computation failed: {e}")
            return 0.0

    async def _push_to_buffer(
        self, results: list[dict], version: int, step: int
    ) -> None:
        """Push completed rollout results to ReplayBuffer."""
        ep = self._buffer.add_batch
        if hasattr(ep, "call_one"):
            await ep.call_one(results, version=version, step=step)
        else:
            await ep.call(results, version=version, step=step)
