"""MonarchAgentWorkflow -- bridge between Forge AgentActor and AReaL PPOTrainer.

Implements AReaL's ``RolloutWorkflow`` interface by delegating to
``AgentActor.run_episode`` via Monarch RPC. This allows AReaL's
PPOTrainer to use Forge's agentic rollout as if it were a native workflow.
"""

from __future__ import annotations

from typing import Any


class MonarchAgentWorkflow:
    """RolloutWorkflow drop-in that delegates to AgentActor via Monarch RPC.

    Usage::

        workflow = MonarchAgentWorkflow(agent_actor)
        result = await workflow.arun_episode(engine, data)
    """

    def __init__(self, agent_actor, chat_template=None, **kwargs):
        self._agent = agent_actor
        self._chat_template = chat_template
        self._extra_kwargs = kwargs
        self._setup_done = False

    async def setup(
        self,
        generator=None,
        reward=None,
        sandbox=None,
        chat_template=None,
    ):
        if not self._setup_done:
            tmpl = chat_template or self._chat_template
            ep = self._agent.setup
            kwargs: dict[str, Any] = dict(
                generator=generator, reward=reward, sandbox=sandbox
            )
            if tmpl is not None:
                kwargs["chat_template"] = tmpl
            if hasattr(ep, "route"):
                await ep.route(**kwargs)
            else:
                await ep.call_one(**kwargs)
            self._setup_done = True

    async def arun_episode(self, engine, data: dict[str, Any]):
        """Delegate to AgentActor and convert result to tensor dict."""
        import torch

        ep = self._agent.run_episode
        if hasattr(ep, "route"):
            res = await ep.route(data)
        else:
            res = await ep.call_one(data)

        tensor_res = {}
        for k, v in res.items():
            if isinstance(v, list):
                if k in ("logprobs", "rewards"):
                    tensor_res[k] = torch.tensor(v, dtype=torch.float32)
                elif k in ("loss_mask", "versions"):
                    tensor_res[k] = torch.tensor(v, dtype=torch.int32)
                elif k == "input_ids":
                    tensor_res[k] = torch.tensor(v, dtype=torch.int32)
                else:
                    tensor_res[k] = v
            elif isinstance(v, (int, float)):
                tensor_res[k] = torch.tensor(v, dtype=torch.float32)
            else:
                tensor_res[k] = v

        return {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in tensor_res.items()
        }
