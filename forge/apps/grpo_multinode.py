"""Multi-node GRPO training: TorchTitan training + remote vLLM inference.

Designed for a 2-node deployment where:
- Node1 (this process): TorchTitan FSDP2 training on 8 NPUs
- Node2 (inference_node.py): vLLM inference on 8 NPUs

Communication:
- HTTP for generation requests (node1 → node2)
- HCCL for weight synchronization (node1 → node2 broadcast)

Usage::

    python -m forge.apps.grpo_multinode \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --inference-addr 192.168.0.23 \\
        --inference-port 8100 \\
        --train-gpus 8 \\
        --total-train-steps 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

import torch
import torch.distributed as dist

_gloo_group = None

logger = logging.getLogger("GRPO-MultiNode")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-node GRPO Training (TorchTitan)")

    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--model-name", default="qwen3", help="TorchTitan model name")
    p.add_argument("--model-flavor", default="1.7B", help="TorchTitan model flavor")

    p.add_argument("--inference-addr", required=True, help="Inference node IP address")
    p.add_argument("--inference-port", type=int, default=8100)

    p.add_argument("--train-gpus", type=int, default=8)
    p.add_argument("--total-train-steps", type=int, default=2)
    p.add_argument("--loss-type", default="grpo", choices=["grpo", "dapo"])

    p.add_argument("--lr", type=float, default=1.7e-5)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=4, help="Samples per prompt")
    p.add_argument("--max-new-tokens", type=int, default=1024)

    p.add_argument(
        "--reward-fn",
        default="areal.reward.gsm8k.gsm8k_reward_fn",
        help="Dotted path to reward function",
    )
    p.add_argument("--dataset", default="openai/gsm8k")

    p.add_argument("--weight-sync-port", type=int, default=29600)
    p.add_argument("--weight-sync-backend", default="hccl")
    p.add_argument("--no-weight-sync", action="store_true")

    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--http-timeout",
        type=int,
        default=600,
        help="Timeout for inference HTTP calls (seconds)",
    )

    return p.parse_args()


class RemoteGenerator:
    """Client for the remote inference node (HTTP API)."""

    def __init__(self, addr: str, port: int, timeout: int = 600):
        self.base_url = f"http://{addr}:{port}"
        self.timeout = timeout

    def health_check(self, retries: int = 30, interval: float = 10.0) -> bool:
        for i in range(retries):
            try:
                req = urllib.request.Request(f"{self.base_url}/health")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get("status") == "ok":
                        logger.info(
                            "Inference node healthy (version=%s)", data.get("version")
                        )
                        return True
            except (urllib.error.URLError, OSError) as e:
                logger.info("Waiting for inference node (%d/%d): %s", i + 1, retries, e)
                time.sleep(interval)
        return False

    def generate_batch(
        self, prompts: list[str], sampling_params: dict | None = None
    ) -> list[dict]:
        payload = {
            "prompts": prompts,
            "sampling_params": sampling_params or {},
        }
        return self._post("/generate_batch", payload)["results"]

    def generate(self, prompt: str, sampling_params: dict | None = None) -> list[dict]:
        payload = {
            "prompt": prompt,
            "sampling_params": sampling_params or {},
        }
        return self._post("/generate", payload)["results"]

    def trigger_weight_sync(self) -> dict:
        return self._post("/weight_sync", {})

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())


def load_reward_fn(fn_path: str):
    """Import a reward function from a dotted module path."""
    module_path, fn_name = fn_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def load_dataset_prompts(dataset_path: str, split: str = "train") -> list[str]:
    """Load prompts from a HuggingFace dataset."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_path, "main", split=split)
        prompts = []
        for item in ds:
            q = item.get("question", item.get("problem", item.get("prompt", "")))
            prompts.append(q)
        return prompts
    except Exception as e:
        logger.warning(
            "Failed to load dataset %s: %s. Using dummy prompts.", dataset_path, e
        )
        return [f"What is {i} + {i * 2}?" for i in range(100)]


def broadcast_weights_hccl(model: torch.nn.Module, group=None):
    """Broadcast all model parameters from rank 0 to all ranks in group."""
    for param in model.parameters():
        dist.broadcast(param.data, src=0, group=group)


def main():
    from forge.bootstraps import ensure_ascend_custom_opp_path

    ensure_ascend_custom_opp_path()

    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_rank0 = rank == 0

    torch.manual_seed(args.seed)

    if is_rank0:
        logger.info("=" * 60)
        logger.info(" Multi-Node GRPO Training (TorchTitan)")
        logger.info(" Model:          %s", args.model)
        logger.info(" Inference node: %s:%d", args.inference_addr, args.inference_port)
        logger.info(
            " Train GPUs:     %d (rank %d/%d)", args.train_gpus, rank, world_size
        )
        logger.info(" Train steps:    %d", args.total_train_steps)
        logger.info(" Loss type:      %s", args.loss_type)
        logger.info("=" * 60)

    generator = None
    reward_fn = None
    all_prompts = None

    if is_rank0:
        generator = RemoteGenerator(
            args.inference_addr, args.inference_port, args.http_timeout
        )
        logger.info("Waiting for inference node to be ready...")
        if not generator.health_check():
            logger.error(
                "Inference node not reachable at %s:%d",
                args.inference_addr,
                args.inference_port,
            )
            sys.exit(1)

        logger.info("Loading reward function: %s", args.reward_fn)
        reward_fn = load_reward_fn(args.reward_fn)

        logger.info("Loading dataset: %s", args.dataset)
        all_prompts = load_dataset_prompts(args.dataset)
        logger.info("Loaded %d prompts", len(all_prompts))

    if is_rank0:
        logger.info("Initializing TorchTitan training engine...")

    from forge.engines.titan.adapter import TitanTrainEngine

    engine = TitanTrainEngine(
        config={
            "model_name": args.model_name,
            "model_flavor": args.model_flavor,
            "hf_model_path": args.model,
            "max_steps": args.total_train_steps,
            "loss_type": args.loss_type,
            "lr": args.lr,
            "dtype": args.dtype,
            "seq_len": args.seq_len,
            "local_batch_size": args.batch_size,
            "dp_shard": -1,
            "tp": 1,
            "pp": 1,
            "loss_config": {"beta": 0.0},
        }
    )

    meta = engine.initialize()
    max_steps = meta["max_steps"]
    if is_rank0:
        logger.info("Training engine initialized: max_steps=%d", max_steps)

    global _gloo_group
    _gloo_group = dist.new_group(backend="gloo")

    sampling_params = {
        "max_tokens": args.max_new_tokens,
        "temperature": 1.0,
        "n": args.n_samples,
    }

    prompt_idx = 0
    for step in range(max_steps):
        step_start = time.time()
        batch = None

        if is_rank0:
            logger.info("[Step %d/%d] Starting rollout...", step, max_steps)

            batch_prompts = []
            for _ in range(args.batch_size):
                batch_prompts.append(all_prompts[prompt_idx % len(all_prompts)])
                prompt_idx += 1

            rollout_start = time.time()
            all_results = generator.generate_batch(batch_prompts, sampling_params)
            rollout_time = time.time() - rollout_start
            logger.info(
                "[Step %d] Rollout: %d completions in %.1fs",
                step,
                len(all_results),
                rollout_time,
            )

            reward_start = time.time()
            episodes = []
            total_reward = 0.0
            for result in all_results:
                prompt = result.get("prompt", "")
                text = result.get("text", "")
                try:
                    r = reward_fn(prompt, text)
                    if isinstance(r, dict):
                        r = r.get("reward", r.get("score", 0.0))
                    r = float(r)
                except Exception:
                    r = 0.0

                total_reward += r
                episodes.append(
                    {
                        "input_ids": result.get("token_ids", []),
                        "loss_mask": [1] * len(result.get("token_ids", [])),
                        "advantages": r,
                        "generator_logprobs": result.get("logprobs", []),
                    }
                )

            avg_reward = total_reward / max(len(episodes), 1)
            reward_time = time.time() - reward_start
            logger.info(
                "[Step %d] Reward: avg=%.4f (%d episodes, %.1fs)",
                step,
                avg_reward,
                len(episodes),
                reward_time,
            )

            from forge.engines.fsdp.batch_adapter import FSDPBatchAdapter

            adapter = FSDPBatchAdapter(max_seq_len=args.seq_len)
            batch = adapter.adapt(episodes)

        dist.barrier(group=_gloo_group)

        if is_rank0:
            import pickle

            batch_bytes = pickle.dumps(batch)
            batch_size_tensor = torch.tensor([len(batch_bytes)], dtype=torch.long)
        else:
            batch_size_tensor = torch.tensor([0], dtype=torch.long)

        dist.broadcast(batch_size_tensor, src=0, group=_gloo_group)

        if is_rank0:
            batch_tensor = torch.frombuffer(bytearray(batch_bytes), dtype=torch.uint8)
        else:
            batch_tensor = torch.empty(batch_size_tensor.item(), dtype=torch.uint8)

        dist.broadcast(batch_tensor, src=0, group=_gloo_group)

        if not is_rank0:
            import pickle

            batch = pickle.loads(batch_tensor.numpy().tobytes())

        train_start = time.time()
        result = engine.train_step(batch, step)
        train_time = time.time() - train_start

        if is_rank0:
            if not args.no_weight_sync:
                sync_start = time.time()
                try:
                    sync_result = generator.trigger_weight_sync()
                    sync_time = time.time() - sync_start
                    logger.info(
                        "[Step %d] Weight sync: %s (%.1fs)",
                        step,
                        sync_result.get("success", "?"),
                        sync_time,
                    )
                except Exception as e:
                    logger.warning("[Step %d] Weight sync failed: %s", step, e)

            step_time = time.time() - step_start
            logger.info(
                "[Step %d/%d] Complete: reward=%.4f, loss=%.4f, time=%.1fs (rollout=%.1f, reward=%.1f, train=%.1f)",
                step,
                max_steps,
                avg_reward,
                result.get("loss", 0),
                step_time,
                rollout_time,
                reward_time,
                train_time,
            )

    engine.shutdown()
    if is_rank0:
        logger.info("Training complete! %d steps finished.", max_steps)


if __name__ == "__main__":
    main()
