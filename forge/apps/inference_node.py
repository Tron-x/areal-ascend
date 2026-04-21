"""Standalone inference node for multi-node Forge deployment.

Runs vLLM inference on a dedicated node and exposes:
- HTTP API for generation requests (from the training node)
- HCCL process group participation for weight synchronization

Architecture::

    Training Node (node1)           Inference Node (node2, this file)
    ┌──────────────────┐           ┌──────────────────────────┐
    │ TorchTitan FSDP  │──HTTP───→│ vLLM (8 NPU)             │
    │ 8 NPU training   │──HCCL───→│ /generate + /weight_sync │
    └──────────────────┘           └──────────────────────────┘

Usage::

    python -m forge.apps.inference_node \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --num-gpus 8 \\
        --port 8100 \\
        --weight-sync-addr 192.168.0.26 \\
        --weight-sync-port 29600 \\
        --weight-sync-world-size 16 \\
        --weight-sync-rank-offset 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

logger = logging.getLogger("InferenceNode")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

_engine = None
_weight_sync_group = None
_weight_sync_meta = None
_generator_version = 0
_lock = asyncio.Lock()


def parse_args():
    p = argparse.ArgumentParser(description="Forge Inference Node")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--port", type=int, default=8100, help="HTTP server port")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--weight-sync-addr", default="", help="Master addr for HCCL weight sync group"
    )
    p.add_argument("--weight-sync-port", type=int, default=29600)
    p.add_argument("--weight-sync-world-size", type=int, default=16)
    p.add_argument(
        "--weight-sync-rank-offset",
        type=int,
        default=8,
        help="Starting rank for inference in the weight sync group",
    )
    p.add_argument("--weight-sync-backend", default="hccl")
    return p.parse_args()


class InferenceEngine:
    """Wraps vLLM LLM for synchronous generation."""

    def __init__(self, args):
        from vllm import LLM, SamplingParams

        logger.info(
            "Initializing vLLM with %d GPUs, model=%s", args.num_gpus, args.model
        )

        self.llm = LLM(
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=args.num_gpus,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            seed=args.seed,
            trust_remote_code=True,
        )
        self.default_params = SamplingParams(
            max_tokens=1024,
            temperature=1.0,
            top_p=1.0,
            logprobs=1,
        )
        logger.info("vLLM initialized successfully")

    def generate(self, prompts: list[str], sampling_params=None) -> list[dict]:
        params = sampling_params or self.default_params
        outputs = self.llm.generate(prompts, params)

        results = []
        for output in outputs:
            for comp in output.outputs:
                token_logprobs = []
                if comp.logprobs:
                    for tid, lp_dict in zip(comp.token_ids, comp.logprobs):
                        token_logprobs.append(
                            lp_dict[tid].logprob if tid in lp_dict else 0.0
                        )
                else:
                    token_logprobs = [0.0] * len(comp.token_ids)

                results.append(
                    {
                        "text": comp.text,
                        "token_ids": list(comp.token_ids),
                        "logprobs": token_logprobs,
                        "finish_reason": comp.finish_reason or "stop",
                        "prompt": output.prompt,
                        "prompt_ids": list(output.prompt_token_ids or []),
                        "generator_version": _generator_version,
                    }
                )
        return results

    def get_model(self):
        return self.llm.llm_engine.model_executor.driver_worker.model_runner.model


class WeightSyncReceiver:
    """Receives weight updates from training node via HCCL broadcast."""

    def __init__(self, engine: InferenceEngine, args):
        self.engine = engine
        self.args = args
        self.group = None
        self.meta = None
        self._initialized = False

    def initialize(self):
        if not self.args.weight_sync_addr:
            logger.info("Weight sync disabled (no --weight-sync-addr)")
            return

        rank_offset = self.args.weight_sync_rank_offset
        world_size = self.args.weight_sync_world_size
        backend = self.args.weight_sync_backend

        init_method = f"tcp://{self.args.weight_sync_addr}:{self.args.weight_sync_port}"

        logger.info(
            "Initializing HCCL weight sync group: init=%s, world=%d, rank_offset=%d, backend=%s",
            init_method,
            world_size,
            rank_offset,
            backend,
        )

        # Creating the TCPStore here registers us as a client of the
        # weight-sync rendezvous; we don't need to keep a reference -- the
        # matching ``new_group`` below uses the global store registered in
        # ``init_process_group``.
        _ = torch.distributed.TCPStore(
            host_name=self.args.weight_sync_addr,
            port=self.args.weight_sync_port,
            world_size=world_size,
            is_master=False,
            timeout=torch.distributed.default_pg_timeout,
        )

        self.group = torch.distributed.new_group(
            ranks=list(range(world_size)),
            backend=backend,
        )

        self._initialized = True
        logger.info("HCCL weight sync group initialized")

    def receive_weights(self) -> dict:
        """Receive broadcasted weights from training ranks."""
        global _generator_version

        if not self._initialized:
            return {"success": False, "message": "Weight sync not initialized"}

        model = self.engine.get_model()
        try:
            for name, param in model.named_parameters():
                torch.distributed.broadcast(param.data, src=0, group=self.group)

            _generator_version += 1
            logger.info("Weight update received, version=%d", _generator_version)
            return {"success": True, "version": _generator_version}
        except Exception as e:
            logger.error("Weight sync failed: %s", e)
            return {"success": False, "message": str(e)}


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for inference requests from training node."""

    engine: InferenceEngine = None
    weight_sync: WeightSyncReceiver = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        payload = json.loads(body) if body else {}

        try:
            if self.path == "/generate":
                result = self._handle_generate(payload)
            elif self.path == "/generate_batch":
                result = self._handle_generate_batch(payload)
            elif self.path == "/weight_sync":
                result = self._handle_weight_sync(payload)
            elif self.path == "/health":
                result = {"status": "ok", "version": _generator_version}
            else:
                self._send_error(404, f"Unknown endpoint: {self.path}")
                return
            self._send_json(result)
        except Exception as e:
            logger.error("Request failed: %s", e, exc_info=True)
            self._send_error(500, str(e))

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "version": _generator_version})
        else:
            self._send_error(404, f"Unknown endpoint: {self.path}")

    def _handle_generate(self, payload: dict) -> dict:
        from vllm import SamplingParams

        prompt = payload.get("prompt", "")
        sp = payload.get("sampling_params", {})
        params = SamplingParams(
            max_tokens=sp.get("max_tokens", 1024),
            temperature=sp.get("temperature", 1.0),
            top_p=sp.get("top_p", 1.0),
            n=sp.get("n", 1),
            logprobs=1,
        )
        results = self.engine.generate([prompt], params)
        return {"results": results}

    def _handle_generate_batch(self, payload: dict) -> dict:
        from vllm import SamplingParams

        prompts = payload.get("prompts", [])
        sp = payload.get("sampling_params", {})
        params = SamplingParams(
            max_tokens=sp.get("max_tokens", 1024),
            temperature=sp.get("temperature", 1.0),
            top_p=sp.get("top_p", 1.0),
            n=sp.get("n", 1),
            logprobs=1,
        )
        results = self.engine.generate(prompts, params)
        return {"results": results}

    def _handle_weight_sync(self, payload: dict) -> dict:
        if self.weight_sync is None:
            return {"success": False, "message": "Weight sync not configured"}
        return self.weight_sync.receive_weights()

    def _send_json(self, data: dict):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str):
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        logger.debug(format, *args)


def main():
    from forge.bootstraps import ensure_ascend_custom_opp_path

    ensure_ascend_custom_opp_path()

    args = parse_args()

    engine = InferenceEngine(args)

    weight_sync = WeightSyncReceiver(engine, args)
    if args.weight_sync_addr:
        weight_sync.initialize()

    RequestHandler.engine = engine
    RequestHandler.weight_sync = weight_sync

    server = HTTPServer((args.host, args.port), RequestHandler)
    logger.info("Inference node listening on %s:%d", args.host, args.port)

    def shutdown_handler(signum, frame):
        logger.info("Shutting down...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    server.serve_forever()


if __name__ == "__main__":
    main()
