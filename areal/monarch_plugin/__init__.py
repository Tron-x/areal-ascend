"""Monarch plugin for AReaL -- non-invasive integration layer.

Embeds vLLM's AsyncLLM inside a Monarch actor with MonarchExecutor for
distributed worker management, and replaces HTTP communication between
trainer and inference with Monarch RPC.

Usage:
    python -m areal.monarch_plugin.launcher \\
        examples/math/gsm8k_rl.py \\
        --config examples/math/gsm8k_grpo_npu.yaml
"""
