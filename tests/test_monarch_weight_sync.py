"""Unit tests for forge weight_sync (XCCL alloc_mode for Monarch).

Run (needs full AReaL deps, e.g. Ascend conda env):

    cd /path/to/AReaL && conda run -n monarch_ascend python -m unittest tests.test_monarch_weight_sync -v
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from areal.api.alloc_mode import AllocationMode, ParallelStrategy
from forge.adapters.areal.weight_sync import (
    build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime,
    monarch_xccl_weight_update_alloc_mode_flat,
    optional_flat_inference_xccl_from_env,
    resolve_xccl_alloc_mode,
)


class TestMonarchWeightSync(unittest.TestCase):
    def test_vllm_runtime_dp4_train4(self):
        """Explicit vLLM-style DP=4 matches four XCCL inference ranks."""
        ps = ParallelStrategy(
            data_parallel_size=4,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        wu = build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime(
            ps, gen_backend="vllm", train_world_size=4
        )
        self.assertEqual(wu.gen.world_size, 4)
        self.assertEqual(wu.train.world_size, 4)

    def test_embedded_dp1_four_inference_devices(self):
        """Cluster d4 devices but vLLM yaml leaves dp=1 → XCCL gen.world_size=1."""
        cluster = AllocationMode.from_str("vllm:d4p1t1+d4p1t1")
        self.assertEqual(cluster.gen.world_size, 4)
        ps = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        wu = build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime(
            ps, gen_backend="vllm", train_world_size=4
        )
        self.assertEqual(wu.gen.world_size, 1)
        self.assertEqual(wu.train.world_size, 4)

    def test_build_preserves_inference_tp(self):
        ps = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        )
        wu = build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime(
            ps, gen_backend="vllm", train_world_size=4
        )
        self.assertEqual(wu.gen.world_size, 4)
        self.assertEqual(wu.gen.tp_size, 4)
        self.assertEqual(wu.gen.dp_size, 1)
        self.assertEqual(wu.train.world_size, 4)

    def test_flat_override(self):
        ps = ParallelStrategy(
            data_parallel_size=4,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        wu = build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime(
            ps,
            gen_backend="vllm",
            train_world_size=4,
            flat_inference_dp=1,
        )
        self.assertEqual(wu.gen.world_size, 1)
        self.assertEqual(wu.train.world_size, 4)

    def test_monarch_xccl_weight_update_alloc_mode_flat(self):
        wu = monarch_xccl_weight_update_alloc_mode_flat(4, 4, rollout_backend="vllm")
        self.assertEqual(wu.gen.world_size, 4)
        self.assertEqual(wu.train.world_size, 4)

    def test_optional_flat_inference_xccl_from_env(self):
        saved = os.environ.pop("MONARCH_INFERENCE_XCCL_PARTICIPANTS", None)
        try:
            self.assertIsNone(optional_flat_inference_xccl_from_env())
        finally:
            if saved is not None:
                os.environ["MONARCH_INFERENCE_XCCL_PARTICIPANTS"] = saved

        with patch.dict(os.environ, {"MONARCH_INFERENCE_XCCL_PARTICIPANTS": "2"}):
            self.assertEqual(optional_flat_inference_xccl_from_env(), 2)

        with patch.dict(os.environ, {"MONARCH_INFERENCE_XCCL_PARTICIPANTS": "0"}):
            with self.assertRaises(ValueError):
                optional_flat_inference_xccl_from_env()

    def test_build_rejects_bad_train_world_size(self):
        ps = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        with self.assertRaises(ValueError):
            build_monarch_xccl_weight_update_alloc_mode_from_vllm_runtime(
                ps, gen_backend="vllm", train_world_size=0
            )


class TestResolveXcclAllocMode(unittest.TestCase):
    """Tests for the high-level ``resolve_xccl_alloc_mode`` entry point."""

    @staticmethod
    def _make_config(vllm_config=None):
        config = MagicMock()
        config.vllm = vllm_config or MagicMock()
        return config

    @patch(
        "forge.adapters.areal.weight_sync._vllm_gen_parallel",
        return_value=ParallelStrategy(1, 1, 1),
    )
    def test_default_dp1_no_env(self, mock_gen_ps):
        """No env override, vLLM runtime dp=1 → gen.world_size=1."""
        config = self._make_config()
        alloc_mode = AllocationMode.from_str("vllm:d4p1t1+d4p1t1")

        result = resolve_xccl_alloc_mode(config, alloc_mode, train_world_size=4)
        self.assertEqual(result.gen.world_size, 1)
        self.assertEqual(result.train.world_size, 4)

    @patch.dict(os.environ, {"MONARCH_INFERENCE_XCCL_PARTICIPANTS": "2"})
    @patch(
        "forge.adapters.areal.weight_sync._vllm_gen_parallel",
        return_value=ParallelStrategy(4, 1, 1),
    )
    def test_env_override_ignores_vllm_runtime(self, mock_gen_ps):
        """Env override forces flat dp=2 even though vLLM runtime says dp=4."""
        config = self._make_config()
        alloc_mode = AllocationMode.from_str("vllm:d4p1t1+d4p1t1")

        result = resolve_xccl_alloc_mode(config, alloc_mode, train_world_size=4)
        self.assertEqual(result.gen.world_size, 2)
        self.assertEqual(result.train.world_size, 4)

    @patch(
        "forge.adapters.areal.weight_sync._vllm_gen_parallel",
        return_value=ParallelStrategy(2, 2, 1),
    )
    def test_preserves_tp_from_runtime(self, mock_gen_ps):
        """TP=2, DP=2 from vLLM runtime → gen.world_size=4, tp=2."""
        config = self._make_config()
        alloc_mode = AllocationMode.from_str("vllm:d4p1t1+d4p1t1")

        result = resolve_xccl_alloc_mode(config, alloc_mode, train_world_size=4)
        self.assertEqual(result.gen.world_size, 4)
        self.assertEqual(result.gen.tp_size, 2)


if __name__ == "__main__":
    unittest.main()
