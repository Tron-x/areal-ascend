#!/usr/bin/env bash
# =============================================================================
# Monarch AReaL -- 1-card inference + 1-card training (GSM8K GRPO on NPU)
#
# Uses NPU 0 for vLLM inference, NPU 1 for FSDP training.
# Total: 2 NPUs.
#
# Architecture:
#   MonarchOrchestrator (main process, no NPU)
#     ├── GeneratorActor      (NPU 0)  — vLLM AsyncLLM inference
#     ├── RewardActor          (CPU)    — reward computation
#     ├── SandboxActor         (CPU)    — isolated code execution
#     ├── AgentActor           (CPU)    — multi-turn orchestration
#     ├── ReplayBufferActor    (CPU)    — async experience replay
#     ├── RolloutActor         (CPU)    — dataloader + rollout production
#     └── TrainerActor[rank=0] (NPU 1)  — FSDP training (single process)
#
# Usage:
#   bash areal/monarch_plugin/scripts/run_1x1.sh
#   bash areal/monarch_plugin/scripts/run_1x1.sh --steps 5
#   bash areal/monarch_plugin/scripts/run_1x1.sh --model /path/to/local/model
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --------------- defaults (override via CLI flags) ---------------
TRAIN_STEPS=3
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
CANN_HOME="${CANN_HOME:-/root/hzz/cann-9.0.0-beta.1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)   TRAIN_STEPS="$2"; shift 2 ;;
        --model)   MODEL_PATH="$2";  shift 2 ;;
        --cann)    CANN_HOME="$2";   shift 2 ;;
        *)         echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# --------------- environment setup ---------------
if [[ -f "$CANN_HOME/set_env.sh" ]]; then
    source "$CANN_HOME/set_env.sh"
else
    echo "[WARN] CANN set_env.sh not found at $CANN_HOME/set_env.sh"
fi

if [[ -f "$CANN_HOME/nnal/atb/set_env.sh" ]]; then
    source "$CANN_HOME/nnal/atb/set_env.sh"
else
    echo "[WARN] ATB set_env.sh not found at $CANN_HOME/nnal/atb/set_env.sh"
fi

# conda (only activate if not already in the environment)
if [[ "${CONDA_DEFAULT_ENV:-}" != "monarch_ascend" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate monarch_ascend
fi

export VLLM_USE_MODELSCOPE=true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# --------------- run ---------------
cd "$AREAL_ROOT"

echo "============================================="
echo " Monarch AReaL: 1+1 (1 inf + 1 train)"
echo " Model:       $MODEL_PATH"
echo " Train steps: $TRAIN_STEPS"
echo " CANN:        $CANN_HOME"
echo "============================================="

python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "allocation_mode=vllm:d1p1t1+d1p1t1" \
    "cluster.n_gpus_per_node=2" \
    "actor.path=$MODEL_PATH" \
    "+total_train_steps=$TRAIN_STEPS"
