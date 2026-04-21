#!/usr/bin/env bash
# =============================================================================
# Forge — 4-card inference + 4-card training (GSM8K GRPO on NPU)
#
# Uses NPU 0-3 for vLLM inference (DP=4), NPU 4-7 for FSDP training (DP=4).
# Total: 8 NPUs.
#
# Usage:
#   bash forge/scripts/run_4x4.sh
#   bash forge/scripts/run_4x4.sh --steps 5
#   bash forge/scripts/run_4x4.sh --model /path/to/local/model
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --------------- defaults (override via CLI flags) ---------------
TRAIN_STEPS=2
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0-beta.1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)   TRAIN_STEPS="$2"; shift 2 ;;
        --model)   MODEL_PATH="$2";  shift 2 ;;
        --cann)    CANN_HOME="$2";   shift 2 ;;
        *)         echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# --------------- environment setup ---------------
set +eu
[[ -f "$CANN_HOME/set_env.sh" ]] && source "$CANN_HOME/set_env.sh"
ASCEND_ROOT="$(dirname "$CANN_HOME")"
[[ -f "$ASCEND_ROOT/nnal/atb/set_env.sh" ]] && source "$ASCEND_ROOT/nnal/atb/set_env.sh"
set -eu

if [[ "${CONDA_DEFAULT_ENV:-}" != "monarch_ascend" ]]; then
    CONDA_EXE="${CONDA_EXE:-$(command -v conda 2>/dev/null || echo "$HOME/miniconda3/bin/conda")}"
    eval "$("$CONDA_EXE" shell.bash hook)"
    conda activate monarch_ascend
fi

export VLLM_USE_MODELSCOPE=true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# --------------- run ---------------
cd "$AREAL_ROOT"

echo "============================================="
echo " Forge: 4+4 (4 inf + 4 train)"
echo " Model:       $MODEL_PATH"
echo " Train steps: $TRAIN_STEPS"
echo " CANN:        $CANN_HOME"
echo "============================================="

python -m forge.apps.grpo \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "actor.path=$MODEL_PATH" \
    "+total_train_steps=$TRAIN_STEPS"
