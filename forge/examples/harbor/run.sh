#!/usr/bin/env bash
# =============================================================================
# Forge + Harbor -- Agentic RL with Harbor workflows on NPU
#
# Bridges Harbor's rllm agent layer (Workflow/Agent/Environment) with
# Forge's training infrastructure (AReaL engine, vLLM inference, GRPO).
#
# POC: runs GSM8K math tasks with the HarborAgentLogic adapter,
# using rllm's math reward function and Forge's training pipeline.
#
# Usage:
#   bash forge/examples/harbor/run.sh
#   bash forge/examples/harbor/run.sh --steps 5
#   bash forge/examples/harbor/run.sh --model /path/to/local/model
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

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
    eval "$(conda shell.bash hook)"
    conda activate monarch_ascend
fi

export VLLM_USE_MODELSCOPE=true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# Ensure Harbor's rllm package is importable
HARBOR_ROOT="${HARBOR_ROOT:-/root/harbor/harbor-verl-train}"
if [[ -d "$HARBOR_ROOT/rllm" ]]; then
    export PYTHONPATH="${HARBOR_ROOT}:${PYTHONPATH:-}"
fi

# Ensure required directories exist
mkdir -p /tmp/areal/experiments /tmp/areal/name_resolve

# --------------- run ---------------
cd "$AREAL_ROOT"

echo "============================================="
echo " Forge + Harbor Adapter POC"
echo " Model:       $MODEL_PATH"
echo " Train steps: $TRAIN_STEPS"
echo " CANN:        $CANN_HOME"
echo " Harbor:      $HARBOR_ROOT"
echo "============================================="

python -m forge.apps.agent_rl \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "actor.path=$MODEL_PATH" \
    "+total_train_steps=$TRAIN_STEPS"
