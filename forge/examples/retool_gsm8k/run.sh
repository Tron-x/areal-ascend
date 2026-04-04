#!/usr/bin/env bash
# =============================================================================
# Forge ReTool GSM8K -- multi-turn tool-integrated reasoning on NPU
#
# Uses Forge's ReToolAgent + PythonSandbox for tool-calling,
# with AReaL as the training engine backend.
#
# Usage:
#   bash forge/examples/retool_gsm8k/run.sh
#   bash forge/examples/retool_gsm8k/run.sh --steps 5
#   bash forge/examples/retool_gsm8k/run.sh --async
#   bash forge/examples/retool_gsm8k/run.sh --model /path/to/model
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --------------- defaults ---------------
TRAIN_STEPS=2
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
CANN_HOME="${CANN_HOME:-/root/hzz/cann-9.0.0-beta.1}"
ASYNC_MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)   TRAIN_STEPS="$2"; shift 2 ;;
        --model)   MODEL_PATH="$2";  shift 2 ;;
        --cann)    CANN_HOME="$2";   shift 2 ;;
        --async)   ASYNC_MODE="1";   shift ;;
        *)         echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# --------------- environment setup ---------------
set +eu
[[ -f "$CANN_HOME/set_env.sh" ]] && source "$CANN_HOME/set_env.sh"
[[ -f "$CANN_HOME/nnal/atb/set_env.sh" ]] && source "$CANN_HOME/nnal/atb/set_env.sh"
set -eu

# Deactivate any virtualenv that might override conda
export VIRTUAL_ENV=""
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '.venv' | tr '\n' ':')

if [[ "${CONDA_DEFAULT_ENV:-}" != "monarch_ascend" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate monarch_ascend
fi

export VLLM_USE_MODELSCOPE=true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

if [[ -n "$ASYNC_MODE" ]]; then
    export FORGE_ASYNC_PIPELINE=1
fi

# --------------- run ---------------
cd "$AREAL_ROOT"

echo "============================================="
echo " Forge ReTool GSM8K"
echo " Model:       $MODEL_PATH"
echo " Train steps: $TRAIN_STEPS"
echo " Async mode:  ${ASYNC_MODE:-disabled}"
echo " CANN:        $CANN_HOME"
echo "============================================="

python -m forge.apps.agent_rl \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "actor.path=$MODEL_PATH" \
    "+total_train_steps=$TRAIN_STEPS"
