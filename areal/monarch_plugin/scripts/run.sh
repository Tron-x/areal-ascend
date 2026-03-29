#!/usr/bin/env bash
# =============================================================================
# Monarch AReaL -- Generic launcher for any device configuration
#
# Automatically configures allocation_mode and n_gpus_per_node based on
# --inf and --train flags.
#
# Usage:
#   bash areal/monarch_plugin/scripts/run.sh --inf 1 --train 1          # 1+1
#   bash areal/monarch_plugin/scripts/run.sh --inf 4 --train 4          # 4+4
#   bash areal/monarch_plugin/scripts/run.sh --inf 2 --train 6          # 2+6
#   bash areal/monarch_plugin/scripts/run.sh --inf 1 --train 3          # 1+3
#   bash areal/monarch_plugin/scripts/run.sh --inf 4 --train 4 --tp 2   # TP=2
#   bash areal/monarch_plugin/scripts/run.sh --inf 1 --train 1 --steps 10 --model /path
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --------------- defaults ---------------
INF_CARDS=1
TRAIN_CARDS=1
TP_SIZE=1
PP_SIZE=1
TRAIN_STEPS=3
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
CANN_HOME="${CANN_HOME:-/root/hzz/cann-9.0.0-beta.1}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --inf)     INF_CARDS="$2";   shift 2 ;;
        --train)   TRAIN_CARDS="$2"; shift 2 ;;
        --tp)      TP_SIZE="$2";     shift 2 ;;
        --pp)      PP_SIZE="$2";     shift 2 ;;
        --steps)   TRAIN_STEPS="$2"; shift 2 ;;
        --model)   MODEL_PATH="$2";  shift 2 ;;
        --cann)    CANN_HOME="$2";   shift 2 ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done

TOTAL_CARDS=$((INF_CARDS + TRAIN_CARDS))
INF_DP=$((INF_CARDS / TP_SIZE / PP_SIZE))
ALLOC_MODE="vllm:d${INF_DP}p${PP_SIZE}t${TP_SIZE}+d${TRAIN_CARDS}p1t1"

# --------------- environment setup ---------------
if [[ -f "$CANN_HOME/set_env.sh" ]]; then
    source "$CANN_HOME/set_env.sh"
fi
if [[ -f "$CANN_HOME/nnal/atb/set_env.sh" ]]; then
    source "$CANN_HOME/nnal/atb/set_env.sh"
fi

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
echo " Monarch AReaL: ${INF_CARDS}+${TRAIN_CARDS}"
echo " Allocation:  $ALLOC_MODE"
echo " Total NPUs:  $TOTAL_CARDS"
echo " Model:       $MODEL_PATH"
echo " Train steps: $TRAIN_STEPS"
echo "============================================="

python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "allocation_mode=$ALLOC_MODE" \
    "cluster.n_gpus_per_node=$TOTAL_CARDS" \
    "actor.path=$MODEL_PATH" \
    "+total_train_steps=$TRAIN_STEPS" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
