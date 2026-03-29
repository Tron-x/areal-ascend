#!/usr/bin/env bash
# =============================================================================
# Monarch AReaL -- Multi-node launcher
#
# Requires Monarch workers running on each node:
#   On each worker node:
#     python -c "from monarch._src.actor.bootstrap import run_worker_loop_forever; \
#                run_worker_loop_forever(address='tcp://0.0.0.0:29600', \
#                ca='trust_all_connections')"
#
# Then on the driver node:
#   bash areal/monarch_plugin/scripts/run_multi_node.sh \
#       --workers "tcp://node0:29600,tcp://node1:29600" \
#       --nodes 2 --inf 4 --train 4
#
# This creates a 2-node cluster where each node contributes 4 inf + 4 train
# NPUs (symmetric placement).
#
# Role-split mode (node0=inference, node1=training):
#   MONARCH_NODE_ROLES=inference,training \
#   bash areal/monarch_plugin/scripts/run_multi_node.sh \
#       --workers "tcp://node0:29600,tcp://node1:29600" \
#       --nodes 2 --inf 8 --train 8
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --------------- defaults ---------------
WORKERS=""
N_NODES=2
GPUS_PER_NODE=8
INF_CARDS=4
TRAIN_CARDS=4
TRAIN_STEPS=2
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
CANN_HOME="${CANN_HOME:-/root/hzz/cann-9.0.0-beta.1}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)       WORKERS="$2";        shift 2 ;;
        --nodes)         N_NODES="$2";        shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2";  shift 2 ;;
        --inf)           INF_CARDS="$2";      shift 2 ;;
        --train)         TRAIN_CARDS="$2";    shift 2 ;;
        --steps)         TRAIN_STEPS="$2";    shift 2 ;;
        --model)         MODEL_PATH="$2";     shift 2 ;;
        --cann)          CANN_HOME="$2";      shift 2 ;;
        *)               EXTRA_ARGS+=("$1");  shift ;;
    esac
done

if [[ -z "$WORKERS" ]]; then
    echo "ERROR: --workers is required (comma-separated list of tcp://host:port)"
    echo "  Example: --workers tcp://node0:29600,tcp://node1:29600"
    exit 1
fi

ALLOC_MODE="vllm:d${INF_CARDS}p1t1+d${TRAIN_CARDS}p1t1"

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
export MONARCH_WORKERS="$WORKERS"

# --------------- run ---------------
cd "$AREAL_ROOT"

echo "============================================="
echo " Monarch AReaL: Multi-Node"
echo " Nodes:        $N_NODES"
echo " Workers:      $WORKERS"
echo " GPUs/node:    $GPUS_PER_NODE"
echo " Allocation:   $ALLOC_MODE"
echo " Model:        $MODEL_PATH"
echo " Train steps:  $TRAIN_STEPS"
echo "============================================="

python -m areal.monarch_plugin.launcher \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    "allocation_mode=$ALLOC_MODE" \
    "cluster.n_nodes=$N_NODES" \
    "cluster.n_gpus_per_node=$GPUS_PER_NODE" \
    "actor.path=$MODEL_PATH" \
    "+total_train_steps=$TRAIN_STEPS" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
