#!/usr/bin/env bash
# =============================================================================
# Forge — 2-node 8+8 (8 inference NPUs + 8 training NPUs)
#
# Node1 (TRAIN_HOST): TorchTitan FSDP2 training on 8 NPUs
# Node2 (INFER_HOST): vLLM inference server on 8 NPUs
#
# Prerequisites:
#   - SSH access between nodes (port configured via SSH_PORT)
#   - Same conda environment (monarch_ascend) on both nodes
#   - Model weights cached on both nodes
#   - AReaL repo at the same path on both nodes
#
# Usage:
#   bash forge/scripts/run_8x8_2node.sh
#   bash forge/scripts/run_8x8_2node.sh --steps 5
#   bash forge/scripts/run_8x8_2node.sh --train-host 192.168.0.26 --infer-host 192.168.0.23
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --------------- defaults (override via CLI flags) ---------------
TRAIN_HOST="192.168.0.26"
INFER_HOST="192.168.0.23"
SSH_PORT=36000
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_STEPS=2
INFER_PORT=8100
TRAIN_GPUS=8
INFER_GPUS=1
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0-beta.1}"
MODEL_NAME="qwen3"
MODEL_FLAVOR="1.7B"
LOSS_TYPE="grpo"
DATASET="openai/gsm8k"
REWARD_FN="areal.reward.gsm8k.gsm8k_reward_fn"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-host)  TRAIN_HOST="$2"; shift 2 ;;
        --infer-host)  INFER_HOST="$2"; shift 2 ;;
        --ssh-port)    SSH_PORT="$2";   shift 2 ;;
        --model)       MODEL_PATH="$2"; shift 2 ;;
        --steps)       TRAIN_STEPS="$2"; shift 2 ;;
        --infer-port)  INFER_PORT="$2"; shift 2 ;;
        --train-gpus)  TRAIN_GPUS="$2"; shift 2 ;;
        --infer-gpus)  INFER_GPUS="$2"; shift 2 ;;
        --cann)        CANN_HOME="$2";  shift 2 ;;
        --model-name)  MODEL_NAME="$2"; shift 2 ;;
        --model-flavor) MODEL_FLAVOR="$2"; shift 2 ;;
        --loss-type)   LOSS_TYPE="$2";  shift 2 ;;
        --dataset)     DATASET="$2";    shift 2 ;;
        --reward-fn)   REWARD_FN="$2";  shift 2 ;;
        *)             echo "Unknown flag: $1"; exit 1 ;;
    esac
done

SSH_CMD="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# --------------- helper: setup env on a node ---------------
REMOTE_ENV_SETUP=$(cat <<'ENVEOF'
set +eu
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0-beta.1}"
[[ -f "$CANN_HOME/set_env.sh" ]] && source "$CANN_HOME/set_env.sh"
ASCEND_ROOT="$(dirname "$CANN_HOME")"
[[ -f "$ASCEND_ROOT/nnal/atb/set_env.sh" ]] && source "$ASCEND_ROOT/nnal/atb/set_env.sh"
CONDA_EXE="${CONDA_EXE:-$(command -v conda 2>/dev/null || echo "$HOME/miniconda3/bin/conda")}"
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate monarch_ascend
set -eu
export VLLM_USE_MODELSCOPE=true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ENVEOF
)

# Inference node needs offline mode (monarch2 has no internet)
INFER_ENV_SETUP=$(cat <<'ENVEOF'
set +eu
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0-beta.1}"
[[ -f "$CANN_HOME/set_env.sh" ]] && source "$CANN_HOME/set_env.sh"
ASCEND_ROOT="$(dirname "$CANN_HOME")"
[[ -f "$ASCEND_ROOT/nnal/atb/set_env.sh" ]] && source "$ASCEND_ROOT/nnal/atb/set_env.sh"
CONDA_EXE="${CONDA_EXE:-$(command -v conda 2>/dev/null || echo "$HOME/miniconda3/bin/conda")}"
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate monarch_ascend
set -eu
export VLLM_USE_MODELSCOPE=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ENVEOF
)

INFER_LOG="/tmp/forge_inference_node.log"
TRAIN_LOG="/tmp/forge_train_node.log"

echo "============================================="
echo " Forge: 2-Node 8+8"
echo " Train:  ${TRAIN_HOST} (${TRAIN_GPUS} NPU, TorchTitan)"
echo " Infer:  ${INFER_HOST} (${INFER_GPUS} NPU, vLLM)"
echo " SSH:    port ${SSH_PORT}"
echo " Model:  ${MODEL_PATH}"
echo " Steps:  ${TRAIN_STEPS}"
echo "============================================="

# --------------- Step 1: Launch inference node on INFER_HOST ---------------
echo "[1/3] Launching inference node on ${INFER_HOST}..."

# Resolve model path for inference node (use local cache if HF name)
INFER_MODEL_PATH="${MODEL_PATH}"
if [[ ! -d "${MODEL_PATH}" ]]; then
    INFER_MODEL_PATH="/root/.cache/modelscope/hub/models/${MODEL_PATH}"
fi

$SSH_CMD root@${INFER_HOST} "
${INFER_ENV_SETUP}
cd ${AREAL_ROOT}

pkill -f 'forge.apps.inference_node' 2>/dev/null || true
sleep 1

nohup python -m forge.apps.inference_node \
    --model '${INFER_MODEL_PATH}' \
    --num-gpus ${INFER_GPUS} \
    --port ${INFER_PORT} \
    --dtype bfloat16 \
    --enforce-eager \
    --seed 1 \
    > ${INFER_LOG} 2>&1 &

echo \"Inference PID=\$!\"
" &
INFER_SSH_PID=$!

# Wait for inference node to start
echo "    Waiting for inference node to initialize..."
for i in $(seq 1 60); do
    sleep 10
    if $SSH_CMD root@${INFER_HOST} "curl -s --connect-timeout 3 http://localhost:${INFER_PORT}/health" 2>/dev/null | grep -q '"ok"'; then
        echo "    Inference node ready!"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "    ERROR: Inference node failed to start after 600s"
        echo "    Check logs: ssh -p ${SSH_PORT} root@${INFER_HOST} 'tail -50 ${INFER_LOG}'"
        kill $INFER_SSH_PID 2>/dev/null || true
        exit 1
    fi
    echo "    Still waiting... (${i}/60)"
done

# --------------- Step 2: Launch training on TRAIN_HOST ---------------
echo "[2/3] Launching training on ${TRAIN_HOST}..."

$SSH_CMD root@${TRAIN_HOST} "
${REMOTE_ENV_SETUP}
cd ${AREAL_ROOT}

torchrun --standalone --nproc_per_node=${TRAIN_GPUS} \
    -m forge.apps.grpo_multinode \
    --model '${MODEL_PATH}' \
    --model-name '${MODEL_NAME}' \
    --model-flavor '${MODEL_FLAVOR}' \
    --inference-addr '${INFER_HOST}' \
    --inference-port ${INFER_PORT} \
    --train-gpus ${TRAIN_GPUS} \
    --total-train-steps ${TRAIN_STEPS} \
    --loss-type '${LOSS_TYPE}' \
    --dataset '${DATASET}' \
    --reward-fn '${REWARD_FN}' \
    --no-weight-sync \
    --seed 1 \
    2>&1 | tee ${TRAIN_LOG}
"
TRAIN_EXIT=$?

# --------------- Step 3: Cleanup ---------------
echo "[3/3] Cleaning up..."

$SSH_CMD root@${INFER_HOST} "pkill -f 'forge.apps.inference_node' 2>/dev/null || true" 2>/dev/null || true
wait $INFER_SSH_PID 2>/dev/null || true

if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo "============================================="
    echo " 2-Node training completed successfully!"
    echo " Train log: ssh -p ${SSH_PORT} root@${TRAIN_HOST} 'cat ${TRAIN_LOG}'"
    echo " Infer log: ssh -p ${SSH_PORT} root@${INFER_HOST} 'cat ${INFER_LOG}'"
    echo "============================================="
else
    echo "============================================="
    echo " Training FAILED (exit code ${TRAIN_EXIT})"
    echo " Train log: ssh -p ${SSH_PORT} root@${TRAIN_HOST} 'cat ${TRAIN_LOG}'"
    echo " Infer log: ssh -p ${SSH_PORT} root@${INFER_HOST} 'cat ${INFER_LOG}'"
    echo "============================================="
    exit ${TRAIN_EXIT}
fi
