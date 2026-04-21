#!/usr/bin/env bash
# =============================================================================
# Forge — Multi-node GRPO training (hostfile-driven)
#
# Manages the full lifecycle: hostname setup → worker start → training → cleanup
#
# Usage:
#   bash forge/scripts/run_multinode.sh --hostfile forge/configs/hostfile.txt
#   bash forge/scripts/run_multinode.sh --hostfile forge/configs/hostfile.txt --steps 5
#   bash forge/scripts/run_multinode.sh --hostfile forge/configs/hostfile.txt --model /path/to/model
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKER_MGR="${SCRIPT_DIR}/worker_manager.sh"

# --------------- defaults ---------------
HOSTFILE=""
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_STEPS=2
WORKER_PORT=22222
SSH_PORT=36000
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0-beta.1}"
BACKEND=""
MODEL_NAME=""
MODEL_FLAVOR=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hostfile)      HOSTFILE="$2"; shift 2 ;;
        --model)         MODEL_PATH="$2"; shift 2 ;;
        --steps)         TRAIN_STEPS="$2"; shift 2 ;;
        --port)          WORKER_PORT="$2"; shift 2 ;;
        --ssh-port)      SSH_PORT="$2"; shift 2 ;;
        --cann)          CANN_HOME="$2"; shift 2 ;;
        --backend)       BACKEND="$2"; shift 2 ;;
        --model-name)    MODEL_NAME="$2"; shift 2 ;;
        --model-flavor)  MODEL_FLAVOR="$2"; shift 2 ;;
        *)               EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done

if [[ -z "$HOSTFILE" ]]; then
    echo "Error: --hostfile is required"
    echo "Usage: $0 --hostfile forge/configs/hostfile.txt [--steps N] [--model PATH]"
    exit 1
fi

# Read the driver node (first host with an active worker acts as driver)
DRIVER_HOST=$(grep -v '^#' "$HOSTFILE" | grep -v '^\s*$' | awk 'NR==2{print $1}')
if [[ -z "$DRIVER_HOST" ]]; then
    DRIVER_HOST=$(grep -v '^#' "$HOSTFILE" | grep -v '^\s*$' | awk 'NR==1{print $1}')
fi

WORKERS=$(bash "$WORKER_MGR" workers-arg --hostfile "$HOSTFILE" --port "$WORKER_PORT" --ssh-port "$SSH_PORT")

echo "============================================="
echo " Forge: Multi-node GRPO"
echo " Hostfile: ${HOSTFILE}"
echo " Driver:   ${DRIVER_HOST}"
echo " Workers:  ${WORKERS}"
echo " Model:    ${MODEL_PATH}"
echo " Backend:  ${BACKEND:-areal}"
echo " Steps:    ${TRAIN_STEPS}"
echo "============================================="

# --------------- Step 1: Setup hostnames ---------------
echo "[1/4] Configuring hostname resolution..."
bash "$WORKER_MGR" hosts-setup --hostfile "$HOSTFILE" --ssh-port "$SSH_PORT"

# --------------- Step 2: Start workers ---------------
echo "[2/4] Starting Monarch workers..."
bash "$WORKER_MGR" stop --hostfile "$HOSTFILE" --port "$WORKER_PORT" --ssh-port "$SSH_PORT" 2>/dev/null || true
sleep 3
bash "$WORKER_MGR" start --hostfile "$HOSTFILE" --port "$WORKER_PORT" --ssh-port "$SSH_PORT" --cann "$CANN_HOME" --areal-root "$AREAL_ROOT"

# --------------- Step 3: Run training ---------------
echo "[3/4] Running GRPO training on ${DRIVER_HOST}..."

SSH_CMD="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"

$SSH_CMD root@${DRIVER_HOST} "\
source ${CANN_HOME}/set_env.sh 2>/dev/null; \
ASCEND_ROOT=\$(dirname ${CANN_HOME}); \
[[ -f \"\${ASCEND_ROOT}/nnal/atb/set_env.sh\" ]] && source \"\${ASCEND_ROOT}/nnal/atb/set_env.sh\"; \
eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\"; \
conda activate monarch_ascend; \
export VLLM_USE_MODELSCOPE=true; \
export HF_ENDPOINT=https://hf-mirror.com; \
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True; \
cd ${AREAL_ROOT}; \
BACKEND_ARGS=''; \
[ -n '${BACKEND}' ] && BACKEND_ARGS=\"--backend ${BACKEND}\"; \
[ -n '${MODEL_NAME}' ] && BACKEND_ARGS=\"\${BACKEND_ARGS} --model-name ${MODEL_NAME}\"; \
[ -n '${MODEL_FLAVOR}' ] && BACKEND_ARGS=\"\${BACKEND_ARGS} --model-flavor ${MODEL_FLAVOR}\"; \
python -m forge.apps.grpo \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    'actor.path=${MODEL_PATH}' \
    '+total_train_steps=${TRAIN_STEPS}' \
    --bare-metal-workers '${WORKERS}' \
    --bare-metal-master-addr '${DRIVER_HOST}' \
    \${BACKEND_ARGS} \
    ${EXTRA_ARGS} \
    2>&1 | tee /tmp/forge_multinode.log"
TRAIN_EXIT=$?

# --------------- Step 4: Cleanup ---------------
echo "[4/4] Stopping workers..."
bash "$WORKER_MGR" stop --hostfile "$HOSTFILE" --port "$WORKER_PORT" --ssh-port "$SSH_PORT" 2>/dev/null || true

if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo "============================================="
    echo " Multi-node training completed successfully!"
    echo "============================================="
else
    echo "============================================="
    echo " Training FAILED (exit code ${TRAIN_EXIT})"
    echo "============================================="
    exit ${TRAIN_EXIT}
fi
