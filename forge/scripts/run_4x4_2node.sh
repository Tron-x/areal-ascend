#!/usr/bin/env bash
# =============================================================================
# Forge — 2-node 4+4 (4-card inference on node2 + 4-card training on node1)
#
# Uses Monarch bare-metal launcher: starts workers on both nodes,
# then runs the same grpo.py as single-machine but with Generator
# placed on the remote node via BareMetalLauncher.
#
# Prerequisites:
#   - SSH between nodes (port configured via SSH_PORT)
#   - Same conda env and AReaL repo on both nodes
#   - Model weights cached on both nodes
#
# Usage:
#   bash forge/scripts/run_4x4_2node.sh
#   bash forge/scripts/run_4x4_2node.sh --steps 5
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --------------- defaults ---------------
TRAIN_HOST="192.168.0.26"
INFER_HOST="192.168.0.23"
SSH_PORT=36000
MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_STEPS=2
WORKER_PORT=22222
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0-beta.1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-host)   TRAIN_HOST="$2"; shift 2 ;;
        --infer-host)   INFER_HOST="$2"; shift 2 ;;
        --ssh-port)     SSH_PORT="$2"; shift 2 ;;
        --model)        MODEL_PATH="$2"; shift 2 ;;
        --steps)        TRAIN_STEPS="$2"; shift 2 ;;
        --worker-port)  WORKER_PORT="$2"; shift 2 ;;
        --cann)         CANN_HOME="$2"; shift 2 ;;
        *)              echo "Unknown flag: $1"; exit 1 ;;
    esac
done

SSH_CMD="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"
PYTHON_EXE="/root/miniconda3/envs/monarch_ascend/bin/python3"

# Env setup snippet for remote shells
ENV_SETUP="source ${CANN_HOME}/set_env.sh 2>/dev/null; \
source $(dirname ${CANN_HOME})/nnal/atb/set_env.sh 2>/dev/null; \
eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\"; \
conda activate monarch_ascend; \
export VLLM_USE_MODELSCOPE=true; \
export HF_ENDPOINT=https://hf-mirror.com; \
export HF_HUB_OFFLINE=0; \
export TRANSFORMERS_OFFLINE=0; \
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"

# For monarch2 (no internet)
INFER_ENV_SETUP="source ${CANN_HOME}/set_env.sh 2>/dev/null; \
source $(dirname ${CANN_HOME})/nnal/atb/set_env.sh 2>/dev/null; \
eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\"; \
conda activate monarch_ascend; \
export VLLM_USE_MODELSCOPE=false; \
export HF_HUB_OFFLINE=1; \
export TRANSFORMERS_OFFLINE=1; \
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"

echo "============================================="
echo " Forge: 2-Node 4+4 (AReaL backend)"
echo " Train:  ${TRAIN_HOST} (4 NPU, Monarch worker)"
echo " Infer:  ${INFER_HOST} (4 NPU, Monarch worker)"
echo " Model:  ${MODEL_PATH}"
echo " Steps:  ${TRAIN_STEPS}"
echo "============================================="

# --------------- Step 1: Kill old workers ---------------
echo "[1/4] Cleaning up old workers..."
$SSH_CMD root@${TRAIN_HOST} "pkill -f run_worker_loop 2>/dev/null || true" 2>/dev/null || true
$SSH_CMD root@${INFER_HOST} "pkill -f run_worker_loop 2>/dev/null || true" 2>/dev/null || true
sleep 2

# --------------- Step 2: Start Monarch workers on both nodes ---------------
echo "[2/4] Starting Monarch workers..."

# Create worker scripts on each node
INFER_WORKER_SCRIPT="/tmp/start_forge_worker.py"
TRAIN_WORKER_SCRIPT="/tmp/start_forge_worker.py"

# Write worker script to INFER_HOST
$SSH_CMD root@${INFER_HOST} "cat > ${INFER_WORKER_SCRIPT} << 'WEOF'
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['VLLM_USE_MODELSCOPE'] = 'false'
os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, '${AREAL_ROOT}')
from monarch.actor import run_worker_loop_forever
run_worker_loop_forever(address='tcp://${INFER_HOST}:${WORKER_PORT}', ca='trust_all_connections')
WEOF
echo 'infer script created'" || true

# Write worker script to TRAIN_HOST
$SSH_CMD root@${TRAIN_HOST} "cat > ${TRAIN_WORKER_SCRIPT} << 'WEOF'
import os, sys
sys.path.insert(0, '${AREAL_ROOT}')
from monarch.actor import run_worker_loop_forever
run_worker_loop_forever(address='tcp://${TRAIN_HOST}:${WORKER_PORT}', ca='trust_all_connections')
WEOF
echo 'train script created'" || true

# Worker on INFER_HOST (monarch2)
$SSH_CMD root@${INFER_HOST} "${INFER_ENV_SETUP}; \
cd ${AREAL_ROOT}; \
nohup ${PYTHON_EXE} ${INFER_WORKER_SCRIPT} > /tmp/monarch_worker.log 2>&1 & \
echo PID=\$!" &

# Worker on TRAIN_HOST (monarch1)
$SSH_CMD root@${TRAIN_HOST} "${ENV_SETUP}; \
cd ${AREAL_ROOT}; \
nohup ${PYTHON_EXE} ${TRAIN_WORKER_SCRIPT} > /tmp/monarch_worker.log 2>&1 & \
echo PID=\$!" &

wait

# Wait for workers to be ready
echo "    Waiting for workers to start..."
for HOST_IP in ${TRAIN_HOST} ${INFER_HOST}; do
    for i in $(seq 1 30); do
        if $SSH_CMD root@${TRAIN_HOST} "python3 -c \"import socket; s=socket.socket(); s.settimeout(2); s.connect(('${HOST_IP}', ${WORKER_PORT})); s.close(); print('ok')\"" 2>/dev/null | grep -q "ok"; then
            echo "    ${HOST_IP}:${WORKER_PORT} ready"
            break
        fi
        sleep 2
        if [ "$i" -eq 30 ]; then
            echo "    ERROR: Worker on ${HOST_IP} did not start"
            exit 1
        fi
    done
done

# --------------- Step 3: Run GRPO training (driver on TRAIN_HOST) ---------------
echo "[3/4] Running GRPO training..."

# First worker = Generator (remote inference), second = Trainer (local training)
WORKERS="tcp://${INFER_HOST}:${WORKER_PORT},tcp://${TRAIN_HOST}:${WORKER_PORT}"

$SSH_CMD root@${TRAIN_HOST} "\
${ENV_SETUP}; \
cd ${AREAL_ROOT}; \
python -m forge.apps.grpo \
    examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    'actor.path=${MODEL_PATH}' \
    '+total_train_steps=${TRAIN_STEPS}' \
    --bare-metal-workers '${WORKERS}' \
    --bare-metal-master-addr '${TRAIN_HOST}' \
    2>&1 | tee /tmp/forge_2node.log"
TRAIN_EXIT=$?

# --------------- Step 4: Cleanup ---------------
echo "[4/4] Cleaning up workers..."
$SSH_CMD root@${TRAIN_HOST} "pkill -f run_worker_loop 2>/dev/null || true" 2>/dev/null || true
$SSH_CMD root@${INFER_HOST} "pkill -f run_worker_loop 2>/dev/null || true" 2>/dev/null || true

if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo "============================================="
    echo " 2-Node 4+4 training completed successfully!"
    echo "============================================="
else
    echo "============================================="
    echo " Training FAILED (exit code ${TRAIN_EXIT})"
    echo "============================================="
    exit ${TRAIN_EXIT}
fi
