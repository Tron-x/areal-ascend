#!/usr/bin/env bash
# =============================================================================
# Monarch Worker Manager — start/stop/status workers from a hostfile
#
# Hostfile format (compatible with DeepSpeed/MPI):
#   192.168.0.26 slots=8
#   192.168.0.23 slots=8
#
# Usage:
#   # Default profile (pure-training): HCCS intra-host + RoCE inter-host.
#   # Use for B-mini, SFT, pretrain, or any workload without in-process
#   # HiXL/torchstore.
#   bash forge/scripts/worker_manager.sh start  --hostfile hostfile.txt
#
#   # GRPO + weight-sync profile: pins HCCL intra-host onto RoCE so
#   # HiXL/HCCL share the same NPU cleanly.  Required by `forge launch`.
#   bash forge/scripts/worker_manager.sh start  --hostfile hostfile.txt \
#         --profile hixl-coexist
#
#   bash forge/scripts/worker_manager.sh stop   --hostfile hostfile.txt
#   bash forge/scripts/worker_manager.sh status --hostfile hostfile.txt
#
# Run `worker_manager.sh help` for the full option list including
# --port, --ssh-port, --cann, --conda-env, --areal-root.
# =============================================================================
set -euo pipefail

ACTION="${1:-help}"
shift || true

# Defaults
HOSTFILE=""
WORKER_PORT=22222
SSH_PORT=36000
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0-beta.1}"
CONDA_ENV="monarch_ascend"
AREAL_ROOT="${AREAL_ROOT:-/root/AReaL}"
# Profile selects which env-var preset the worker bootstrap installs.
# See worker_script() below for the full table; in short:
#   pure-training : minimal HCCL env, intra-host HCCS + inter-host RoCE
#                   (auto-selected by HCCL).  Use for B-mini, SFT, pretrain,
#                   or any workload that does NOT mix HiXL/torchstore into
#                   the same process as HCCL.
#   hixl-coexist  : the historical GRPO + weight-sync env (pins HCCL to
#                   RoCE intra-host, opens 60000-60255 for HCCL to clear
#                   the 16666 range for HiXL, configures torchstore RDMA
#                   pool).  Use for `forge launch` GRPO runs.
PROFILE="${PROFILE:-pure-training}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hostfile)     HOSTFILE="$2"; shift 2 ;;
        --port)         WORKER_PORT="$2"; shift 2 ;;
        --ssh-port)     SSH_PORT="$2"; shift 2 ;;
        --cann)         CANN_HOME="$2"; shift 2 ;;
        --conda-env)    CONDA_ENV="$2"; shift 2 ;;
        --areal-root)   AREAL_ROOT="$2"; shift 2 ;;
        --profile)      PROFILE="$2"; shift 2 ;;
        *)              echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Validate PROFILE early so a typo fails fast on the launcher host instead
# of silently falling through to the default branch on every remote.
case "$PROFILE" in
    pure-training|hixl-coexist) ;;
    *)
        echo "Error: --profile must be one of: pure-training, hixl-coexist (got: '${PROFILE}')"
        exit 1
        ;;
esac

if [[ -z "$HOSTFILE" && "$ACTION" != "help" ]]; then
    echo "Error: --hostfile is required"
    exit 1
fi

SSH_CMD="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# Parse hostfile: extract IPs (ignore slots= and comments)
parse_hosts() {
    grep -v '^#' "$HOSTFILE" | grep -v '^\s*$' | awk '{print $1}'
}

# Common env block: every profile gets these (CANN sourcing, conda activate,
# PYTHONPATH, NPU allocator, generic CANN/HCCL logging knobs, generic HCCL
# connect timeout).  Anything that's specific to a workload variant lives in
# profile_env_block() below, NOT here.
common_env_block() {
    cat << CEOF
source ${CANN_HOME}/set_env.sh 2>/dev/null
ASCEND_ROOT="\$(dirname ${CANN_HOME})"
[[ -f "\${ASCEND_ROOT}/nnal/atb/set_env.sh" ]] && source "\${ASCEND_ROOT}/nnal/atb/set_env.sh"
eval "\$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate ${CONDA_ENV}
export PYTHONPATH="${AREAL_ROOT}:/root/torchstore:/root/monarch/python:\${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# -------------------------------------------------------------------------
# CANN / HCCL debug logging (profile-agnostic).
#
# CANN's slog (dlog) is shared across HCCL, HiXL, and other libraries
# under module id GE, so ASCEND_GLOBAL_LOG_LEVEL drives all of them.
# Levels: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR.  INFO (=1) dumps every
# aclnn op via kernel_utils.h:SplitDataAndPrint at ~100 MB/min, so we
# default to ERROR (=3) which still keeps real failures visible
# (HcclCommPrepare 0x13, RegisterMem 503900, etc. all log >= ERROR).
# Bump back to 1 only for active deep-dive repros.
#
# HCCL_DEBUG=INFO is a separate channel inside HCCL itself (not gated
# by ASCEND_GLOBAL_LOG_LEVEL) and surfaces the LinkEstablish state
# machine + Connect handshake — cheap enough to leave on by default.
#
# ASCEND_SLOG_PRINT_TO_STDOUT=1 streams slog into the proc's stdout
# so HCCL/HiXL errors land in /tmp/forge_worker.log on the host and
# in the driver's interleaved log for SSH-spawned actor procs.
# -------------------------------------------------------------------------
export ASCEND_GLOBAL_LOG_LEVEL="\${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export ASCEND_SLOG_PRINT_TO_STDOUT="\${ASCEND_SLOG_PRINT_TO_STDOUT:-1}"
export HCCL_DEBUG="\${HCCL_DEBUG:-INFO}"
# HCCL requires HCCL_CONNECT_TIMEOUT in [120, 7200] seconds — anything
# smaller is rejected with EI0001 at HcclCommInitClusterInfoMemConfig.
# Profile-agnostic safety floor.
export HCCL_CONNECT_TIMEOUT=120
CEOF
}

# Profile-specific env block.  This is the *only* place the two profiles
# diverge — keeping the diff narrow makes it obvious to reviewers what each
# profile actually opts into.
#
# Why split the env into profiles instead of a single union?
#   The HiXL-coexist env actively *degrades* HCCL (forces RoCE intra-host,
#   moves HCCL off port 16666) to give HiXL a clean slot.  Workloads that
#   don't use HiXL (B-mini, SFT, pretrain, eval) pay a ~10x intra-host
#   bandwidth penalty for nothing.  Splitting lets each workload pick the
#   minimal env it needs and keeps the GRPO-vetted combo intact for the
#   weight-sync path.
profile_env_block() {
    case "$PROFILE" in
        pure-training)
            cat << PEOF
# === Profile: pure-training =============================================
# Workloads: B-mini TorchTitan, SFT, pretrain, eval — anything that
# doesn't import HiXL/torchstore into the HCCL process.
#
# Intentionally MINIMAL: HCCL auto-selects HCCS for intra-host pairs
# (~400 GB/s) and RoCE for inter-host pairs (~25 GB/s).  No port range
# overrides, no torchstore staging pool, no MONARCH_HIXL_* vars.
# That's the whole point of this profile.
# ========================================================================
PEOF
            ;;
        hixl-coexist)
            cat << PEOF
# === Profile: hixl-coexist ==============================================
# Workloads: forge launch GRPO runs that mix HCCL collectives with HiXL
# single-sided RDMA + torchstore weight sync inside the same proc.
#
# Cross-node transport: HiXL must use RoCE (HCCS is intra-supernode
# only).  Each NPU has a RoCE port configured via hccn_tool
# (29.191.0.0/16 network), ~200 Gb/s (24 GB/s) cross-host bandwidth.
export MONARCH_HIXL_TRANSPORT=roce
# CANN reads HCCL_INTRA_ROCE_ENABLE once at library load time — Monarch
# setting it later from Rust is too late, so we set it here before
# python starts.  Forces HCCL intra-host onto RoCE so the link-type
# negotiation with HiXL's RoCE Connect doesn't deadlock.  Costs ~10x
# intra-host bandwidth vs. the default HCCS path — accepted tradeoff
# for the GRPO weight-sync correctness it unlocks.
export HCCL_INTRA_ROCE_ENABLE=1
# HiXL + HCCL coexistence in the same process: HiXL's Connect goes
# through HcclCommPrepare like a normal HCCL communicator.  When the
# training backend (FSDP, torchtitan) already created an HCCL process
# group on default port 16666, HiXL Connect collides and returns
# HcclCommPrepare ret=0x13.  Opening a wider port range gives the
# second communicator a free slot.  256 ports is the empirically
# validated minimum for FSDP 4-rank + HiXL sub-comm on the same NPU
# (verified on CANN 9.0 / 910B with ~18-19 GB/s steady-state).
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-60000-60255}"
# Let torchstore use its RDMA transport
# (MonarchRDMATransportBuffer → HiXL RoCE).
unset TORCHSTORE_RDMA_ENABLED
# HiXL needs NPU-resident 2MB-aligned memory.  Keep tensors on NPU —
# the default eager D2H path does .cpu() + malloc which is (a) not
# 2MB-aligned and (b) bypasses HiXL.  Users who really need CPU
# tensors should .cpu() explicitly before ts.put().
export TORCHSTORE_MONARCH_RDMA_EAGER_D2H=0
# Both *client* (trainer / generator) and *server* (storage) procs
# need to agree on the staging pool: torchstore's transport buffer
# allocation path (_empty_for_rdma) uses the pool as the destination
# on get, and MonarchRDMATransportBuffer.allocate uses the pool to
# stage any non-pool-resident source tensor into an aligned pool slot
# (critical -- it is the only thing that keeps FSDP state_dict
# tensors from triggering a fresh HiXL register_mem each put, which
# CANN RA rejects with ra_hdc_typical_mr ret=-13).  So keep the pool
# env here too.  The 8 GB per-proc footprint is acceptable because
# trainers only use it as transient staging per ts.put; it is not
# held for the optimizer state.
export TORCHSTORE_MONARCH_RDMA_STORAGE_DEVICE=npu:0
# Pool size: honor whatever the launcher shell set (so
# run_multinode.sh can propagate a lower value when NPU 0 is tight),
# default 8192 for the common Qwen3-0.6B case.
export TORCHSTORE_MONARCH_RDMA_POOL_MB=${TORCHSTORE_MONARCH_RDMA_POOL_MB:-8192}
# ========================================================================
PEOF
            ;;
    esac
}

# Generate worker start script for a given host IP.  Composed of:
#   1. shebang
#   2. common_env_block (CANN, conda, PYTHONPATH, generic HCCL)
#   3. profile_env_block (pure-training | hixl-coexist)
#   4. exec the Monarch worker loop
worker_script() {
    local HOST_IP="$1"
    echo "#!/bin/bash"
    echo "# Generated by worker_manager.sh — profile=${PROFILE}"
    common_env_block
    profile_env_block
    cat << WEOF
exec python3 -c "
from monarch.actor import run_worker_loop_forever
run_worker_loop_forever(address='tcp://${HOST_IP}:${WORKER_PORT}', ca='trust_all_connections')
"
WEOF
}

do_start() {
    echo "Starting Monarch workers (port=${WORKER_PORT}, profile=${PROFILE})..."
    for HOST in $(parse_hosts); do
        echo -n "  ${HOST}: "
        # Write worker script locally then scp to target
        local TMPSCRIPT=$(mktemp /tmp/forge_worker_XXXX.sh)
        worker_script "$HOST" > "$TMPSCRIPT"
        chmod +x "$TMPSCRIPT"
        scp -P ${SSH_PORT} -o StrictHostKeyChecking=no "$TMPSCRIPT" root@${HOST}:/tmp/forge_worker.sh 2>/dev/null
        rm -f "$TMPSCRIPT"
        $SSH_CMD root@${HOST} "nohup bash /tmp/forge_worker.sh > /tmp/forge_worker.log 2>&1 & echo pid=\$!"
    done

    echo "Waiting for workers to be ready..."
    local ALL_READY=true
    for HOST in $(parse_hosts); do
        local READY=false
        for i in $(seq 1 30); do
            if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('${HOST}', ${WORKER_PORT})); s.close()" 2>/dev/null; then
                echo "  ${HOST}:${WORKER_PORT} ready"
                READY=true
                break
            fi
            sleep 2
        done
        if ! $READY; then
            echo "  ${HOST}:${WORKER_PORT} FAILED to start"
            ALL_READY=false
        fi
    done

    if $ALL_READY; then
        echo "All workers ready."
    else
        echo "WARNING: Some workers failed to start."
        exit 1
    fi
}

do_stop() {
    echo "Stopping Monarch workers..."
    for HOST in $(parse_hosts); do
        echo -n "  ${HOST}: "
        $SSH_CMD root@${HOST} "
            pkill -9 -f run_worker_loop 2>/dev/null
            pkill -9 -f forge_worker 2>/dev/null
            pkill -9 -f monarch_bootstrap 2>/dev/null
            pkill -9 -f 'multiprocessing.spawn' 2>/dev/null
            pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null
            sleep 1
            echo stopped
        " 2>/dev/null || echo "unreachable"
    done
    sleep 2
    echo "Workers stopped."
}

do_status() {
    echo "Worker status (port=${WORKER_PORT}):"
    for HOST in $(parse_hosts); do
        echo -n "  ${HOST}:${WORKER_PORT} "
        if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('${HOST}', ${WORKER_PORT})); s.close()" 2>/dev/null; then
            echo "UP"
        else
            echo "DOWN"
        fi
    done
}

do_hosts_setup() {
    echo "Configuring /etc/hosts for cross-node hostname resolution..."
    # Collect hostname→IP mappings from all nodes
    local ENTRIES=""
    for HOST in $(parse_hosts); do
        local HNAME=$($SSH_CMD root@${HOST} "python3 -c 'import socket; print(socket.gethostname())'" 2>/dev/null)
        if [[ -n "$HNAME" ]]; then
            ENTRIES="${ENTRIES}${HOST} ${HNAME}\n"
            echo "  ${HOST} → ${HNAME}"
        fi
    done

    # Distribute to all nodes
    for HOST in $(parse_hosts); do
        echo -n "  Writing to ${HOST}: "
        echo -e "$ENTRIES" | $SSH_CMD root@${HOST} "
            while read LINE; do
                [[ -z \"\$LINE\" ]] && continue
                grep -qF \"\$LINE\" /etc/hosts 2>/dev/null || echo \"\$LINE\" >> /etc/hosts
            done
            echo done
        "
    done
    echo "Hostname resolution configured."
}

# Generate workers list for --bare-metal-workers CLI arg
do_workers_arg() {
    local WORKERS=""
    for HOST in $(parse_hosts); do
        [[ -n "$WORKERS" ]] && WORKERS="${WORKERS},"
        WORKERS="${WORKERS}tcp://${HOST}:${WORKER_PORT}"
    done
    echo "$WORKERS"
}

case "$ACTION" in
    start)       do_start ;;
    stop)        do_stop ;;
    status)      do_status ;;
    hosts-setup) do_hosts_setup ;;
    workers-arg) do_workers_arg ;;
    help|*)
        echo "Usage: $0 {start|stop|status|hosts-setup|workers-arg} --hostfile FILE [options]"
        echo ""
        echo "Commands:"
        echo "  start        Start Monarch workers on all nodes"
        echo "  stop         Stop workers on all nodes"
        echo "  status       Check worker status"
        echo "  hosts-setup  Configure /etc/hosts on all nodes"
        echo "  workers-arg  Print --bare-metal-workers argument"
        echo ""
        echo "Options:"
        echo "  --hostfile FILE      Node list (required)"
        echo "  --port PORT          Worker port (default: 22222)"
        echo "  --ssh-port PORT      SSH port (default: 36000)"
        echo "  --cann PATH          CANN install root (default: \$CANN_HOME)"
        echo "  --conda-env NAME     Conda env to activate (default: monarch_ascend)"
        echo "  --areal-root PATH    AReaL repo root for PYTHONPATH (default: \$AREAL_ROOT)"
        echo "  --profile NAME       Worker env preset (default: pure-training):"
        echo "                         pure-training  Minimal HCCL env, intra-host HCCS +"
        echo "                                        inter-host RoCE auto-selected. Use for"
        echo "                                        B-mini, SFT, pretrain, or anything"
        echo "                                        without HiXL/torchstore in-process."
        echo "                         hixl-coexist   Historical GRPO + weight-sync env."
        echo "                                        Forces HCCL intra-host onto RoCE so"
        echo "                                        HiXL/HCCL share NPU cleanly. Use for"
        echo "                                        \`forge launch\` GRPO runs."
        ;;
esac
