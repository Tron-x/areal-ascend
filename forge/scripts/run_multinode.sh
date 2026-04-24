#!/usr/bin/env bash
# =============================================================================
# Forge — Multi-node GRPO training (hostfile-driven)
#
# NOTE (unified-entry refactor, 2026-04-23):
#
#   This script is now a *thin wrapper* around ``python -m forge launch``.
#   New code paths should call ``python -m forge launch`` directly:
#
#       python -m forge launch examples/math/gsm8k_grpo_npu.yaml \\
#           --steps 3 --backend titan --model-name qwen3 --model-flavor 0.6B \\
#           --model /path/to/model -- \\
#           allocation_mode=vllm:d1p1t1+d4p1t1  gconfig.max_new_tokens=128
#
#   (the algo YAML picks a cluster preset under
#   ``forge/configs/clusters/`` via its ``launcher_preset:`` key.)
#
#   The wrapper translates the historical CLI surface of this script
#   (``--hostfile``, ``--model``, ``--steps``, ``--backend``, ...) and
#   the pre-refactor FORGE_* env vars (FORGE_WEIGHT_SYNC,
#   FORGE_SHARD_PUBLISH, ...) into the equivalent ``forge launch``
#   invocation.  Existing smoke scripts, CI pipelines, and muscle-
#   memory invocations keep working.
#
#   The Python CLI is the authoritative entry point going forward:
#     * YAML is the single source of truth (launcher.weight_sync.*);
#     * env vars still work as overrides but now emit deprecation
#       warnings via ``forge.engines.weight_sync._config_resolver``;
#     * pre-flight checks, SSH bookkeeping, and cleanup-on-exit all
#       live in Python and get exercised uniformly by every caller.
#
#   See ``forge/docs/env_to_yaml_mapping.md`` for the YAML schema and
#   ``forge/cli/launch.py`` for the CLI itself.
#
# Usage (unchanged):
#   bash forge/scripts/run_multinode.sh --hostfile forge/configs/hostfile.txt
#   bash forge/scripts/run_multinode.sh --hostfile ... --steps 5
#   bash forge/scripts/run_multinode.sh --hostfile ... --model /path/to/model
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --------------- defaults ---------------
HOSTFILE=""
MODEL_PATH=""
TRAIN_STEPS=""
WORKER_PORT=22222
SSH_PORT=36000
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0-beta.1}"
BACKEND=""
MODEL_NAME=""
MODEL_FLAVOR=""
EXTRA_ARGS=()
# FORGE_CONFIG overrides the YAML path (backward compat: same env name
# as the pre-refactor script).
FORGE_CONFIG_PATH="${FORGE_CONFIG:-forge/configs/clusters/2node_colocated.yaml}"

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
        --forge-config)  FORGE_CONFIG_PATH="$2"; shift 2 ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$HOSTFILE" ]]; then
    echo "Error: --hostfile is required" >&2
    echo "Usage: $0 --hostfile forge/configs/hostfile.txt [--steps N] [--model PATH]" >&2
    exit 1
fi

# Pick the Python interpreter from the conda env if not already active.
# ``forge launch`` needs the forge package on sys.path.
if command -v python >/dev/null 2>&1; then
    PYBIN="python"
else
    PYBIN="/root/miniconda3/envs/monarch_ascend/bin/python"
fi

# Translate the historical CLI surface to ``forge launch`` args.
LAUNCH_ARGS=(
    launch
    "$FORGE_CONFIG_PATH"
    --hostfile "$HOSTFILE"
    --worker-port "$WORKER_PORT"
    --ssh-port "$SSH_PORT"
    --cann "$CANN_HOME"
)
[[ -n "$TRAIN_STEPS" ]] && LAUNCH_ARGS+=(--steps "$TRAIN_STEPS")
[[ -n "$MODEL_PATH"  ]] && LAUNCH_ARGS+=(--model "$MODEL_PATH")
[[ -n "$BACKEND"     ]] && LAUNCH_ARGS+=(--backend "$BACKEND")
[[ -n "$MODEL_NAME"  ]] && LAUNCH_ARGS+=(--model-name "$MODEL_NAME")
[[ -n "$MODEL_FLAVOR" ]] && LAUNCH_ARGS+=(--model-flavor "$MODEL_FLAVOR")

# Any remaining flags get forwarded to forge.apps.grpo as OmegaConf
# overrides (e.g. ``allocation_mode=...`` / ``gconfig.max_new_tokens=128``).
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    LAUNCH_ARGS+=(--)
    LAUNCH_ARGS+=("${EXTRA_ARGS[@]}")
fi

# Historical env vars still work; ``forge launch`` forwards them to the
# driver host and ``forge.engines.weight_sync._config_resolver`` picks
# them up as overrides (with a deprecation warning pointing at the YAML
# field).  Nothing to do here beyond invoking Python with the same
# environment we inherited.

cd "$AREAL_ROOT"
exec "$PYBIN" -m forge "${LAUNCH_ARGS[@]}"
