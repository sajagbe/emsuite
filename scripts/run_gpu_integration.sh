#!/usr/bin/env bash
# Run GPU channel integration tests on a SLURM GPU allocation.
#
# Usage:
#   ./scripts/run_gpu_integration.sh              # srun on qDEV with 1 GPU
#   ./scripts/run_gpu_integration.sh --local      # already on a GPU node
#   SLURM_PARTITION=qGPU48 ./scripts/run_gpu_integration.sh
#
# Environment overrides:
#   SLURM_PARTITION  (default: qDEV)
#   SLURM_GPUS       (default: 1)
#   SLURM_CPUS       (default: 8)
#   SLURM_MEM        (default: 32G)
#   SLURM_TIME       (default: 02:00:00)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PARTITION="${SLURM_PARTITION:-qDEV}"
GPUS="${SLURM_GPUS:-1}"
CPUS="${SLURM_CPUS:-8}"
MEM="${SLURM_MEM:-32G}"
TIME="${SLURM_TIME:-02:00:00}"

run_tests() {
    local run_id run_dir
    run_id="$(date -u +%Y%m%dT%H%M%SZ)"
    run_dir="${EMSUITE_INTEGRATION_RUN_DIR:-$REPO_ROOT/tests/integration_runs/gpu-${run_id}}"
    export EMSUITE_INTEGRATION_RUN_DIR="$run_dir"
    mkdir -p "$run_dir"

    echo "=== EMSuite GPU integration ==="
    echo "Host:     $(hostname)"
    echo "Run dir:  $run_dir"
    echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo ""

    export PYTHONUTF8=1
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"

    python -c "
from emsuite.core import GPU_AVAILABLE, check_gpu_info
n = check_gpu_info() or 0
if not GPU_AVAILABLE or n < 1:
    raise SystemExit(f'GPU preflight failed: GPU_AVAILABLE={GPU_AVAILABLE}, count={n}')
print(f'GPU preflight OK: {n} device(s)')
"

    local -a pytest_args=(-m gpu -v "tests/integration/test_gpu_channels.py")
    if [[ "${1:-}" == "--" ]]; then
        shift
        pytest_args+=("$@")
    elif [[ $# -gt 0 ]]; then
        pytest_args+=("$@")
    fi

    python -m pytest "${pytest_args[@]}" \
        --junitxml="$run_dir/junit.xml" \
        --log-file="$run_dir/pytest.log" \
        --log-file-level=INFO

    echo ""
    echo "Artifacts: $run_dir"
}

if [[ "${1:-}" == "--local" ]]; then
    shift
    run_tests "$@"
    exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]] && [[ "${SLURM_GPUS_ON_NODE:-0}" -gt 0 || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    run_tests "$@"
    exit 0
fi

echo "Allocating GPU via srun (partition=$PARTITION, gres=gpu:$GPUS) ..."
exec srun \
    --partition="$PARTITION" \
    --gres="gpu:${GPUS}" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --export=ALL \
    bash "$0" --local "$@"
