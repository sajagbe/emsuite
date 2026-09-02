#!/usr/bin/env bash
# Water SMILES → GPU tuning via CLI (and optional Python API).
#
# Usage:
#   ./run_gpu.sh              # CLI smoke (surface.in + tuning_smoke.in)
#   ./run_gpu.sh --full       # CLI full property set (tuning.in)
#   ./run_gpu.sh --api        # Python API smoke (run_api.py)
#   ./run_gpu.sh --local      # skip srun (already on GPU node)
#
# Environment:
#   SLURM_PARTITION  (default: qDEV)
#   SLURM_GPUS       (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

MODE=cli
FULL=0
LOCAL=0

for arg in "$@"; do
    case "$arg" in
        --api) MODE=api ;;
        --full) FULL=1 ;;
        --local) LOCAL=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

TUNING_IN=tuning_smoke.in
[[ "$FULL" -eq 1 ]] && TUNING_IN=tuning.in

run_job() {
    export PYTHONUTF8=1
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"
    export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

    echo "=== Water GPU tuning ==="
    echo "Host:   $(hostname)"
    echo "Mode:   $MODE"
    echo "Input:  $TUNING_IN"
    echo ""

    python -c "
from emsuite.core import check_gpu_info
import gpu4pyscf.dft  # noqa: F401
n = check_gpu_info() or 0
if n < 1:
    raise SystemExit(f'GPU preflight failed: {n} device(s)')
print(f'GPU preflight OK: {n} device(s)')
"

    if [[ "$MODE" == "api" ]]; then
        if [[ "$FULL" -eq 1 ]]; then
            python run_api.py --full
        else
            python run_api.py
        fi
    else
        emsuite -s surface.in
        emsuite -t "$TUNING_IN"
    fi

    echo ""
    echo "Done. Results in: $(ls -d results_Water_* 2>/dev/null | tail -1 || echo '(see cwd)')"
}

if [[ "$LOCAL" -eq 1 ]] || [[ -n "${SLURM_JOB_ID:-}" ]]; then
    run_job
    exit 0
fi

PARTITION="${SLURM_PARTITION:-qDEV}"
GPUS="${SLURM_GPUS:-1}"
exec srun \
    --partition="$PARTITION" \
    --gres="gpu:${GPUS}" \
    --cpus-per-task=8 \
    --mem=32G \
    --time=04:00:00 \
    --export=ALL \
    bash "$0" --local "$@"
