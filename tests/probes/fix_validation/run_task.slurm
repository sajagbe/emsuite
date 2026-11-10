#!/usr/bin/env bash
# One fix-validation task per array index. Submit via ./submit.sh (sets partition/gres).
# Task file lines:  tune <case_dir> | baseline <case_dir> <N> | pytest <test_path>
#SBATCH --account=CHEM9C4
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --exclude=acidsgcn007,acidsgcn001

set -euo pipefail
HERE="${SLURM_SUBMIT_DIR:?submit from tests/probes/fix_validation via ./submit.sh}"
[[ -f "$HERE/prepare.py" ]] || { echo "ERROR: $HERE is not fix_validation/" >&2; exit 1; }
ROOT="$(cd "$HERE/../../.." && pwd)"
BIN="$ROOT/.venv/bin"
TASKS="$HERE/${TASKS_FILE:?TASKS_FILE unset}"
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")"
read -r KIND ARG1 ARG2 <<< "$LINE"
DEVICE="${DEVICE:?DEVICE unset (gpu|cpu)}"

export PYTHONUTF8=1 LC_ALL="${LC_ALL:-en_US.UTF-8}" RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PATH="$BIN:$PATH"
module purge || true
if [[ "$DEVICE" == gpu ]]; then module load cuda/12.1 || true; fi
if [[ "$DEVICE" == cpu ]]; then
    export PYTHONPATH="$HERE/cpu_only_shim${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=""
fi

echo "task=$SLURM_ARRAY_TASK_ID line='$LINE' device=$DEVICE host=$(hostname) date=$(date -u -Iseconds)"
echo "git_head=$(git -C "$ROOT" rev-parse HEAD) src_diff_sha1=$(git -C "$ROOT" diff HEAD -- src | sha1sum | cut -c1-12)"
"$BIN/python" - "$DEVICE" <<'PY'
import sys
import pyscf
from emsuite import core
dev = sys.argv[1]
n = core.check_gpu_info() or 0
print(f"pyscf={pyscf.__version__} CUPY_AVAILABLE={core.CUPY_AVAILABLE} gpus={n}")
if dev == "gpu":
    import gpu4pyscf
    print(f"gpu4pyscf={gpu4pyscf.__version__}")
    assert n >= 1, "GPU preflight failed"
else:
    assert not core.CUPY_AVAILABLE and n == 0, "CPU task can see CuPy/GPU"
PY

case "$KIND" in
    tune)
        cd "$HERE/$ARG1"
        if compgen -G "results_*" > /dev/null; then echo "skip: results already present"; exit 0; fi
        "$BIN/emsuite" -t tuning.in ;;
    baseline)
        cd "$HERE/$ARG1"
        rm -f baseline_repeats.jsonl
        "$BIN/python" "$HERE/baseline_repeat.py" . "$ARG2" ;;
    pytest)
        cd "$ROOT"
        "$BIN/python" -m pytest -q -m slow "$ARG1" ;;
    *) echo "unknown task kind: $KIND" >&2; exit 2 ;;
esac
echo "DONE $(date -u -Iseconds)"
