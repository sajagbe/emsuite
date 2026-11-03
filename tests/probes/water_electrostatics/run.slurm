#!/usr/bin/env bash
#SBATCH --job-name=we-h2o
#SBATCH --partition=qDEV
#SBATCH --account=CHEM9C4
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --exclude=acidsgcn007,acidsgcn001

set -euo pipefail
ROOT="/data/PHO_WORK/sajagbe2/packages/emsuite"
BIN="$ROOT/.venv/bin"
# Under sbatch, BASH_SOURCE points at a copy in /var/spool/slurmd/job<ID>/,
# so use the submit directory (submit from this folder: `sbatch run.slurm`).
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -f "$HERE/prepare.py" ]]; then
    echo "ERROR: $HERE has no prepare.py; cd to tests/probes/water_electrostatics and sbatch run.slurm there" >&2
    exit 1
fi
cd "$HERE"

export PYTHONUTF8=1
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PATH="$BIN:$PATH"
module purge || true
module load cuda/12.1 || true

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} date=$(date -u -Iseconds)"
nvidia-smi -L || true
"$BIN/python" -c "
from emsuite.core import check_gpu_info
import gpu4pyscf.dft  # noqa: F401
n = check_gpu_info() or 0
if n < 1:
    raise SystemExit(f'GPU preflight failed: {n}')
print(f'GPU preflight OK: {n} device(s)')
"

if [[ ! -f jobs.txt ]]; then
    "$BIN/python" prepare.py
fi

n=0
while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    n=$((n + 1))
    echo "=== [$n] $rel $(date -u -Iseconds) ==="
    (
        cd "$HERE/$rel"
        if python - "$HERE/$rel" <<'PY'
import csv, sys
from pathlib import Path
ok = False
for p in Path(sys.argv[1]).glob("results_Water_*/Water_tuning_summary.csv"):
    with p.open() as f:
        rows = list(csv.DictReader(f))
    if rows and any(k.strip() == "s1_exe_effect" for k in rows[0]):
        ok = True
        break
raise SystemExit(0 if ok else 1)
PY
        then
            echo "skip, successful results already present"
            exit 0
        fi
        "$BIN/emsuite" -t tuning.in
    )
done < jobs.txt

echo "DONE $(date -u -Iseconds)"
"$BIN/python" analyze.py
