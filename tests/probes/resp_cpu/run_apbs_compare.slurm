#!/bin/bash
# Radius comparison for APBS surface maps (see apbs_radii_compare.py). CPU only.
#SBATCH --job-name=apbs-radii
#SBATCH --partition=qCPU24
#SBATCH --account=CHEM9C4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=apbs_compare/slurm-%j.out
#SBATCH --error=apbs_compare/slurm-%j.err

set -euo pipefail
HERE="${SLURM_SUBMIT_DIR:?submit from tests/probes/resp_cpu}"
[[ -f "$HERE/apbs_radii_compare.py" ]] || { echo "ERROR: $HERE is not resp_cpu/" >&2; exit 1; }
cd "$HERE"
BIN="$(cd ../../.. && pwd)/.venv/bin"
export PATH="$BIN:$PATH" PYTHONUTF8=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
echo "host=$(hostname) job=${SLURM_JOB_ID} date=$(date -u -Iseconds)"
"$BIN/python" apbs_radii_compare.py LF SB rPSB
echo "DONE $(date -u -Iseconds)"
