#!/usr/bin/env bash
# GPU vs CPU comparison of the RESP port. Submit from this folder: sbatch run.slurm
#SBATCH --job-name=resp-cmp
#SBATCH --partition=qGPU24
#SBATCH --account=CHEM9C4
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --exclude=acidsgcn007,acidsgcn001

set -euo pipefail
HERE="${SLURM_SUBMIT_DIR:?submit from tests/probes/resp_cpu}"
[[ -f "$HERE/resp_cpu.py" ]] || { echo "ERROR: $HERE is not resp_cpu/" >&2; exit 1; }
cd "$HERE"
BIN="$(cd ../../.. && pwd)/.venv/bin"
export PYTHONUTF8=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
module purge || true
module load cuda/12.1 || true
echo "host=$(hostname) job=${SLURM_JOB_ID} date=$(date -u -Iseconds)"
nvidia-smi -L || true
"$BIN/python" compare_gpu.py water LF SB rPSB
echo "DONE $(date -u -Iseconds)"
