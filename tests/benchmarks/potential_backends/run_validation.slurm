#!/bin/bash
#SBATCH --job-name=resp-default-check
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/benchmarks/potential_backends/runs/slurm-%j.out

# Prepared validation scope; requires approval before sbatch.
set -euo pipefail
cd /data/PHO_WORK/sajagbe2/packages/emsuite
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
"$PWD/.worktrees/ie-ea-check/.venv/bin/python" -u \
  tests/benchmarks/potential_backends/validate_gpu.py \
  "tests/benchmarks/potential_backends/runs/job-$SLURM_JOB_ID"
