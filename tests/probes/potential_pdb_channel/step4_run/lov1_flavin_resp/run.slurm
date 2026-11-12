#!/bin/bash
#SBATCH --job-name=lov1-flavin-resp
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/probes/potential_pdb_channel/step4_run/lov1_flavin_resp/slurm-%j.out
# One B3LYP/3-21G RESP calculation on the 50-atom flavin, then APBS. Approved by the user 2026-11-12.
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=/data/PHO_WORK/sajagbe2/packages/emsuite/src
cd /data/PHO_WORK/sajagbe2/packages/emsuite/tests/probes/potential_pdb_channel/step4_run/lov1_flavin_resp
/data/PHO_WORK/sajagbe2/packages/emsuite/.worktrees/ie-ea-check/.venv/bin/python -u run.py
