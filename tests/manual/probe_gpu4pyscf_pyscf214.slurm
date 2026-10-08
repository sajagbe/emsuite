#!/bin/bash
#SBATCH --job-name=probe-g4p-214
#SBATCH --partition=qDEV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/manual/probe_gpu4pyscf_pyscf214_%j.out
#SBATCH --error=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/manual/probe_gpu4pyscf_pyscf214_%j.err
set -euo pipefail
cd /data/PHO_WORK/sajagbe2/packages/emsuite
export PYTHONPATH=src:${PYTHONPATH:-}
PY="${PWD}/.venv/bin/python"
nvidia-smi -L || true
"$PY" tests/manual/probe_gpu4pyscf_pyscf214.py
