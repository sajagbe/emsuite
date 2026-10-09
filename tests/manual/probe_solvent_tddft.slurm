#!/bin/bash
#SBATCH --job-name=solv-td-probe
#SBATCH --partition=qDEV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/manual/probe_solvent_tddft_%j.out
#SBATCH --error=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/manual/probe_solvent_tddft_%j.err

set -euo pipefail
cd /data/PHO_WORK/sajagbe2/packages/emsuite
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTHONUTF8=1
PY=python
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
fi
echo "Python: $PY"
nvidia-smi -L || true
"$PY" tests/manual/probe_solvent_tddft.py
