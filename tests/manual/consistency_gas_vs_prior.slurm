#!/bin/bash
#SBATCH --job-name=consist-gas
#SBATCH --partition=qDEV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/manual/consistency_gas_vs_prior_%j.out
#SBATCH --error=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/manual/consistency_gas_vs_prior_%j.err
set -euo pipefail
cd /data/PHO_WORK/sajagbe2/packages/emsuite
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTHONUTF8=1
PY=python
[[ -x .venv/bin/python ]] && PY=.venv/bin/python
echo "Python: $PY"
nvidia-smi -L || true
"$PY" -c "import pyscf,emsuite; print('pyscf',pyscf.__version__); print('emsuite',emsuite.__version__, emsuite.__file__)"
"$PY" tests/manual/consistency_gas_vs_prior.py
