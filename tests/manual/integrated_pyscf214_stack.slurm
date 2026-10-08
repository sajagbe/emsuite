#!/bin/bash
#SBATCH --job-name=osc-stack-int
#SBATCH --partition=qDEV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu/tests/manual/integrated_pyscf214_stack_%j.out
#SBATCH --error=/data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu/tests/manual/integrated_pyscf214_stack_%j.err

set -euo pipefail
cd /data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTHONUTF8=1

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: worktree .venv missing — create with PYSCF214_ENV_NOTES.md" >&2
  exit 2
fi
PY=.venv/bin/python

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $PY"
nvidia-smi -L || true
"$PY" -c "import numpy, pyscf, gpu4pyscf; print('numpy', numpy.__version__); print('pyscf', pyscf.__version__, pyscf.__file__); print('gpu4pyscf', getattr(gpu4pyscf,'__version__','?'), gpu4pyscf.__file__)"
"$PY" tests/manual/integrated_pyscf214_stack.py
