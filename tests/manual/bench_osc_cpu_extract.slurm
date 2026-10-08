#!/bin/bash
#SBATCH --job-name=osc-cpu-bench
#SBATCH --partition=qDEV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/manual/bench_osc_cpu_extract_%j.out
#SBATCH --error=/data/PHO_WORK/sajagbe2/packages/emsuite/tests/manual/bench_osc_cpu_extract_%j.err

set -euo pipefail
cd /data/PHO_WORK/sajagbe2/packages/emsuite
export PYTHONPATH=src:${PYTHONPATH:-}
# Prefer repo-local .venv (PySCF 2.14) when present; else system python.
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $PY"
nvidia-smi -L || true
"$PY" -c "import numpy, pyscf; print('numpy', numpy.__version__); print('pyscf', pyscf.__version__, pyscf.__file__)"
"$PY" tests/manual/bench_osc_cpu_extract.py
