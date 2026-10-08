#!/bin/bash
#SBATCH --job-name=osc-pyscf214
#SBATCH --partition=qDEV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu/tests/manual/bench_osc_pyscf214_%j.out
#SBATCH --error=/data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu/tests/manual/bench_osc_pyscf214_%j.err

# GPU smoke against the worktree-isolated PySCF 2.14 venv (.venv).
# Does NOT touch ~/.local pyscf 2.10 used by live opsins jobs.

set -euo pipefail
cd /data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu
export PYTHONPATH=src:${PYTHONPATH:-}
PY="${PWD}/.venv/bin/python"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $PY"
nvidia-smi -L || true
"$PY" -c "import numpy, pyscf; print('numpy', numpy.__version__); print('pyscf', pyscf.__version__, pyscf.__file__)"
"$PY" -c "import gpu4pyscf, gpu4pyscf.scf, gpu4pyscf.dft; print('gpu4pyscf OK', getattr(gpu4pyscf,'__version__','?'), gpu4pyscf.__file__)"
"$PY" tests/manual/bench_osc_cpu_extract.py
