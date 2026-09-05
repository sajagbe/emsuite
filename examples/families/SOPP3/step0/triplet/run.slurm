#!/usr/bin/env bash
#SBATCH --job-name=sog-s0-comb-SOPP3-T
#SBATCH --partition=qGPU48
#SBATCH --account=CHEM9C4
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=47:59:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=acidsgcn007,acidsgcn001

set -euo pipefail
export PYTHONUTF8=1
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PATH="${HOME}/.local/bin:${PATH}"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CALC_DIR="$PWD"
LF_PROTO_ROOT="${LF_PROTO_ROOT:-$(cd "$CALC_DIR/../../../.." && pwd)}"
PATCH="$LF_PROTO_ROOT/scripts/patch_num_procs.py"
[[ -f "$PATCH" ]] || PATCH="$LF_PROTO_ROOT/prep/patch_num_procs.py"
python -c "import gpu4pyscf.dft; from emsuite.core import check_gpu_info; n = check_gpu_info() or 0; assert n >= 1, f'GPU preflight failed: {n}'"
python "$PATCH" coupled.in
emsuite -c coupled.in
