#!/bin/bash
#SBATCH --job-name=osc-cpu-bench
#SBATCH --partition=qDEV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu/tests/manual/bench_osc_cpu_extract_%j.out
#SBATCH --error=/data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu/tests/manual/bench_osc_cpu_extract_%j.err

set -euo pipefail
cd /data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu
export PYTHONPATH=src:${PYTHONPATH:-}
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
python tests/manual/bench_osc_cpu_extract.py
