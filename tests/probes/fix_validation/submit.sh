#!/usr/bin/env bash
# Submit all fix-validation tasks (GPU first) and move them to the top of the user's queue.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
n() { grep -c . "$1"; }
ids=()
sub() {  # name tasks_file device sbatch-args...
    local name=$1 file=$2 dev=$3; shift 3
    local id
    id=$(sbatch --parsable --job-name="$name" --array=0-$(( $(n "$file") - 1 )) \
         --export=ALL,TASKS_FILE="$file",DEVICE="$dev" "$@" run_task.slurm)
    echo "$name: $id ($(n "$file") tasks)"; ids+=("$id")
}
sub fv-gpu4 tasks_gpu4.txt gpu --partition=qGPU24 --gres=gpu:V100:4 --cpus-per-task=32 --mem=64G --time=12:00:00
sub fv-gpu1 tasks_gpu1.txt gpu --partition=qGPU24 --gres=gpu:V100:1 --cpus-per-task=8 --mem=48G --time=12:00:00
sub fv-cpu  tasks_cpu.txt  cpu --partition=qCPU24 --cpus-per-task=32 --mem=96G --time=23:59:00
printf '%s\n' "${ids[@]}" > SUBMITTED.txt
for id in "${ids[@]}"; do scontrol top "$id" 2>&1 || echo "note: scontrol top $id not permitted"; done
