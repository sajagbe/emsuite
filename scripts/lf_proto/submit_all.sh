#!/usr/bin/env bash
# Submit the full LF Proto batch (prep + 14 SLURM jobs).
#
# Usage (from lf-proto work directory):
#   export LF_PROTO_ROOT=/path/to/lf-proto
#   bash /path/to/emsuite/scripts/lf_proto/submit_all.sh
#
# Or copy/symlink this script into LF_PROTO_ROOT and run ./submit_all.sh

set -euo pipefail

LF_PROTO_ROOT="${LF_PROTO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EMSUITE_ROOT="${EMSUITE_ROOT:-$(cd "$LF_PROTO_ROOT/../../../packages/emsuite" 2>/dev/null && pwd || cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPTS_DIR="${LF_PROTO_SCRIPTS:-$EMSUITE_ROOT/scripts/lf_proto}"
GRO_ROOT="${LF_PROTO_GRO_ROOT:-$LF_PROTO_ROOT/../Selected100-L1vL2}"
LF_XYZ_SRC="${LF_PROTO_LF_XYZ:-$LF_PROTO_ROOT/../ChargeUpdates/LumiflavinRESP2025/LF/LF.xyz}"

export LF_PROTO_ROOT
export SLURM_GPUS="${SLURM_GPUS:-4}"
export SLURM_CPUS="${SLURM_CPUS:-32}"
export SLURM_PARTITION_GPU="${SLURM_PARTITION_GPU:-qGPU48}"
export SLURM_PARTITION_CPU="${SLURM_PARTITION_CPU:-qCPU120}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-CHEM9C4}"
export SLURM_TIME_GPU="${SLURM_TIME_GPU:-47:59:00}"
export SLURM_TIME_CPU="${SLURM_TIME_CPU:-120:00:00}"
export SLURM_CPUS_GPU="${SLURM_CPUS_GPU:-32}"
export SLURM_CPUS_CPU="${SLURM_CPUS_CPU:-16}"

SYSTEMS=(AT1 AT2 AS1 AS2 CR1 CR2)
GRO_NAMES=(AT1Sel AT2Sel AS1Sel AS2Sel CR1Sel CR2Sel)

echo "LF_PROTO_ROOT=$LF_PROTO_ROOT"
echo "EMSUITE_ROOT=$EMSUITE_ROOT"
echo "SLURM_PARTITION_GPU=$SLURM_PARTITION_GPU"
echo "SLURM_PARTITION_CPU=$SLURM_PARTITION_CPU"
echo "SLURM_ACCOUNT=$SLURM_ACCOUNT"
echo "SLURM_TIME_GPU=$SLURM_TIME_GPU"
echo "SLURM_TIME_CPU=$SLURM_TIME_CPU"
echo "SLURM_GPUS=$SLURM_GPUS"
echo "SLURM_CPUS=$SLURM_CPUS"
echo "SLURM_CPUS_GPU=$SLURM_CPUS_GPU"
echo "SLURM_CPUS_CPU=$SLURM_CPUS_CPU"

mkdir -p "$LF_PROTO_ROOT"/{prep,lf-homogeneous/{singlet,triplet},lov-protein/frames,lov-protein/potential,lov-protein/coupled}

# Symlink LF geometry
ln -sfn "$LF_XYZ_SRC" "$LF_PROTO_ROOT/prep/LF.xyz"
ln -sfn "$LF_XYZ_SRC" "$LF_PROTO_ROOT/lf-homogeneous/singlet/LF.xyz"
ln -sfn "$LF_XYZ_SRC" "$LF_PROTO_ROOT/lf-homogeneous/triplet/LF.xyz"

# Link helper scripts into prep
ln -sfn "$SCRIPTS_DIR/prepare_frames.py" "$LF_PROTO_ROOT/prep/prepare_frames.py"
ln -sfn "$SCRIPTS_DIR/patch_num_procs.py" "$LF_PROTO_ROOT/prep/patch_num_procs.py"

# Extract protein frames
for i in "${!SYSTEMS[@]}"; do
    sys="${SYSTEMS[$i]}"
    gro="${GRO_ROOT}/${GRO_NAMES[$i]}.gro"
    out="$LF_PROTO_ROOT/lov-protein/frames/$sys"
    python3 "$SCRIPTS_DIR/prepare_frames.py" --gro "$gro" --out-dir "$out" --system "$sys"
done

# Surface prep (local or sbatch)
if [[ -f "$LF_PROTO_ROOT/prep/LF.surf" ]]; then
    echo "prep/LF.surf already exists — skipping surface generation"
else
    echo "Running surface prep..."
    (cd "$LF_PROTO_ROOT/prep" && bash run_surface.sh) || echo "WARNING: surface prep failed or emsuite unavailable"
fi

if [[ -f "$LF_PROTO_ROOT/prep/LF.surf" ]]; then
    ln -sfn "$LF_PROTO_ROOT/prep/LF.surf" "$LF_PROTO_ROOT/lf-homogeneous/singlet/LF.surf"
    ln -sfn "$LF_PROTO_ROOT/prep/LF.surf" "$LF_PROTO_ROOT/lf-homogeneous/triplet/LF.surf"
fi

# Submit jobs
JOB_IDS=()
submit() {
    local script="$1"
    local id
    id=$(sbatch "$script" | awk '{print $4}')
    JOB_IDS+=("$id")
    echo "Submitted $script -> job $id"
}

submit "$LF_PROTO_ROOT/lf-homogeneous/singlet/run.slurm"
submit "$LF_PROTO_ROOT/lf-homogeneous/triplet/run.slurm"
for sys in "${SYSTEMS[@]}"; do
    submit "$LF_PROTO_ROOT/lov-protein/potential/$sys/run.slurm"
    submit "$LF_PROTO_ROOT/lov-protein/coupled/$sys/run.slurm"
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"
