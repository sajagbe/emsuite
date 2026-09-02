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
if [[ -z "${EMSUITE_ROOT:-}" ]]; then
    if [[ -d "$LF_PROTO_ROOT/../../../packages/emsuite" ]]; then
        EMSUITE_ROOT="$(cd "$LF_PROTO_ROOT/../../../packages/emsuite" && pwd)"
    else
        EMSUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
SCRIPTS_DIR="${LF_PROTO_SCRIPTS:-$EMSUITE_ROOT/scripts/lf_proto}"
GRO_ROOT="${LF_PROTO_GRO_ROOT:-$LF_PROTO_ROOT/../Selected100-L1vL2}"
LF_XYZ_SRC="${LF_PROTO_LF_XYZ:-$LF_PROTO_ROOT/../ChargeUpdates/LumiflavinRESP2025/LF/LF.xyz}"

export LF_PROTO_ROOT
export SLURM_CPUS="${SLURM_CPUS:-16}"

_wait_slurm_job() {
    local job_id="$1"
    local label="${2:-job}"
    while squeue -j "$job_id" -h 2>/dev/null | grep -q .; do
        sleep 15
    done
    local state batch_flag
    state="$(sacct -j "$job_id" --format=State -n -P 2>/dev/null | head -1 | cut -d'|' -f1)"
    state="${state:-UNKNOWN}"
    batch_flag="$(sacct -j "$job_id" --format=BatchFlag -n -P 2>/dev/null | head -1 | cut -d'|' -f1)"
    echo "$label job $job_id final state: $state (BatchFlag=${batch_flag:-?})"
    if [[ "${batch_flag:-}" != "1" ]]; then
        echo "ERROR: $label job $job_id was not submitted via sbatch" >&2
        return 1
    fi
    if [[ "$state" != "COMPLETED" ]]; then
        echo "ERROR: $label job $job_id did not complete successfully" >&2
        return 1
    fi
}

SYSTEMS=(AT1 AT2 AS1 AS2 CR1 CR2)
GRO_NAMES=(AT1Sel AT2Sel AS1Sel AS2Sel CR1Sel CR2Sel)

echo "LF_PROTO_ROOT=$LF_PROTO_ROOT"
echo "EMSUITE_ROOT=$EMSUITE_ROOT"
echo "SLURM_CPUS=$SLURM_CPUS"

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

# Surface prep (sbatch only — qCPU120)
if [[ -f "$LF_PROTO_ROOT/prep/LF.surf" ]]; then
    echo "prep/LF.surf already exists — skipping surface generation"
elif [[ ! -f "$LF_PROTO_ROOT/prep/run_surface.slurm" ]]; then
    echo "ERROR: missing $LF_PROTO_ROOT/prep/run_surface.slurm (re-run bootstrap_workdir.sh)" >&2
    exit 1
else
    echo "Submitting surface prep via sbatch..."
    surface_job_id="$(cd "$LF_PROTO_ROOT/prep" && sbatch --parsable run_surface.slurm)"
    if [[ ! "$surface_job_id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: sbatch failed for prep/run_surface.slurm: $surface_job_id" >&2
        exit 1
    fi
    echo "  surface prep job_id=$surface_job_id"
    _wait_slurm_job "$surface_job_id" "surface prep"
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
