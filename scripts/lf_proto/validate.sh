#!/usr/bin/env bash
# Validate lf-proto inputs, paths, and SLURM scripts (no production submission).
#
# Usage:
#   ./validate.sh              # parse + path checks + sbatch --test-only
#   ./validate.sh --run-smoke  # also submit short GPU smoke jobs

set -uo pipefail

_wait_slurm_job() {
    local job_id="$1"
    local label="${2:-smoke}"
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
    case "$state" in
        COMPLETED) return 0 ;;
        *)
            echo "ERROR: $label job $job_id did not complete successfully (state=$state)" >&2
            return 1
            ;;
    esac
}

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$_SCRIPTS_DIR")" == scripts ]]; then
    LF_PROTO_ROOT="${LF_PROTO_ROOT:-$(cd "$_SCRIPTS_DIR/.." && pwd)}"
else
    LF_PROTO_ROOT="${LF_PROTO_ROOT:-$_SCRIPTS_DIR}"
fi
SCRIPTS_DIR="${LF_PROTO_SCRIPTS:-$_SCRIPTS_DIR}"
RUN_SMOKE=0

for arg in "$@"; do
    case "$arg" in
        --run-smoke) RUN_SMOKE=1 ;;
        -h|--help)
            echo "Usage: $0 [--run-smoke]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

export LF_PROTO_ROOT

echo "=== lf-proto validation ==="
echo "LF_PROTO_ROOT=$LF_PROTO_ROOT"
echo ""

echo "--- Step 1: validate_inputs.py ---"
python3 "$SCRIPTS_DIR/validate_inputs.py" "$LF_PROTO_ROOT"
INPUT_STATUS=$?
echo ""

echo "--- Step 2: check_paths.py ---"
python3 "$SCRIPTS_DIR/check_paths.py" "$LF_PROTO_ROOT"
PATH_STATUS=$?
echo ""

echo "--- Step 3: sbatch --test-only on SLURM scripts ---"
SLURM_STATUS=0
RUN_SLURMS=(
    "$LF_PROTO_ROOT/prep/run_surface.slurm"
    "$LF_PROTO_ROOT/lf-homogeneous/singlet/run.slurm"
    "$LF_PROTO_ROOT/lf-homogeneous/triplet/run.slurm"
)
for sys in AT1 AT2 AS1 AS2 CR1 CR2; do
    RUN_SLURMS+=(
        "$LF_PROTO_ROOT/lov-protein/potential/$sys/run.slurm"
        "$LF_PROTO_ROOT/lov-protein/coupled/$sys/run.slurm"
    )
done

if command -v sbatch >/dev/null 2>&1; then
    for script in "${RUN_SLURMS[@]}"; do
        if [[ -f "$script" ]]; then
            calc_dir="$(dirname "$script")"
            echo "sbatch --test-only $(basename "$calc_dir")/$(basename "$script")"
            if ! (cd "$calc_dir" && sbatch --test-only "$(basename "$script")"); then
                SLURM_STATUS=1
            fi
        else
            echo "WARN: missing $script"
            SLURM_STATUS=1
        fi
    done
else
    echo "WARN: sbatch not available — skipping SLURM script validation"
    SLURM_STATUS=0
fi
echo ""

SMOKE_STATUS=0
if [[ "$RUN_SMOKE" -eq 1 ]]; then
    echo "--- Step 4: smoke jobs via sbatch (--run-smoke) ---"
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: --run-smoke requires sbatch (no srun/interactive fallback)" >&2
        exit 1
    fi
    if ! command -v sacct >/dev/null 2>&1; then
        echo "ERROR: --run-smoke requires sacct to verify batch submission" >&2
        exit 1
    fi
    echo "Smoke jobs are submitted with sbatch only (never srun or login-node emsuite)."
    echo ""

    SMOKE_SLURMS=(
        "$LF_PROTO_ROOT/lf-homogeneous/singlet/run_smoke.slurm"
        "$LF_PROTO_ROOT/lf-homogeneous/triplet/run_smoke.slurm"
        "$LF_PROTO_ROOT/lov-protein/coupled/AT1/run_smoke.slurm"
    )
    SMOKE_JOBS=()
    for script in "${SMOKE_SLURMS[@]}"; do
        if [[ ! -f "$script" ]]; then
            echo "ERROR: missing smoke script $script" >&2
            SMOKE_STATUS=1
            continue
        fi
        calc_dir="$(dirname "$script")"
        base="$(basename "$script")"
        echo "Submitting smoke via sbatch from $calc_dir: $base"
        job_id="$(cd "$calc_dir" && sbatch --parsable "$base" 2>&1)" || {
            echo "ERROR: sbatch failed for $calc_dir/$base" >&2
            SMOKE_STATUS=1
            continue
        }
        if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
            echo "ERROR: sbatch returned invalid job id for $base: $job_id" >&2
            SMOKE_STATUS=1
            continue
        fi
        echo "  job_id=$job_id"
        SMOKE_JOBS+=("$job_id")
    done

    if [[ ${#SMOKE_JOBS[@]} -gt 0 ]]; then
        echo "Waiting for smoke jobs (squeue/sacct poll): ${SMOKE_JOBS[*]}"
        for job_id in "${SMOKE_JOBS[@]}"; do
            if ! _wait_slurm_job "$job_id" "smoke"; then
                SMOKE_STATUS=1
            fi
        done

        echo ""
        echo "Log grep (4 GPU parallelism):"
        rg -n "Using 4 parallel processes on GPU" "$LF_PROTO_ROOT"/**/slurm-smoke-*.out "$LF_PROTO_ROOT"/slurm-smoke-*.out 2>/dev/null || true
    fi
else
    echo "--- Step 4: smoke jobs skipped (pass --run-smoke to submit) ---"
fi

OVERALL=$(( INPUT_STATUS || PATH_STATUS || SLURM_STATUS || SMOKE_STATUS ))
echo "=== validation complete (exit $OVERALL) ==="
exit "$OVERALL"
