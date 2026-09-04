#!/usr/bin/env bash
# Create/populate the lf-proto work directory (no production sbatch).
set -euo pipefail

LF_PROTO_ROOT="${1:-${LF_PROTO_ROOT:-/data/PHO_WORK/sajagbe2/QMMM/LOVCalculations/lf-proto}}"
EMSUITE_ROOT="${EMSUITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPTS_DIR="${LF_PROTO_SCRIPTS:-$EMSUITE_ROOT/scripts/lf_proto}"
GRO_ROOT="${LF_PROTO_GRO_ROOT:-$LF_PROTO_ROOT/../Selected100-L1vL2}"
LF_XYZ_SRC="${LF_PROTO_LF_XYZ:-$LF_PROTO_ROOT/../ChargeUpdates/LumiflavinRESP2025/LF/LF.xyz}"
SYSTEMS=(AT1 AT2 AS1 AS2 CR1 CR2)
GRO_NAMES=(AT1Sel AT2Sel AS1Sel AS2Sel CR1Sel CR2Sel)
QMMM_ROOT="${LF_PROTO_QMMM_ROOT:-$LF_PROTO_ROOT/..}"
MOL2_SOURCES=(
    "$QMMM_ROOT/atLOV1/2Z6D_FMN_APEC_Calculation_Files/2Z6D_FMN.mol2"
    "$QMMM_ROOT/mutantTests/realatLOV2/4eep_FMN_APEC_Calculation_Files/4eep_FMN.mol2"
    "$QMMM_ROOT/asLOV1/0NaBak/asLOV1_FMN.mol2"
    "$QMMM_ROOT/asLOV2/2V1A_FMN_APEC_Calculation_Files/2V1A_FMN.mol2"
    "$QMMM_ROOT/crLOV1/1n9l_FMN_APEC_Calculation_Files/1n9l_FMN.mol2"
    "$QMMM_ROOT/crLOV2/crLOV2_FMN_APEC_Calculation_Files/crLOV2_FMN.mol2"
)

export LF_PROTO_ROOT="$LF_PROTO_ROOT"

mkdir -p "$LF_PROTO_ROOT"/{prep,lf-homogeneous/{singlet,triplet},lov-protein/{frames,potential,coupled}}

# Symlink helper scripts
for script in patch_num_procs.py prepare_frames.py validate_inputs.py check_paths.py validate.sh submit_all.sh; do
    ln -sfn "$SCRIPTS_DIR/$script" "$LF_PROTO_ROOT/$script"
done
ln -sfn "$SCRIPTS_DIR/patch_num_procs.py" "$LF_PROTO_ROOT/prep/patch_num_procs.py"
ln -sfn "$SCRIPTS_DIR/prepare_frames.py" "$LF_PROTO_ROOT/prep/prepare_frames.py"

# LF geometry
ln -sfn "$LF_XYZ_SRC" "$LF_PROTO_ROOT/prep/LF.xyz"
ln -sfn "$LF_XYZ_SRC" "$LF_PROTO_ROOT/lf-homogeneous/singlet/LF.xyz"
ln -sfn "$LF_XYZ_SRC" "$LF_PROTO_ROOT/lf-homogeneous/triplet/LF.xyz"

# --- prep ---
cat > "$LF_PROTO_ROOT/prep/surface.in" <<'EOF'
input_type = 'XYZ'
input_data = 'LF.xyz'
output_surf = 'LF.surf'
surface_density = 1.0
surface_scale = 1.0
surface_type = 'homogenous'
surface_charge = 0.10
optimize = False
EOF

cat > "$LF_PROTO_ROOT/prep/run_surface.slurm" <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=lf-proto-surface
#SBATCH --partition=qCPU120
#SBATCH --account=CHEM9C4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=23000MB
#SBATCH --time=04:00:00
#SBATCH --output=slurm-surface-%j.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
emsuite -s surface.in
EOF
chmod +x "$LF_PROTO_ROOT/prep/run_surface.slurm"

# --- homogeneous tuning inputs ---
write_tuning() {
    local dir="$1"
    local triplet="$2"
    local outfile="$3"
    local basis="$4"
    local props="$5"
    local soi="$6"
    cat > "$dir/$outfile" <<EOF
molecule = 'LF.xyz'
surface_file = 'LF.surf'

charge = 0
spin = 0
basis_set = '$basis'
method = 'dft'
functional = 'b3lyp'
solvent = None

calc_type = 'separate'
properties = $props
state_of_interest = $soi
triplet = $triplet

parallel = True
num_procs = None
EOF
}

# Production LF homogeneous: excited-state exe with soi=3 (not homo/lumo/gap)
write_tuning "$LF_PROTO_ROOT/lf-homogeneous/singlet" False tuning.in '6-31G*' "['exe']" 3
write_tuning "$LF_PROTO_ROOT/lf-homogeneous/triplet" True tuning.in '6-31G*' "['exe']" 3
write_tuning "$LF_PROTO_ROOT/lf-homogeneous/singlet" False tuning_smoke.in 'sto-3g' "['homo']" 1
write_tuning "$LF_PROTO_ROOT/lf-homogeneous/triplet" True tuning_smoke.in 'sto-3g' "['homo']" 1

write_gpu_slurm() {
    local path="$1"
    local jobname="$2"
    local input="$3"
    local channel="$4"
    local lf_up="${5:-../..}"
    cat > "$path" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$jobname
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
export LC_ALL="\${LC_ALL:-en_US.UTF-8}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-32}"
export PATH="\${HOME}/.local/bin:\${PATH}"

cd "\${SLURM_SUBMIT_DIR:-\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)}"
CALC_DIR="\$PWD"
LF_PROTO_ROOT="\${LF_PROTO_ROOT:-\$(cd "\$CALC_DIR/$lf_up" && pwd)}"
PATCH="\${LF_PROTO_ROOT}/scripts/patch_num_procs.py"
[[ -f "\$PATCH" ]] || PATCH="\${LF_PROTO_ROOT}/prep/patch_num_procs.py"
python -c "import gpu4pyscf.dft; from emsuite.core import check_gpu_info; n = check_gpu_info() or 0; assert n >= 1, f'GPU preflight failed: {n}'"
python "\$PATCH" $input
emsuite -$channel $input
EOF
}

write_smoke_slurm() {
    local path="$1"
    local jobname="$2"
    local input="$3"
    local channel="$4"
    local lf_up="${5:-../..}"
    cat > "$path" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$jobname
#SBATCH --partition=qGPU48
#SBATCH --account=CHEM9C4
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-smoke-%j.out
#SBATCH --exclude=acidsgcn007,acidsgcn001

set -euo pipefail

export PYTHONUTF8=1
export LC_ALL="\${LC_ALL:-en_US.UTF-8}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-32}"
export PATH="\${HOME}/.local/bin:\${PATH}"

cd "\${SLURM_SUBMIT_DIR:-\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)}"
CALC_DIR="\$PWD"
LF_PROTO_ROOT="\${LF_PROTO_ROOT:-\$(cd "\$CALC_DIR/$lf_up" && pwd)}"
PATCH="\${LF_PROTO_ROOT}/scripts/patch_num_procs.py"
[[ -f "\$PATCH" ]] || PATCH="\${LF_PROTO_ROOT}/prep/patch_num_procs.py"
python -c "import gpu4pyscf.dft; from emsuite.core import check_gpu_info; n = check_gpu_info() or 0; assert n >= 1, f'GPU preflight failed: {n}'"
NGPU="\${SLURM_GPUS_ON_NODE:-4}"
python "\$PATCH" $input "\$NGPU"
emsuite -$channel $input
EOF
}

write_gpu_slurm "$LF_PROTO_ROOT/lf-homogeneous/singlet/run.slurm" lf-proto-singlet tuning.in t
write_gpu_slurm "$LF_PROTO_ROOT/lf-homogeneous/triplet/run.slurm" lf-proto-triplet tuning.in t
write_smoke_slurm "$LF_PROTO_ROOT/lf-homogeneous/singlet/run_smoke.slurm" lf-proto-singlet-smoke tuning_smoke.in t
write_smoke_slurm "$LF_PROTO_ROOT/lf-homogeneous/triplet/run_smoke.slurm" lf-proto-triplet-smoke tuning_smoke.in t

# --- lov-protein frames ---
for i in "${!SYSTEMS[@]}"; do
    sys="${SYSTEMS[$i]}"
    gro="${GRO_ROOT}/${GRO_NAMES[$i]}.gro"
    out="$LF_PROTO_ROOT/lov-protein/frames/$sys"
    mol2_src="${MOL2_SOURCES[$i]}"
    if [[ -f "$gro" ]]; then
        python3 "$SCRIPTS_DIR/prepare_frames.py" \
            --gro "$gro" \
            --out-dir "$out" \
            --system "$sys" \
            --mol2-source "$mol2_src"
    else
        echo "WARN: missing $gro — skipping frame extraction for $sys"
        mkdir -p "$out"
    fi
    if [[ ! -f "$out/CHR.mol2" && -f "$mol2_src" ]]; then
        echo "WARN: CHR.mol2 not written for $sys (mol2 remap failed?)"
    fi
done

# --- potential + coupled per system ---
for sys in "${SYSTEMS[@]}"; do
    pot_dir="$LF_PROTO_ROOT/lov-protein/potential/$sys"
    coup_dir="$LF_PROTO_ROOT/lov-protein/coupled/$sys"
    mkdir -p "$pot_dir" "$coup_dir"

    ln -sfn "../../frames/$sys/ligand.xyz" "$pot_dir/ligand.xyz"
    ln -sfn "../../frames/$sys/complex.pdb" "$pot_dir/complex.pdb"
    ln -sfn "../../frames/$sys/CHR.mol2" "$pot_dir/CHR.mol2"
    ln -sfn "../../frames/$sys/ligand.xyz" "$coup_dir/ligand.xyz"
    ln -sfn "../../frames/$sys/complex.pdb" "$coup_dir/complex.pdb"
    ln -sfn "../../frames/$sys/CHR.mol2" "$coup_dir/CHR.mol2"

    cat > "$pot_dir/potential.in" <<EOF
molecule = 'ligand.xyz'
protein = 'complex.pdb'
protein_format = 'pdb'
ligand_resname = 'CHR'
ligand_mol2 = 'CHR.mol2'
output_surf = 'potential.surf'
surface_density = 0.5
surface_scale = 1.0
method = 'apbs'
quantity = 'charge'
ligand_atoms = 'present'
forcefield = 'AMBER'
ph = 7.0
pdie = 2.0
sdie = 78.54
charge = 0
spin = 0
EOF

    cat > "$coup_dir/coupled.in" <<EOF
molecule = 'ligand.xyz'
protein = 'complex.pdb'
protein_format = 'pdb'
ligand_resname = 'CHR'
ligand_mol2 = 'CHR.mol2'
output_surf = 'coupled.surf'
surface_density = 0.5
surface_scale = 1.0
potential_method = 'apbs'
potential_quantity = 'charge'
ligand_atoms = 'present'
forcefield = 'AMBER'
ph = 7.0
pdie = 2.0
sdie = 78.54
properties = ['homo', 'lumo', 'gap']
basis_set = '6-31G*'
method = 'dft'
functional = 'b3lyp'
charge = 0
spin = 0
solvent = None
calc_type = 'separate'
parallel = True
num_procs = None
state_of_interest = 2
triplet = False
EOF

    cat > "$pot_dir/run.slurm" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lf-proto-pot-$sys
#SBATCH --partition=qCPU120
#SBATCH --account=CHEM9C4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=23000MB
#SBATCH --time=120:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-8}"
cd "\${SLURM_SUBMIT_DIR:-\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)}"
emsuite -p potential.in
EOF

    write_gpu_slurm "$coup_dir/run.slurm" "lf-proto-coup-$sys" coupled.in c ../../..
done

# AT1 coupled smoke only
cat > "$LF_PROTO_ROOT/lov-protein/coupled/AT1/coupled_smoke.in" <<'EOF'
molecule = 'ligand.xyz'
protein = 'complex.pdb'
protein_format = 'pdb'
ligand_resname = 'CHR'
ligand_mol2 = 'CHR.mol2'
output_surf = 'coupled_smoke.surf'
surface_density = 0.3
surface_scale = 1.0
potential_method = 'apbs'
potential_quantity = 'charge'
ligand_atoms = 'present'
forcefield = 'AMBER'
ph = 7.0
pdie = 2.0
sdie = 78.54
properties = ['homo', 'lumo', 'gap']
basis_set = 'sto-3g'
method = 'dft'
functional = 'b3lyp'
charge = 0
spin = 0
solvent = None
calc_type = 'separate'
parallel = True
num_procs = None
state_of_interest = 1
triplet = False
EOF

write_smoke_slurm "$LF_PROTO_ROOT/lov-protein/coupled/AT1/run_smoke.slurm" lf-proto-coup-AT1-smoke coupled_smoke.in c ../../..

# Surface prep: sbatch-only (submit via submit_all.sh or manually)
if [[ -f "$LF_PROTO_ROOT/prep/LF.surf" ]]; then
    ln -sfn "$LF_PROTO_ROOT/prep/LF.surf" "$LF_PROTO_ROOT/lf-homogeneous/singlet/LF.surf"
    ln -sfn "$LF_PROTO_ROOT/prep/LF.surf" "$LF_PROTO_ROOT/lf-homogeneous/triplet/LF.surf"
else
    echo "prep/LF.surf missing — submit surface prep with:"
    echo "  cd $LF_PROTO_ROOT/prep && sbatch run_surface.slurm"
fi

echo "lf-proto work directory ready at $LF_PROTO_ROOT"
