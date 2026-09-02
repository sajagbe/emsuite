# LF Proto batch workflow

Batch electrostatic tuning for lumiflavin (LF) in homogeneous and LOV-protein heterogeneous environments.

## Layout

Work directory (outside the repo):

```
lf-proto/
├── prep/                  # LF surface generation
├── lf-homogeneous/        # singlet + triplet exe tuning on bare LF
├── lov-protein/frames/    # extracted GROMACS frames (ligand.xyz, complex.pdb, CHR.mol2)
├── lov-protein/potential/ # APBS charge maps per system (CPU)
├── lov-protein/coupled/   # potential → HOMO/LUMO/gap tuning (GPU)
├── submit_all.sh
└── validate.sh
```

## Prerequisites

- `emsuite` on `PATH` (GPU nodes for tuning/coupled; CPU for potential)
- SLURM with `sbatch`
- GROMACS `.gro` trajectories under `LOVCalculations/Selected100-L1vL2/`
- LF geometry at `LOVCalculations/ChargeUpdates/LumiflavinRESP2025/LF/LF.xyz`

## Quick start

```bash
export LF_PROTO_ROOT=/data/PHO_WORK/sajagbe2/QMMM/LOVCalculations/lf-proto
export EMSUITE_ROOT=/data/PHO_WORK/sajagbe2/packages/emsuite

# Bootstrap work directory (frames, surface, inputs, symlinks):
bash "$EMSUITE_ROOT/scripts/lf_proto/bootstrap_workdir.sh"

# Validate inputs and SLURM scripts (no job submission):
cd "$LF_PROTO_ROOT" && ./validate.sh

# Submit all 14 production jobs:
./submit_all.sh
```

Or from a populated `lf-proto/` directory:

```bash
export LF_PROTO_ROOT=/path/to/lf-proto
./validate.sh
./submit_all.sh
```

## Validation workflow

`validate.sh` runs three checks before production submission:

1. **`validate_inputs.py`** — parse every `.in` file with EMSuite's `SurfaceInput` / `TuningInput` / `PotentialInput` / `CoupledInput` and verify referenced paths exist relative to each job directory.
2. **`check_paths.py`** — confirm the expected lf-proto tree (frames, LF.xyz, LF.surf, run scripts, smoke inputs).
3. **`sbatch --test-only`** — dry-run all 14 production `run.slurm` scripts (singlet, triplet, 6× potential, 6× coupled). Skipped with a warning if `sbatch` is unavailable.

```bash
export LF_PROTO_ROOT=/path/to/lf-proto
cd "$LF_PROTO_ROOT"
./validate.sh              # parse + paths + sbatch --test-only
./validate.sh --run-smoke  # also submit short GPU smoke jobs (see below)
```

## Smoke tests

Smoke inputs use a minimal basis (`sto-3g`) and reduced property set for fast GPU verification. Production inputs use `6-31G*`.

| Location | Input | SLURM | Notes |
|----------|-------|-------|-------|
| `lf-homogeneous/singlet/` | `tuning_smoke.in` | `run_smoke.slurm` | `homo` only, 4 GPUs, 30 min |
| `lf-homogeneous/triplet/` | `tuning_smoke.in` | `run_smoke.slurm` | `triplet = True` |
| `lov-protein/coupled/AT1/` | `coupled_smoke.in` | `run_smoke.slurm` | `surface_density = 0.3`, homo/lumo/gap |

Before running smoke jobs, confirm 4 GPUs are visible on the target node:

```bash
nvidia-smi -L   # expect 4 devices
```

Submit smoke jobs only (not production):

```bash
export LF_PROTO_ROOT=/path/to/lf-proto
cd "$LF_PROTO_ROOT"
./validate.sh --run-smoke
```

Each smoke `run_smoke.slurm` patches `num_procs` from `SLURM_GPUS_ON_NODE` (default 4) and runs `emsuite`. Check the log for:

```
Using 4 parallel processes on GPU
```

## GPU parallelism model

Each **tuning** or **coupled** job requests **4 GPUs** on one node. EMSuite runs surface points in parallel — one Ray worker per GPU (`num_gpus=1` per worker), not one GPU per separate job.

| Setting | Value | Meaning |
|---------|-------|---------|
| `#SBATCH --gres=gpu:4` | 4 GPU devices | SLURM allocates 4 GPUs to this job |
| `#SBATCH --cpus-per-task=32` | 32 CPUs | 8 CPUs per GPU for host-side work |
| `parallel = True` | Ray on | Enable point-wise parallelism |
| `num_procs = None` | patched at runtime | `patch_num_procs.py` sets this to `SLURM_GPUS_ON_NODE` (4) |
| `OMP_NUM_THREADS` | `$SLURM_CPUS_PER_TASK` | CPU threads within each point calculation |

With 4 GPUs, up to **4 surface points** are computed simultaneously within a single job. Potential jobs are CPU-only (APBS) and do not use `num_procs`.

Default exports in `submit_all.sh`:

```bash
export SLURM_GPUS=4
export SLURM_CPUS=32
export LF_PROTO_ROOT=/path/to/lf-proto
```

## Scripts (repo)

| Script | Purpose |
|--------|---------|
| `scripts/lf_proto/prepare_frames.py` | Middle frame from `.gro` → `ligand.xyz` + `complex.pdb` |
| `scripts/lf_proto/patch_num_procs.py` | Set `num_procs` from `SLURM_GPUS_ON_NODE` at job start |
| `scripts/lf_proto/validate_inputs.py` | Parse all `.in` files and check referenced paths |
| `scripts/lf_proto/check_paths.py` | Standalone lf-proto tree path checker |
| `scripts/lf_proto/validate.sh` | Orchestrate validation + optional smoke submission |
| `scripts/lf_proto/bootstrap_workdir.sh` | Create/populate lf-proto work directory |
| `scripts/lf_proto/submit_all.sh` | Frame prep + `sbatch` 14 jobs |

### Frame extraction

```bash
python3 scripts/lf_proto/prepare_frames.py \
  --gro /path/to/AT1Sel.gro \
  --out-dir /path/to/lf-proto/lov-protein/frames/AT1 \
  --system AT1
```

Writes `ligand.xyz` (CHR, for VDW surface), `complex.pdb` (protein + CHR HETATM for pdb2pqr), and `metadata.json`. Coordinates are converted nm → Å (×10). Symlink FMN MOL2 as `CHR.mol2` per system (see `bootstrap_workdir.sh` for source paths).

### Protein field (pdb2pqr path)

Potential and coupled jobs use `protein_format = 'pdb'` with the full complex PDB (protein + CHR HETATM). EMSuite selects the CHR residue by `ligand_resname`, runs pdb2pqr with AMBER charges at pH 7.0, and uses `CHR.mol2` for ligand cavity radii (`ligand_atoms = 'present'`). No `protein.xyz` or RDKit Gasteiger charges are needed on this path.

## Job breakdown (14 total)

| Job | Channel | Partition | GPUs | CPUs |
|-----|---------|-----------|------|------|
| singlet, triplet | `emsuite -t` | `qDEV` | 4 each | 32 |
| potential × 6 | `emsuite -p` | `qDEV` | 0 | 16 |
| coupled × 6 | `emsuite -c` | `qDEV` | 4 each | 32 |
