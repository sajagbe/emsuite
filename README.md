# EMSuite - Electrostatic Map Suite

A Python package for electrostatic tuning maps, APBS potential surfaces, and coupled
potential→tuning workflows on molecular systems.


![Water molecule tuning example](docs/_static/water-test.gif)


**Figure 1:** Tuning effects of a +1 *e* charge on the S<sub>1</sub> excitation energy of water, calculated at the B3LYP/cc-pVTZ level in vacuum.


## Overview

EMSuite qualifies and quantifies how external electrostatic environments shift
electronic structure. Four channels work together:

| Channel | CLI | Role |
|---------|-----|------|
| **Surface** | `emsuite -s surface.in` | Build a VDW surface (`.surf`) from SMILES or XYZ |
| **Tuning** | `emsuite -t tuning.in` | QM/MM tuning maps for molecular properties |
| **Potential** | `emsuite -p potential.in` | APBS φ or Gauss-law charges on a surface |
| **Coupled** | `emsuite -c coupled.in` | Potential map → tuning in one run |

The tuning channel extends the electrostatic spectral tuning approach
([Gozem et al., 2019](https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489)).

## Installation

```bash
pip install emsuite          # CPU
pip install emsuite[gpu]     # CPU + GPU4PySCF + CuPy (CUDA 12.x)
```

EMSuite detects GPUs automatically and falls back to CPU when none are available.

---

## Command-line interface

The CLI is the primary interface. Every workflow is driven by a **`.in` file**
(a Python-syntax assignment list). Copy a template from `examples/templates/`,
edit paths and parameters, then run one command.

```bash
emsuite -s surface.in      # generate VDW surface
emsuite -t tuning.in         # electrostatic tuning maps
emsuite -p potential.in      # APBS potential / Gauss-law charge map
emsuite -c coupled.in        # potential → tuning pipeline
emsuite --help
```

| Flag | Channel | Input |
|------|---------|-------|
| `-s`, `--surface` | Surface | `surface.in` |
| `-t`, `--tuning` | Tuning | `tuning.in` |
| `-p`, `--potential` | Potential | `potential.in` |
| `-c`, `--coupled` | Coupled | `coupled.in` |

### Typical workflows

**A. Homogeneous probe charge (SMILES → surface → tuning)**

```text
SMILES/XYZ  ──►  emsuite -s surface.in  ──►  molecule.surf + molecule.xyz
                                                      │
                                                      ▼
                                            emsuite -t tuning.in
                                                      │
                                                      ▼
                                            results_{molecule}_{timestamp}/
```

**B. Protein field (potential → coupled tuning)**

```text
ligand.xyz + protein.pdb  ──►  emsuite -c coupled.in
                                        │
                    APBS Gauss-law charges on ligand VDW
                                        │
                                        ▼
                              tuning maps on heterogeneous .surf
```

**C. Potential only (inspect φ or charges before tuning)**

```text
molecule.xyz  ──►  emsuite -p potential.in  ──►  potential.surf (+ .csv)
```

---

### Tutorial 1 — Water from SMILES (GPU tuning)

A complete worked example lives in `examples/water-gpu/`.

**Step 1 — `surface.in`**

```python
input_type = 'SMILES'
input_data = 'O'

output_surf = 'Water.surf'
optimized_xyz = 'Water.xyz'

surface_density = 1.0
surface_type = 'homogenous'
surface_charge = 0.10          # uniform +0.10 e probe on every point

optimize = True
optimize_method = 'uff'
```

```bash
emsuite -s surface.in
```

Produces `Water.xyz` (UFF geometry) and `Water.surf` (~30 VDW points).

**Step 2 — `tuning.in`**

```python
molecule = 'Water.xyz'
surface_file = 'Water.surf'

basis_set = '6-31G*'
method = 'dft'
functional = 'b3lyp'
charge = 0
spin = 0
solvent = None

calc_type = 'separate'         # one QM/MM calc per surface point
properties = ['homo', 'lumo', 'gap']
state_of_interest = 1
triplet = False

parallel = True                # Ray + GPU when available
num_procs = 1                  # one GPU worker (set to match --gres=gpu:N)
```

```bash
emsuite -t tuning.in
```

On a SLURM cluster, run from `examples/water-gpu/`:

```bash
./run_gpu.sh              # smoke (CLI)
./run_gpu.sh --full       # full property set (see tuning.in in that folder)
```

---

### Tutorial 2 — Ethanol excited-state tuning

**`surface.in`**

```python
input_type = 'SMILES'
input_data = 'CCO'
output_surf = 'CCO.surf'
optimized_xyz = 'CCO.xyz'
surface_charge = 0.10
optimize = True
optimize_method = 'uff'
```

**`tuning.in`**

```python
molecule = 'CCO.xyz'
surface_file = 'CCO.surf'
properties = ['exe', 'osc']
state_of_interest = 1
triplet = False
calc_type = 'separate'
parallel = True
```

```bash
emsuite -s surface.in
emsuite -t tuning.in
```

Sample inputs and outputs: `examples/tuning/CCO2-exe/`.

---

### Tutorial 3 — Ligand in a protein field (coupled)

Use when the electrostatic environment comes from a macromolecule and you want
Gauss-law surface charges feeding tuning.

**`coupled.in`**

```python
molecule = 'ligand.xyz'        # ligand geometry; VDW surface is built on this

protein = 'complex.pdb'
protein_format = 'pdb'
ligand_resname = 'LIG'
ligand_atoms = 'present'       # 'present' | 'absent' | 'charged'
ligand_mol2 = 'ligand.mol2'    # required for present/charged with PDB protein

output_surf = 'coupled.surf'
potential_method = 'apbs'
potential_quantity = 'charge'  # Gauss-law q (e) at each surface point

properties = ['homo', 'lumo', 'gap']
basis_set = '6-31G*'
calc_type = 'separate'
parallel = True
num_procs = 1
```

```bash
emsuite -c coupled.in
```

To run potential and tuning separately instead:

```bash
emsuite -p potential.in      # writes heterogeneous potential.surf
# edit tuning.in to point surface_file at that .surf
emsuite -t tuning.in
```

Reuse a precomputed potential surface in coupled mode (skip APBS on repeat runs):

```python
potential_surf = 'precomputed.surf'   # skips potential channel entirely
```

---

### Tutorial 4 — Potential map only

**`potential.in`**

```python
molecule = 'ligand.xyz'
surface_file = None            # auto-generate ligand VDW if omitted
output_surf = 'potential.surf'
method = 'apbs'
quantity = 'potential'         # 'potential' = interpolated φ; 'charge' = Gauss-law q

# Optional protein field (XYZ Gasteiger or PDB via pdb2pqr):
# protein = 'protein.xyz'
# ligand_atoms = 'absent'

pdie = 2.0
sdie = 78.54
```

```bash
emsuite -p potential.in
```

The `.surf` fourth column holds APBS potential (when `quantity='potential'`) or
Gauss-law charge in *e* (when `quantity='charge'`). A companion `.csv` is written
alongside the `.surf` file.

---

## Input file reference

Templates with comments: `examples/templates/{surface,tuning,potential,coupled}.in`.

### `surface.in` — `emsuite -s`

| Key | Default | Description |
|-----|---------|-------------|
| `input_type` | *(required)* | `'SMILES'` or `'XYZ'` |
| `input_data` | *(required)* | SMILES string or path to XYZ |
| `output_surf` | `'surface.surf'` | Output `.surf` path |
| `optimized_xyz` | auto | Path for optimized geometry |
| `surface_density` | `1.0` | Points per Å² |
| `surface_scale` | `1.0` | VDW radii scale factor |
| `surface_type` | `'homogenous'` | `'homogenous'` or `'heterogenous'` |
| `surface_charge` | `0.10` | Uniform charge (homogenous only) |
| `optimize` | auto | `True` for SMILES, `False` for XYZ |
| `optimize_method` | `'mmff'` | `'mmff'`, `'uff'`, or `'pyscf'` |
| `method`, `basis_set`, `functional` | DFT defaults | Used when `optimize_method='pyscf'` |
| `charge`, `spin` | `0`, `0` | Molecular charge and 2S spin |
| `solvent` | `None` | PCM solvent name or `None` |

`heterogenous` writes zero charges as placeholders for hand-editing.

### `tuning.in` — `emsuite -t`

| Key | Default | Description |
|-----|---------|-------------|
| `molecule` | *(required)* | Path to XYZ geometry |
| `surface_file` | *(required)* | Path to `.surf` file |
| `properties` | `['all']` | Property codes (see table below) |
| `basis_set` | `'6-31G*'` | Basis set |
| `method` | `'dft'` | `'dft'` or `'hf'` |
| `functional` | `'b3lyp'` | XC functional (DFT only) |
| `charge`, `spin` | `0`, `0` | Molecular charge and 2S spin |
| `solvent` | `None` | PCM solvent |
| `calc_type` | `'separate'` | `'separate'` (per-point) or `'combined'` |
| `state_of_interest` | `2` | TD states (for `exe`/`osc`) |
| `triplet` | `False` | Triplet TD states |
| `parallel` | `True` | Ray parallel workers |
| `num_procs` | `None` | Worker count (`None` = auto-detect GPUs/CPUs) |

Interrupted runs resume from `logs_*/.resume_metadata.json` when parameters match.

### `potential.in` — `emsuite -p`

| Key | Default | Description |
|-----|---------|-------------|
| `molecule` | *(required)* | Ligand / molecule XYZ (`ligand` alias) |
| `protein` | `None` | Protein XYZ or PDB for the external field |
| `protein_format` | `'xyz'` | `'xyz'` (Gasteiger) or `'pdb'` (pdb2pqr) |
| `ligand_atoms` | `'present'` | `'present'`, `'absent'`, or `'charged'` (PDB only) |
| `ligand_resname` | `None` | HETATM residue name (PDB) |
| `ligand_mol2` | `None` | Ligand MOL2 for pdb2pqr occupancy |
| `forcefield` | `'AMBER'` | pdb2pqr force field |
| `ph` | `7.0` | pdb2pqr protonation pH (`None` to disable) |
| `surface_file` | `None` | Existing VDW surface; generated if omitted |
| `output_surf` | `'potential.surf'` | Output heterogeneous `.surf` |
| `surface_density` | `0.5` | VDW density when generating surface |
| `method` | `'apbs'` | APBS Poisson–Boltzmann |
| `quantity` | `'potential'` | `'potential'` or `'charge'` (Gauss-law) |
| `pdie`, `sdie` | `2.0`, `78.54` | APBS dielectric constants |

When `protein` is set, the APBS box spans protein **and** ligand coordinates so a
pocket ligand is not clipped.

### `coupled.in` — `emsuite -c`

Accepts all **potential** keys (prefixed `potential_` where needed) plus all
**tuning** keys. Notable defaults:

| Key | Default | Description |
|-----|---------|-------------|
| `potential_method` | `'apbs'` | Potential backend |
| `potential_quantity` | `'charge'` | Gauss-law charges for tuning |
| `potential_surf` | `None` | Skip potential step; reuse this `.surf` |
| `calc_type` | `'separate'` | Tuning mode |
| `parallel` | `False` | Parallel tuning (set `True` on GPU) |

---

## Available tuning properties

| Property | Description | Units |
|----------|-------------|-------|
| `gse` | Ground state energy | kcal/mol |
| `homo` | HOMO energy | eV |
| `lumo` | LUMO energy | eV |
| `gap` | HOMO–LUMO gap | eV |
| `dm` | Dipole moment magnitude | Debye |
| `spin` | Spin magnitude | dimensionless |
| `ie` | Ionization energy | kcal/mol |
| `ea` | Electron affinity | kcal/mol |
| `cp` | Chemical potential | kcal/mol |
| `eng` | Electronegativity | eV |
| `hard` | Chemical hardness | eV |
| `efl` | Electrophilicity | eV |
| `nfl` | Nucleophilicity | eV |
| `fukui_plus` | Nucleophilic Fukui index | eV |
| `fukui_minus` | Electrophilic Fukui index | eV |
| `exe` | Excitation energies | eV |
| `osc` | Oscillator strengths | dimensionless |

Use `properties = ['all']` for the full registry. Dependencies (e.g. `gap` pulls
in `homo` and `lumo`) are resolved automatically.

---

## Output

### Tuning (`emsuite -t` / coupled tuning step)

```
results_{molecule}_{timestamp}/
├── {molecule}_tuning_summary.csv
├── {molecule}_{property}.mol2
├── {molecule}_{property}_normalized.mol2
├── logs/
│   ├── calculation_summary.out
│   ├── .resume_metadata.json
│   └── point_*.log
└── README.txt
```

### Potential (`emsuite -p` / coupled potential step)

```
potential.surf          # heterogeneous x, y, z, value
potential.csv           # same data, tabular
```

### Surface (`emsuite -s`)

```
molecule.surf           # x, y, z, q
molecule.xyz            # optimized geometry (when optimize=True)
```

---

## Python API

The CLI is a thin wrapper around frozen input dataclasses. Every `.in` file maps
to `from_file()`; every key maps to `from_config(**kwargs)`.

```python
from emsuite import SurfaceInput, TuningInput, PotentialInput, CoupledInput

# Equivalent to: emsuite -s surface.in
SurfaceInput.from_file("surface.in").run()

# Equivalent to: emsuite -t tuning.in
TuningInput.from_file("tuning.in").run()

# Build inline (no .in file):
surf = SurfaceInput.from_config(
    input_type="SMILES", input_data="O",
    output_surf="Water.surf", optimized_xyz="Water.xyz",
    surface_charge=0.10,
).run()

TuningInput.from_config(
    molecule="Water.xyz",
    surface_file=surf.path,
    properties=["homo", "lumo", "gap"],
    parallel=True,
    num_procs=1,
).run()
```

`from_config` accepts `config=` (path or dict) with keyword overrides:

```python
TuningInput.from_config(config="tuning.in", parallel=False).run()
```

Lower-level runners (same calculations, no result objects):

```python
import emsuite

emsuite.run_surface_calculation("surface.in")
emsuite.run_tuning_calculation("tuning.in")
emsuite.run_potential_calculation("potential.in")
emsuite.run_coupled_calculation("coupled.in")
```

See `examples/water-gpu/run_api.py` for a full SMILES→tuning script.

---

## GPU notes

- Install with `pip install emsuite[gpu]` on the compute node (or in your job env).
- Set `parallel = True` and `num_procs` to the number of GPUs allocated.
- Verify: `python -c "import gpu4pyscf.dft; from emsuite.core import check_gpu_info; print(check_gpu_info())"`
- Integration tests: `pytest -m gpu` (CUDA required)
- Water benchmark: `examples/water-gpu/run_gpu.sh`

If `gpu4pyscf` import fails after install, reinstall cleanly:

```bash
pip uninstall -y gpu4pyscf-cuda12x gpu4pyscf-libxc-cuda12x gpu4pyscf-libxc-cuda11x
rm -rf ~/.local/lib/python*/site-packages/gpu4pyscf
pip install --no-cache-dir 'gpu4pyscf-cuda12x==1.4.3'
```

---

## Examples

| Path | Contents |
|------|----------|
| `examples/templates/` | Annotated `.in` templates for all four channels |
| `examples/water-gpu/` | SMILES water → GPU tuning (CLI + Python) |
| `examples/tuning/CCO2-exe/` | Ethanol `exe` tuning sample outputs |
| `examples/families/` | LOV / miniSOG / SOPP3 combined `exe` examples by family |

---

## Citation

> Gozem, S., et al. "Electrostatic Tuning of Molecular Properties" *J. Phys. Chem. B* **2019**, DOI: [10.1021/acs.jpcb.9b00489](https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489)

## License

MIT License — see [LICENSE](LICENSE).

## Support

Open an issue on GitHub for bugs and feature requests.
