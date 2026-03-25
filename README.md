# EMSuite - Electrostatic Map Suite

A comprehensive Python package for calculating electrostatic tuning effects on molecular properties using quantum mechanical methods.


![Water molecule tuning example](water-test.gif)


**Figure 1:** Tuning effects of a +1 *e* charge on the S<sub>1</sub> excitation energy of water, calculated at the B3LYP/cc-pVTZ level in vacuum.


## Overview

EMSuite is aimed at qualifying and quantifying the influence of external electrostatic fields on electronic structure and corresponding chemistry. The current implementation includes the `tuning` module, inspired by the concluding sentence of Electrostatic Spectral Tuning Maps for Biological Chromophores's ([Gozem et al., 2019](https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489)) abstract. This module extends the central approach of ESTMs to 13 new chemical properties enabling systematic exploration of supramolecular electronic influence. This approach enables the prediction and visualization of electrostatic tuning effects, which are crucial for understanding molecular behavior in different environments.

## Features

- **Two-Stage Workflow**: Generate a VDW surface first, then run electrostatic tuning calculations
- **Multiple Input Formats**: Support for SMILES strings (with automatic optimization) and XYZ coordinate files
- **Comprehensive Property Calculations**: Ground state energies, orbital energies, dipole moments, ionization potentials, electron affinities, and excited state properties
- **GPU Acceleration**: Full GPU support via GPU4PySCF for enhanced computational speed (CPU fallback immediately available).
- **Implicit Solvation**: Built-in support for solvent effects using the PCM model.
- **Visualization Output**: Raw and normalized MOL2 files for 3D visualization, plus CSV summaries.
- **Resume Support**: Interrupted tuning runs can be resumed from log metadata.

## Installation

```bash

#CPU Installation
pip install emsuite

#GPU Installation
pip install emsuite[gpu]

```

EMSuite automatically detects available hardware and uses GPU acceleration when available, falling back to CPU mode otherwise.

## Command-Line Interface

EMSuite exposes two mutually exclusive workflows:

```bash
emsuite -s surface.in
emsuite -t tuning.in
```

- `-s, --surface INPUT_FILE`: Generate a `.surf` file from SMILES or XYZ input.
- `-t, --tuning INPUT_FILE`: Run electrostatic tuning using an existing XYZ structure and `.surf` file.

## Quick Start

1. **Create and run surface generation input** (`surface.in`):

```python
input_type = 'SMILES'      # 'SMILES' or 'XYZ'
input_data = 'CCO'         # SMILES string or XYZ path

surface_density = 1.0
surface_scale = 1.0
surface_type = 'homogenous'
surface_charge = 0.10
output_surf = 'CCO.surf'

optimize = True
optimize_method = 'uff'    # 'mmff', 'uff', or 'pyscf'
optimized_xyz = 'CCO_opt.xyz'
```

```bash
emsuite -s surface.in
```

2. **Create and run tuning input** (`tuning.in`):

```python
molecule = 'CCO_opt.xyz'   # or xyz_file = 'CCO_opt.xyz'
surface_file = 'CCO.surf'

charge = 0
spin = 0
basis_set = '6-31G*'
method = 'dft'
functional = 'b3lyp'
solvent = None

calc_type = 'separate'     # 'separate' or 'combined'
properties = ['exe']
state_of_interest = 1
triplet = False

parallel = True
num_procs = None           # Auto-detect if None
```

```bash
emsuite -t tuning.in
```

## Available Properties

The following molecular properties can be calculated:

| Property | Description | Units |
|----------|-------------|-------|
| `'gse'` | Ground state energy | kcal/mol |
| `'homo'` | HOMO energy | eV |
| `'lumo'` | LUMO energy | eV |
| `'gap'` | HOMO-LUMO gap | eV |
| `'dm'` | Dipole moment magnitude | Debye |
| `'ie'` | Ionization energy | kcal/mol |
| `'ea'` | Electron affinity | kcal/mol |
| `'cp'` | Chemical potential | kcal/mol |
| `'eng'` | Electronegativity | eV |
| `'hard'` | Chemical hardness | eV |
| `'efl'` | Electrophilicity | eV |
| `'nfl'` | Nucleophilicity | eV |
| `'exe'` | Excitation energies | eV |
| `'osc'` | Oscillator strengths | dimensionless |

Use `'all'` to calculate all available properties.

## Input File Reference

### surface.in (`emsuite -s surface.in`)

Required keys:
- `input_type`: `'SMILES'` or `'XYZ'`
- `input_data`: SMILES string or XYZ file path

Optional keys and defaults:
- `output_surf = 'surface.surf'`
- `optimized_xyz = None`
- `surface_density = 1.0`
- `surface_scale = 1.0`
- `surface_type = 'homogenous'`
- `surface_charge = 0.10`
- `optimize = None` (auto behavior: optimize for SMILES, do not optimize for XYZ)
- `optimize_method = 'mmff'` (`'mmff'`, `'uff'`, `'pyscf'`)
- `method = 'dft'`
- `basis_set = '6-31G*'`
- `functional = 'b3lyp'`
- `solvent = None`
- `charge = 0`
- `spin = 0`

Notes:
- If `input_type = 'XYZ'`, geometry optimization is only supported with `optimize_method = 'pyscf'`.
- `surface_type = 'heterogenous'` writes zero-valued placeholder charges for manual editing.

### tuning.in (`emsuite -t tuning.in`)

Required keys:
- `molecule` or `xyz_file`: Path to XYZ file
- `surface_file`: Path to `.surf` file

Optional keys and defaults:
- `basis_set = '6-31G*'`
- `method = 'dft'`
- `functional = 'b3lyp'`
- `charge = 0`
- `spin = 0`
- `solvent = None`
- `calc_type = 'separate'` (`'separate'` or `'combined'`)
- `properties = ['all']`
- `state_of_interest = 2`
- `triplet = False`
- `parallel = True`
- `num_procs = None` (auto-detect CPU/GPU worker count)

### Methods and Basis Sets
- **Methods**: `'dft'`, `'hf'`
- **Functionals**: See `method-info/functionals.csv` on GitHub for complete list
- **Basis Sets**: See `method-info/basis-sets/` on GitHub for available options
- **Solvents**: See `method-info/solvents/` on GitHub for available solvents

## Output Files

For each tuning run, EMSuite creates a timestamped results directory:

- `results_{molecule_name}_{timestamp}/`

Typical contents:

1. **Raw MOL2 files**: `{molecule_name}_{property}.mol2`
2. **Normalized MOL2 files**: `{molecule_name}_{property}_normalized.mol2`
3. **CSV summary**: `{molecule_name}_tuning_summary.csv`
4. **Logs directory**: `logs/` containing point logs, summary output, and `.resume_metadata.json`
5. **Run summary file**: `README.txt`


## Example Usage

### Example A: Generate surface from SMILES (CCO)
```python
input_type = 'SMILES'
input_data = 'CCO'
surface_density = 1.0
surface_scale = 1.0
surface_type = 'homogenous'
surface_charge = 0.1
output_surf = 'CCO2.surf'
optimize = True
optimize_method = 'uff'
optimized_xyz = 'CCO_opt2.xyz'
```

### Example B: Run tuning on generated files
```python
molecule = 'CCO_opt2.xyz'
surface_file = 'CCO2.surf'
basis_set = '6-31G*'
method = 'dft'
functional = 'b3lyp'
properties = ['exe']
calc_type = 'separate'
parallel = True
num_procs = 16
```

### Example C: Excited-state analysis
```python
molecule = 'molecule.xyz'
surface_file = 'molecule.surf'
properties = ['exe', 'osc']
state_of_interest = 5
triplet = True
```

Sample inputs and outputs for tuning runs are available in `examples/tuning/CCO2-exe/`.

## Citation

If you use EMSuite in your research, please cite:

> Gozem, S., et al. "Electrostatic Tuning of Molecular Properties" *J. Phys. Chem. B* **2019**, DOI: [10.1021/acs.jpcb.9b00489](https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, bug reports, or feature requests, please open an issue on GitHub.