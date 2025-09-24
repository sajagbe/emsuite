# EMSuite - Electrostatic Map Suite

A comprehensive Python package for calculating electrostatic tuning effects on molecular properties using quantum mechanical methods.


![Water molecule tuning example](water-test/water-test.gif)
**Figure 1:** Tuning effects of a +1 *e* charge on the S<sub>1</sub> excitation energy of water, calculated at the B3LYP/cc-pVTZ level in vacuum.


## Overview

EMSuite is aimed at qualifying and quantifying the influence of external electrostatic fields on electronic structure and corresponding chemistry. The current implementation includes the `tuning` module, inspired by the concluding sentence of Electrostatic Spectral Tuning Maps for Biological Chromophores's ([Gozem et al., 2019](https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489)) abstract. This module extends the central approach of ESTMs to 13 new chemical properties enabling systematic exploration of supramolecular electronic influence. This approach enables the prediction and visualization of electrostatic tuning effects, which are crucial for understanding molecular behavior in different environments.

## Features

- **Multiple Input Formats**: Support for SMILES strings (with automatic QM optimization) and XYZ coordinate files
- **Comprehensive Property Calculations**: Ground state energies, orbital energies, dipole moments, ionization potentials, electron affinities, and excited state properties
- **GPU Acceleration**: Full GPU support via GPU4PySCF for enhanced computational speed (CPU fallback immediately available).
- **Implicit Solvation**: Built-in support for solvent effects using the SMD model.
- **Visualization Output**: MOL2 files for 3D visualization and CSV summaries for data analysis.

## Installation

```bash

#CPU Installation
pip install emsuite

#GPU Installation
pip install emsuite[gpu]

```

EMSuite automatically detects available hardware and uses GPU acceleration when available, falling back to CPU mode otherwise.

## Quick Start

1. **Create a tuning input file** (`tuning.in`):

```python
input_type = 'SMILES'      # Or 'xyz' for coordinate files
input_data = 'O'           # SMILES string or path to xyz file
method = 'dft'             # HF also permissible
basis_set = '6-31G*'       # Full list in method-info/basis-sets on GitHub
functional = 'pbe0'        # Full list in method-info/functionals.csv on GitHub
charge = 0                 # Molecular charge
spin = 0                   # Spin multiplicity
surface_charge = 1.0       # Point charge magnitude (can be fractional, e.g., 0.005)
solvent = None             # Solvent name or None for gas phase
properties = ['gse']       # List of properties to calculate (see below)
state_of_interest = 1      # Number of excited states (if exe/osc requested)
triplet = False            # Set True for triplet excited states
```

2. **Run the calculation**:

```bash
emsuite tuning.in
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

## Input Options

### Input Types
- **SMILES**: Automatically generates 3D coordinates and performs geometry optimization
- **XYZ**: Uses provided coordinate file directly

### Methods and Basis Sets
- **Methods**: `'dft'`, `'hf'`
- **Functionals**: See `method-info/functionals.csv` on GitHub for complete list
- **Basis Sets**: See `method-info/basis-sets/` on GitHub for available options
- **Solvents**: See `method-info/solvents/` on GitHub for available solvents

## Output Files

For each calculated property, EMSuite tuning generates:

1. **MOL2 files**: `{molecule_name}_{property}.mol2` - 3D visualization files showing property effects mapped to surface coordinates
2. **CSV summary**: `{molecule_name}_tuning_summary.csv` - Tabular data with coordinates and all property effects

## Example Usage

### Water molecule LUMO tuning:
```python
input_type = 'SMILES'
input_data = 'O'
basis_set = '6-31G*'
functional = 'b3lyp'
properties = ['lumo']
surface_charge = 1.0
```

### Benzene with multiple properties:
```python
input_type = 'SMILES'
input_data = 'c1ccccc1'
basis_set = 'def2-SVP'
functional = 'pbe0'
properties = ['homo', 'lumo', 'gap', 'ie', 'ea']
solvent = 'water'
```

### Excited state analysis:
```python
input_type = 'xyz'
input_data = 'molecule.xyz'
properties = ['exe', 'osc']
state_of_interest = 5
triplet = True
```

Sample Outputs for `tuning.in` in `water-test/`

## Citation

If you use EMSuite in your research, please cite:

> Gozem, S., et al. "Electrostatic Tuning of Molecular Properties" *J. Phys. Chem. B* **2019**, DOI: [10.1021/acs.jpcb.9b00489](https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, bug reports, or feature requests, please open an issue on GitHub.