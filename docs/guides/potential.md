---
icon: lucide/waves
---

# Potential Mapping

The Potential channel runs APBS and maps either electrostatic potential or
Gauss-law surface charge onto a ligand van der Waals surface.

## Standalone molecule

```python title="potential.in"
molecule = "ligand.xyz"
surface_file = None
output_surf = "ligand-potential.surf"
surface_density = 0.5
surface_scale = 1.0
method = "apbs"
quantity = "potential"
pdie = 2.0
sdie = 78.54
charge = 0
spin = 0
```

```bash
emsuite -p potential.in
```

Set `quantity="charge"` when the output will be used as a heterogeneous tuning
surface.

## Protein field with XYZ inputs

```python title="potential.in"
ligand = "ligand.xyz"
protein = "protein.xyz"
protein_format = "xyz"
ligand_atoms = "present"
quantity = "charge"
output_surf = "protein-field.surf"
```

The surface remains the ligand surface. `ligand_atoms` controls the APBS model:

| Mode | Meaning |
| --- | --- |
| `present` | Include ligand atoms with neutralized charges |
| `absent` | Omit ligand atoms; requires a protein |
| `charged` | Include ligand atoms with their assigned charges |

## Protein PDB through PDB2PQR

```python title="potential.in"
ligand = "ligand.xyz"
protein = "complex.pdb"
protein_format = "pdb"
ligand_resname = "LIG"
ligand_chain = "A"
ligand_resseq = 401
ligand_mol2 = "ligand.mol2"
ligand_atoms = "charged"
forcefield = "AMBER"
ph = 7.0
quantity = "charge"
output_surf = "complex-charge.surf"
```

For PDB input, `protein`, `ligand_resname`, and—when ligand atoms are present
or charged—`ligand_mol2` are required. Chain and residue number disambiguate
repeated residue names.

## Python

```python
from emsuite import PotentialInput

result = PotentialInput.from_file("potential.in").run()
print(result.quantity, result.values.min(), result.values.max())
result.to_mol2("potential.mol2")
```

## Boundaries

- APBS is the only implemented potential backend.
- `esp` and `mep` are reserved but not implemented.
- The removed `coulomb` method and bond-axis scan keys are rejected explicitly.
