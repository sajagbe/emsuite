---
icon: lucide/workflow
---

# Coupled Workflows

Coupled runs Potential and Tuning as one workflow. The Potential result's
fourth `.surf` column becomes the perturbation consumed by Tuning.

## Generate the potential surface

```python title="coupled.in"
molecule = "ligand.xyz"
protein = "protein.xyz"
output_surf = "coupled-charge.surf"
potential_method = "apbs"
potential_quantity = "charge"
ligand_atoms = "present"
pdie = 2.0
sdie = 78.54

properties = ["homo", "lumo", "gap"]
basis_set = "6-31G*"
method = "dft"
functional = "b3lyp"
calc_type = "separate"
parallel = False
```

```bash
emsuite -c coupled.in
```

`potential_quantity="charge"` is the normal choice for tuning because it gives
each point an elementary-charge perturbation.

## Reuse an existing potential surface

Set `potential_surf` to bypass APBS:

```python
molecule = "ligand.xyz"
potential_surf = "validated-charge.surf"
potential_quantity = "charge"
properties = ["homo", "lumo", "gap"]
```

This is useful when the potential stage has already been reviewed or when an
external workflow produced the heterogeneous surface.

## Python

```python
from emsuite import CoupledInput

result = CoupledInput.from_file("coupled.in").run()
print(result.potential.path)
print(result.tuning.results_dir)
```

The result keeps both stages visible, so applications can validate the
intermediate field before accepting the tuning maps.
