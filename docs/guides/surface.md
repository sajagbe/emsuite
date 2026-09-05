---
icon: lucide/orbit
---

# Surface Generation

The Surface channel samples a molecule's van der Waals envelope and writes the
coordinates to `.surf`. Inputs can be an XYZ file or a SMILES string.

## From XYZ

```python title="surface.in"
input_type = "XYZ"
input_data = "molecule.xyz"
output_surf = "molecule.surf"
surface_density = 1.0
surface_scale = 1.0
surface_type = "homogenous"
surface_charge = 0.10
optimize = False
```

```bash
emsuite -s surface.in
```

## From SMILES

```python title="surface.in"
input_type = "SMILES"
input_data = "CCO"
output_surf = "ethanol.surf"
optimized_xyz = "ethanol.xyz"
surface_density = 0.5
surface_scale = 1.0
optimize = True
optimize_method = "mmff"
```

SMILES inputs are embedded into 3D before sampling. `mmff` and `uff` provide
fast molecular-mechanics optimization; `pyscf` enables quantum-mechanical
optimization using the configured method, basis, functional, charge, spin,
and solvent.

## Density and scale

- `surface_density` controls the number of samples per square ångström. Higher
  values improve spatial resolution and increase downstream cost.
- `surface_scale` multiplies van der Waals radii. Values above 1 move the
  sampling envelope farther from the atoms.
- `surface_charge` is written to every point for `surface_type="homogenous"`.

## Python

```python
from emsuite import SurfaceInput

result = SurfaceInput.from_config(
    "surface.in",
    surface_density=0.5,
).run()

print(result.coords.shape)
result.to_xyz("surface-points.xyz")
result.to_mol2("surface-points.mol2")
```

`SurfaceResult.values` contains the fourth `.surf` column. The conversion
methods make the point cloud easy to inspect in molecular viewers.

## Common failures

- Ensure `input_type` is `XYZ` or `SMILES` as accepted by the current runner.
- Resolve relative paths from the directory where EMSuite is launched.
- Use a lower density for exploratory calculations before committing to an
  expensive tuning run.
