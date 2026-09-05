---
icon: lucide/code-xml
---

# Python Workflows

The public Python API is centered on immutable input and result objects exported
from `emsuite`.

## Build from keywords

```python
from emsuite import SurfaceInput

config = SurfaceInput.from_config(
    input_type="SMILES",
    input_data="CCO",
    output_surf="ethanol.surf",
    optimize=True,
)

result = config.run()
```

## Load and override a file

```python
from emsuite import TuningInput

config = TuningInput.from_config(
    "tuning.in",
    parallel=False,
    properties=["homo", "lumo", "gap"],
)

print(config.to_dict())
result = config.run()
print(result.results_dir)
```

## Build from a mapping

```python
from emsuite import PotentialInput

config = PotentialInput.from_mapping(
    {
        "molecule": "ligand.xyz",
        "quantity": "charge",
        "output_surf": "charge.surf",
    }
)
```

## Compose channels

Use `CoupledInput` when the desired data flow is Potential followed by Tuning.
Use the individual inputs when an application needs to inspect or transform the
intermediate surface itself.

```python
potential = PotentialInput.from_file("potential.in").run()
potential.to_mol2("potential-preview.mol2")

tuning = TuningInput.from_config(
    molecule="ligand.xyz",
    surface_file=potential.path,
    properties=["homo", "gap"],
).run()
```

## Error handling

Configuration errors raise `ConfigValidationError`, a `ValueError` subclass,
before expensive channel work begins. Filesystem, external-tool, SCF, and APBS
errors may arise during `run()` and should be logged with the complete input
configuration and environment.
