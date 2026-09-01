---
icon: lucide/braces
---

# Python API

The following names are exported from `emsuite`.

## Inputs

| Type | Required fields | Returns from `run()` |
| --- | --- | --- |
| `SurfaceInput` | `input_type`, `input_data` | `SurfaceResult` |
| `PotentialInput` | `molecule` | `PotentialResult` |
| `TuningInput` | `molecule`, `surface_file` | `TuningResult` |
| `CoupledInput` | `molecule` | `CoupledResult` |

Every input is a frozen dataclass and provides:

```python
Input.from_mapping(params)
Input.from_file(path)
Input.from_config(config=None, **overrides)
Input.from_any(config)
input.to_dict()
input.run()
```

`from_config()` is the recommended application entry point because it supports
defaults, a file or mapping, and explicit overrides in one call.

## Results

### `SurfaceResult`

```python
coords: numpy.ndarray
values: numpy.ndarray
path: str | None

SurfaceResult.from_surf(path)
result.to_surf(path=None)
result.to_xyz(path=None)
result.to_mol2(path=None)
```

### `PotentialResult`

```python
coords: numpy.ndarray
values: numpy.ndarray
quantity: str
path: str | None

PotentialResult.from_surf(path, quantity="potential")
result.to_surf(path=None)
result.to_mol2(path=None)
```

### `TuningResult`

```python
results_dir: str | None
```

### `CoupledResult`

```python
potential: PotentialResult
tuning: TuningResult
```

## Runner functions

The package also exports `run_surface_calculation`,
`run_potential_calculation`, `run_tuning_calculation`, and
`run_coupled_calculation`. New application code should normally prefer the
typed input objects.
