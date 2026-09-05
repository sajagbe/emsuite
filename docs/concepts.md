---
icon: lucide/lightbulb
---

# Concepts

## The channel model

EMSuite separates geometry sampling, electrostatic field construction, and
property response into composable channels:

```text
XYZ / SMILES
     │
     ▼
  Surface ───────────────► homogeneous .surf ─────► Tuning
     │                                               │
     └────► Potential ──► heterogeneous .surf ───────┘
                           │
                           └──────────────────────► Coupled
```

- **Surface** decides *where* the environment is sampled.
- **Potential** decides *what electrostatic value* belongs at each point.
- **Tuning** measures *how molecular properties respond* to those values.
- **Coupled** runs Potential and Tuning as one typed workflow.

## The `.surf` contract

The shared interchange format is a text table:

```text
x          y          z          q
0.330051   0.258893   1.478588   0.100000
```

The first three columns are Cartesian coordinates in ångström. The fourth
column depends on the producer:

- a Surface workflow writes a uniform probe value for homogeneous tuning;
- Potential with `quantity="potential"` writes interpolated APBS potential;
- Potential with `quantity="charge"` writes Gauss-law surface charge in
  elementary-charge units.

Tuning consumes the fourth column as the point perturbation. This is why the
quantity must be chosen deliberately in coupled calculations.

## Typed inputs

Each channel has an immutable dataclass:

```python
SurfaceInput
PotentialInput
TuningInput
CoupledInput
```

They share four construction patterns:

```python
TuningInput.from_file("tuning.in")
TuningInput.from_mapping({"molecule": "water.xyz", ...})
TuningInput.from_config("tuning.in", parallel=False)
TuningInput.from_any(existing_input)
```

Validation happens before a runner starts. `to_dict()` provides a serializable
view, and `run()` returns the corresponding result type.

## Typed results

Surface and Potential results expose numeric coordinates and values:

```python
result.coords
result.values
result.path
result.to_surf("copy.surf")
```

`TuningResult` identifies the organized result directory. `CoupledResult`
contains both the potential and tuning results, allowing a caller to inspect
the intermediate field and final response maps.

## Separate and combined tuning

`calc_type="separate"` runs one calculation for each surface point and produces
a spatial response map. `calc_type="combined"` applies all surface charges in a
single calculation. This tuning setting is distinct from the **Coupled**
channel, which means Potential followed by Tuning.

## Protein and ligand occupancy

Potential and Coupled workflows can construct APBS inputs from a protein PDB
and a ligand geometry. Occupancy controls whether ligand atoms are present in
the PQR cavity model while keeping the sampling surface tied to the ligand.
See [Potential Mapping](guides/potential.md) for the supported modes.
