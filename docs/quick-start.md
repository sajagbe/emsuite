---
icon: lucide/rocket
---

# Quick Start

This walkthrough creates a small Water surface and prepares a tuning
calculation. Quantum-chemistry runtime depends on method, basis, hardware, and
surface size; the commands are intentionally explicit rather than promising a
fixed completion time.

## 1. Install EMSuite

=== "uv"

    ```bash
    uv add emsuite
    ```

=== "pip"

    ```bash
    pip install emsuite
    ```

Confirm the command is available:

```bash
emsuite --help
```

## 2. Create the molecule

Save this as `water.xyz`:

```text
3
Water
O  0.008935  0.404022  0.000000
H -0.787313 -0.184699  0.000000
H  0.778378 -0.219323  0.000000
```

## 3. Generate a surface

Save this as `surface.in`:

```python
input_type = "XYZ"
input_data = "water.xyz"
output_surf = "water.surf"
surface_density = 0.25
surface_scale = 1.0
surface_type = "homogenous"
surface_charge = 0.10
optimize = False
```

Run:

```bash
emsuite -s surface.in
```

The output is a four-column `water.surf` file containing `x`, `y`, `z`, and
probe-charge values.

!!! note "Compatibility spelling"

    The current configuration literal is `homogenous`. Documentation uses the
    correct English word *homogeneous* everywhere else.

## 4. Configure tuning

Save this as `tuning.in`:

```python
molecule = "water.xyz"
surface_file = "water.surf"
properties = ["homo", "lumo", "gap"]
basis_set = "sto-3g"
method = "dft"
functional = "b3lyp"
charge = 0
spin = 0
solvent = None
calc_type = "separate"
parallel = False
```

Run:

```bash
emsuite -t tuning.in
```

## 5. Inspect the result

EMSuite creates a timestamped directory:

```text
results_water_YYYY-MM-DD_HH-MM-SS/
├── water_tuning_summary.csv
├── water_homo.mol2
├── water_homo_normalized.mol2
├── water_lumo.mol2
├── water_lumo_normalized.mol2
├── water_gap.mol2
├── water_gap_normalized.mol2
├── README.txt
└── logs/
```

Use the CSV for quantitative analysis and the MOL2 files for spatial
visualization.

## Next steps

- [Tuning guide](guides/tuning.md)
- [Property reference](reference/properties.md)
- [Output reference](reference/outputs.md)
- [Python workflows](guides/python.md)
