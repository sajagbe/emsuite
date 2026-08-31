# Water Tuning Reproduction Test

## Goal

Validate that the tuning channel on `feat/streamline-inputs-ligand-occupancy`
can accurately and consistently reproduce the historical Water calculation in:

```text
/Users/stephenajagbe/Desktop/Academic_Docs/Personal_Dissertation_Projects/emsuite/emsuite/results_Water_2026-03-18_05-04-06
```

The validation checks:

- exact input parsing and scientific configuration;
- successful completion of all 30 surface points;
- output inventory and MOL2 topology;
- pointwise raw effects, normalized effects, and baselines;
- repeatability across two independent runs;
- separation of scientific differences from timestamps and other runtime metadata.

This audit did not change scientific source code. The historical inputs and
results were treated as read-only.

## Code and environment

The test was performed on August 31, 2026 with:

```text
Branch: feat/streamline-inputs-ligand-occupancy
Commit: 9e25e42cf5bee4a2a258afadec26f10528120f23
Python: 3.11.15
PySCF: 2.10.0
NumPy: 2.3.2
SciPy: 1.16.1
Compute: CPU, 8 Ray workers available (16 requested by tuning.in)
```

The repository was clean before and after the calculation.

## Exact input files

The following historical files were copied byte-for-byte into each isolated
temporary run directory:

```text
/Users/stephenajagbe/Desktop/Academic_Docs/Personal_Dissertation_Projects/emsuite/emsuite/Water.xyz
/Users/stephenajagbe/Desktop/Academic_Docs/Personal_Dissertation_Projects/emsuite/emsuite/Water.surf
/Users/stephenajagbe/Desktop/Academic_Docs/Personal_Dissertation_Projects/emsuite/emsuite/tuning.in
```

SHA-256 hashes:

```text
17987b1741982a31aa10ee3a240f4e4751283892831b24730d3b03de45f09246  Water.xyz
b9add3c191cb357f8859de60ad6a51fdc28d1c0dc5d0072237245a296fef8d67  Water.surf
f34ea5e9c553b407b1e1e7a84a4ff82b87b2cc14f94de18b3659ffd41a8b1ee5  tuning.in
```

### `Water.xyz`

```text
3
O
O 0.008935 0.404022 0.000000
H -0.787313 -0.184699 0.000000
H 0.778378 -0.219323 0.000000
```

### `Water.surf`

```text
x          y          z          q
0.330051   0.258893   1.478588   0.100000
-0.565266  0.694724   1.377021   0.100000
-0.441561  -0.335765  1.249067   0.100000
0.302295   1.232470   1.240167   0.100000
1.128925   0.652071   0.997243   0.100000
-0.563737  1.551753   0.815574   0.100000
-1.310898  0.690356   0.697463   0.100000
0.092047   -0.955989  0.673694   0.100000
0.933721   1.505656   0.491501   0.100000
0.089336   1.917644   0.113504   0.100000
1.478851   0.787945   0.048480   0.100000
-1.124211  1.417116   -0.004607  0.100000
-0.054436  -1.081258  -0.316743  0.100000
0.861093   1.556469   -0.506056  0.100000
-1.334070  0.658830   -0.664689  0.100000
-0.459584  1.660302   -0.715996  0.100000
1.240166   0.459497   -0.889603  0.100000
0.387657   -0.593696  -1.082371  0.100000
-0.591999  -0.352939  -1.173153  0.100000
0.297469   1.205620   -1.258805  0.100000
-0.609346  0.716120   -1.353042  0.100000
-1.419041  -0.300020  1.013716   0.100000
-0.899176  -1.136302  0.722454   0.100000
-1.746437  -0.863365  0.243913   0.100000
1.945962   -0.423830  -0.186880  0.100000
1.326456   -1.270293  -0.187278   0.100000
-1.936329  0.126782   0.150804   0.100000
-1.042451  -1.312665  -0.320309  0.100000
-1.750647  -0.426295  -0.673512  0.100000
0.965254   -0.449502  1.162796   0.100000
```

### `tuning.in`

```python
molecule = 'Water.xyz'
charge = 0
spin = 0

basis_set = '6-31G*'
method = 'dft'
functional = 'b3lyp'
solvent = None

surface_file = 'Water.surf'
calc_type = 'separate'

properties = ['gse','homo','lumo','gap','dm','ie','ea','cp','eng','hard','efl','nfl']
state_of_interest = 1
triplet = False

parallel = True
num_procs = 16
```

The branch's `TuningInput` parser resolved this file to the same molecule,
surface, charge, spin, method, functional, basis, calculation type, property
set, state of interest, triplet setting, and parallel configuration.

## Commands and steps

### 1. Verify the branch test suite with a clean import path

The shell had a stale `PYTHONPATH` pointing at an older Desktop checkout. It was
explicitly removed so the feature branch was the only package under test:

```bash
env -u PYTHONPATH uv run --extra dev pytest tests/unit tests/regression -q
```

Result:

```text
62 passed in 2.20s
```

### 2. Create the first isolated fixture

```bash
fixture_dir=$(mktemp -d /tmp/emsuite-water-validation.XXXXXX)

cp \
  /Users/stephenajagbe/Desktop/Academic_Docs/Personal_Dissertation_Projects/emsuite/emsuite/Water.xyz \
  /Users/stephenajagbe/Desktop/Academic_Docs/Personal_Dissertation_Projects/emsuite/emsuite/Water.surf \
  /Users/stephenajagbe/Desktop/Academic_Docs/Personal_Dissertation_Projects/emsuite/emsuite/tuning.in \
  "$fixture_dir"
```

Actual fixture directory:

```text
/tmp/emsuite-water-validation.6tDgjt
```

### 3. Run the first calculation

From the fixture directory:

```bash
env -u PYTHONPATH \
  /Users/stephenajagbe/orca/emsuite/.venv/bin/emsuite \
  --tuning tuning.in
```

Result directory:

```text
/tmp/emsuite-water-validation.6tDgjt/results_Water_2026-08-31_18-37-50
```

### 4. Repeat independently

The same three inputs were copied into a second new directory:

```text
/tmp/emsuite-water-repeat.368rrd
```

The same command was run from that directory:

```bash
env -u PYTHONPATH \
  /Users/stephenajagbe/orca/emsuite/.venv/bin/emsuite \
  --tuning tuning.in
```

Second result directory:

```text
/tmp/emsuite-water-repeat.368rrd/results_Water_2026-08-31_18-38-53
```

Both calculations completed all 30 of 30 points successfully, with no resume
and no failed points.

## Results and differences

### New run versus new repeat run

The scientific outputs are fully repeatable:

```text
Water_tuning_summary.csv: byte-for-byte identical
MOL2 files:               24/24 byte-for-byte identical
Coordinates/order:        identical
Successful points:        30/30 in both runs
Maximum numeric delta:    0
```

The only differences between the two new result directories are:

- result and log timestamps;
- timestamps in `.resume_metadata.json`;
- nondeterministic property display order in `README.txt`;
- an incidental macOS `.DS_Store` in the second directory.

The README property lines demonstrate the display-order issue:

```diff
-Properties:         homo, dm, cp, lumo, gap, hard, nfl, efl, gse, ie, ea, eng
+Properties:         eng, ea, nfl, lumo, dm, gap, homo, ie, hard, gse, efl, cp
```

This ordering difference does not affect CSV columns, MOL2 files, calculations,
or values.

### Both new runs versus the March 18 reference

Because the two new CSV files and all their MOL2 files are identical, both have
the same comparison against the March 18 reference:

```text
Coordinates and point indices: exact
MOL2 files:                   24/24 byte-for-byte identical
CSV rows:                     30 in both
CSV byte-for-byte identical:  no (sub-roundoff floating-point differences)
```

Maximum absolute differences:

| Category | Maximum absolute difference | Location |
| --- | ---: | --- |
| Raw effect | `5.82076609134674e-11` | `gse_effect`, point 7 |
| Normalized effect | `2.90867330221545e-11` | `ea_effect_normalized`, point 27 |
| Baseline | `7.27595761418343e-12` | `gse_baseline`, point 29 |
| Coordinates/index | `0` | all points |

The three maximum-difference examples are:

```diff
gse_effect, point 7
-3.0577615055444767
+3.0577615054862690
delta = -5.8207660913467407e-11

ea_effect_normalized, point 27
-0.62433405145596277
+0.62433405148504950
delta = +2.9086733022154476e-11

gse_baseline, point 29
--47945.568560894841
+-47945.568560894833
delta = +7.2759576141834259e-12
```

The historical/new README difference is limited to runtime metadata and
property display order:

```diff
 Molecule:           Water
-Timestamp:          2026-03-18_05-04-06
-Properties:         hard, gap, cp, gse, lumo, homo, eng, efl, ie, ea, dm, nfl
+Timestamp:          2026-08-31_18-37-50
+Properties:         homo, dm, cp, lumo, gap, hard, nfl, efl, gse, ie, ea, eng
```

## Conclusion

At commit `9e25e42cf5bee4a2a258afadec26f10528120f23`, the streamlined tuning channel
reproduces the March 18 Water calculation accurately and consistently. Both new
runs completed all points, produced identical scientific artifacts to each
other, produced MOL2 files byte-identical to the historical reference, and
differed from the historical CSV only at approximately `1e-11` or smaller.

The only observed consistency defect is nondeterministic property ordering in
human-readable README output. It does not affect scientific results.
