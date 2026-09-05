---
icon: lucide/folder-output
---

# Output Files

## Surface and Potential

Both channels write `.surf` point tables. Python results can additionally write
MOL2 point clouds; `SurfaceResult` can also write XYZ.

## Tuning directory

```text
results_<molecule>_<timestamp>/
├── <molecule>_tuning_summary.csv
├── <molecule>_<property>.mol2
├── <molecule>_<property>_normalized.mol2
├── README.txt
└── logs/
    ├── calculation_summary.out
    ├── point_0000.out
    ├── ...
    └── .resume_metadata.json
```

### CSV

The summary contains one row per surface point and columns for:

- point index and `x`, `y`, `z` coordinates;
- raw effect for each property;
- normalized effect for each property;
- no-surface baseline for each property.

### MOL2 maps

Raw MOL2 charge columns contain property effects. Normalized maps rescale each
property for comparative visualization. Point coordinates and order should
match the input surface.

### Logs

Point logs record status, coordinates, charge, timestamp, and calculated
effects. The summary records completion statistics and baselines. Resume
metadata identifies completed points after interruption.

Timestamps, worker order, and README property display order are runtime
metadata; compare CSV values and MOL2 contents for scientific reproducibility.
