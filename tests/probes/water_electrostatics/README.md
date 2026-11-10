# Water electrostatics: q-scan, sign, superposition

Same water, same 32-point surface, B3LYP/6-31G* vacuum as
`tests/integration_runs/recreate-maps-2026-10-07T035624Z/water`.

| Test | Jobs | Result |
|---|---|---|
| q-scan + sign | 3 sentinel points × 8 charges | `out/qscan.csv`, `out/qscan.png` |
| few-charge superposition | 3 combined jobs (AB @ 0.10, ABC @ 0.10, AB @ 0.50) | `out/superpose.csv`, `out/superpose.png` |
| full-shell superposition | already on disk; no new QM | `out/full_shell.csv` |

## Run

```bash
cd tests/probes/water_electrostatics
python prepare.py          # write jobs/ from the existing map
sbatch run.slurm           # or: SLURM_PARTITION=qGPU24 sbatch run.slurm
python analyze.py          # full-shell now; q-scan/superpose when jobs finish
```

`prepare.py` picks RED / BLUE / ZERO from the existing S1 map (min, max, nearest-zero).
The q-scan jobs are also the single-charge legs of the superposition test.
