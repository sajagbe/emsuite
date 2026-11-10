# Fix validation (2026-11-03)

Validates three fixes on CPU and GPU against results produced before the IE/EA
baseline drift:

1. `tuning/output.py`: a failed surface point no longer loses the whole map
   (failed points are blank in the CSV).
2. `tuning/properties/thermo.py`: ie/ea/cp in kcal/mol, eng/hard/efl in eV,
   nfl in 1/eV (previously all Hartree, mislabeled).
3. `core/molecule.py` + `tuning/runner.py`: ion SCF no longer accepts collapsed
   solutions (SOSCF restarts from a fresh guess; states with <S^2> > S(S+1)+1 or
   |E_ion - E_neutral| > 1 Ha are rejected).

The water probe `run.slurm` path fix is tested separately by job 4836193.

## Cases

| Kind | Cases | Devices | Reference |
|---|---|---|---|
| tuning (subset of points) | water_full (all 30), lf_gas (8), rPSB_gas, SB_gas, rPSB_nhexane, rPSB_dmso, SB_water (6 each) | GPU (V100) and CPU | the earlier map at the same points (`manifest.json`) |
| baseline repeats | rPSB_gas, SB_gas, rPSB_nhexane, rPSB_chloroform, rPSB_acetonitrile, SB_dmso | GPU ×5, CPU ×2 | sane earlier baselines (|IE|,|EA| < 1 Ha) |
| pytest | `tests/integration/test_failed_point_recovery.py` (real QM, one point forced to fail) | CPU | — |

CPU tasks put `cpu_only_shim/` first on `PYTHONPATH`, so `import cupy` fails and
emsuite runs exactly as a CPU-only install would. (Without that, `resurrect_mol`
moves per-point SCFs to GPU whenever CuPy is importable, even with no GPU visible.)

Reference thermo values were written in Hartree; `analyze.py` converts them before
comparing, and skips thermo columns whose reference baseline was unphysical.

## Run

```bash
python prepare.py      # builds cases/, tasks_*.txt, manifest.json (done)
./submit.sh            # 3 arrays: fv-gpu4, fv-gpu1, fv-cpu  (jobs in SUBMITTED.txt)
python analyze.py      # writes REPORT.md
```

Code under test: `code_under_test.head` + `code_under_test.diff`. Jobs import the
live `src/` tree, so do not edit `src/` while they run.
