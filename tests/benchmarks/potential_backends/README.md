# RESP backend and APBS default benchmarks

Defaults chosen by the user on 2026-11-12: automatic usable-GPU RESP with CPU
fallback on unavailability, and maximum APBS grid spacing 0.125 Å for both
potential and coupled channels. Explicit grid_spacing=None preserves the legacy
65-point grid. The low-level APBS helper retains explicit fixed-grid control.

## Original measurements

`reference_results.json` retains complete results, settings, geometries and limitations.
GPU RESP prototype job 4959364: qGPU24, one GPU, four CPUs, 32 GB, HF/6-31G*.
Identical geometry and two-stage restraints/equivalence rules were used throughout.

| Case | CPU SCF s | GPU SCF s | CPU ESP s | GPU ESP s | Max final charge difference e |
|---|---:|---:|---:|---:|---:|
| Water | 0.292 | 4.875 | 0.0072 | 0.3721 | 5.836e-7 |
| Ethanol | 0.329 | 5.090 | 0.0289 | 0.0091 | 5.205e-7 |
| Flavin, 50 atoms | 642.863 | 97.851 | 3.8754 | 0.1928 | 1.324e-6 |

For flavin, matching-input CuPy/NumPy fitting differed by 1.061e-11 e; same-density
GPU/CPU ESP differed by 9.486e-13 atomic units. Tiny cases can be slower on GPU.
These are single timings including setup effects, not repeated performance estimates.

APBS refinement job 4959363: qCPU24, eight requested CPUs, 96 GB; fixed LOV1 PQR,
box and 148 evaluation points. Times include the old solver plus four ASCII grids,
their reading and interpolation; they predate selective potential-only I/O.

| Requested Å | Actual Å | Seconds | Mean difference from finest kT/e | Maximum difference kT/e |
|---:|---:|---:|---:|---:|
| 0.30 | 0.29195 | 27.0 | 0.586189 | 5.424542 |
| 0.25 | 0.24329 | 45.3 | 0.406750 | 6.519884 |
| 0.20 | 0.18247 | 108.9 | 0.229644 | 7.261466 |
| 0.175 | 0.16219 | 183.4 | 0.180732 | 4.193404 |
| 0.15 | 0.14598 | 223.7 | 0.128622 | 1.559521 |
| 0.125 | 0.12165 | 375.6 | 0.072932 | 1.457379 |
| 0.10 | 0.09732 | 2809.0 | reference | reference |

0.125 Å is a user-selected accuracy/cost default, not an exactness guarantee. The
finest grid is not an analytic reference; potential convergence does not establish
Gauss-law charge or tuning convergence. Actual spacing is rounded to an APBS-compatible size.

## Production regression validation

Unit tests cover absent CuPy, unusable CUDA, explicit CPU/forced GPU, one-device
selection/restoration, propagation of numerical errors, historical charge references,
and default/explicit APBS spacing behavior.

Local validation after the default change: **235 passed** (227 unit tests plus
8 CPU integration checks), one dependency warning, 291.21 s. Full output is in
`runs/cpu_validation.log`. These checks exercise the CPU fallback on a host without
usable CUDA and real APBS with the new channel defaults.

`validate_gpu.py` exercises the actual production automatic backend against forced CPU
on water, ethanol and flavin (HF), water (B3LYP), and open-shell OH (UHF). It records
chosen backend, method/basis/charge/spin, energy, stage times, charges and fixed acceptance
gates. GPU validation cannot pass by falling back to CPU. Prepared `run_validation.slurm`
requests one GPU/four CPUs/32 GB/20 minutes and requires approval before submission.
Production validation job **4961029** completed successfully; incremental JSON results and stdout
are under `runs/job-4961029/` and `runs/slurm-4961029.out`. Source hashes and resources
are recorded in `production_validation_submission.json`. **All five cases passed**, with
automatic GPU selection verified. Full charges, settings, energy/charge agreement and
stage timings are retained in `production_validation_results.json`.
The table above is prior prototype evidence. Production `total_s` includes backend
setup and SCF/ESP/fit, but excludes construction of the initial PySCF molecule.
