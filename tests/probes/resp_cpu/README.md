# CPU RESP port: test against gpu4pyscf

`resp_cpu.py` is a line-for-line NumPy/PySCF port of `gpu4pyscf/pop/esp.py`
(gpu4pyscf 1.4.3): Merz-Kollman shells at 1.4/1.6/1.8/2.0 x GAMESS radii,
least-squares fit of atom charges to the electrostatic potential of the QM
density, total-charge constraint, single-stage RESP restraint (a = 5e-4,
b = 0.1, hydrogens unrestrained). The only changes: NumPy instead of CuPy, and
PySCF's `int1e_grids` (evaluated in blocks of 256 points) instead of gpu4pyscf's.

Not in this port yet (same as the GPU code): radii stop at Cl, so Br, I and
metals fail; one RESP stage only (Amber RESP uses two).

## Test (`compare_gpu.py`, B3LYP/6-31G*)
Molecules: water, LF, SB (neutral), rPSB (+1).
1. Same density into both codes: points, potential at every point, fit matrix,
   ESP and RESP charges. Expect agreement near numerical precision.
2. Full chains (CPU SCF + port vs GPU SCF + gpu4pyscf): charges only; also
   includes the CPU/GPU SCF difference.

Run: `sbatch run.slurm` (1 V100). Results: `results_<host>.json`, `slurm-*.out`.
Local CPU-only smoke test: `python compare_gpu.py water`.
