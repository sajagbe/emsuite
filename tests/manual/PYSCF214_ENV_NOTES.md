# Isolated PySCF 2.14 env (worktree only)

Purpose: validate the NumPy ≥2.4 `lib.einsum` fix (PySCF PR #3099 / release 2.14.0)
**without** changing the live user install (`~/.local` PySCF 2.10.0 / gpu4pyscf 1.4.3)
used by opsins GPU jobs.

## Create / use

```bash
cd /data/PHO_WORK/sajagbe2/packages/emsuite-osc-cpu
python -m venv --system-site-packages .venv
.venv/bin/pip install -U pip wheel setuptools
.venv/bin/pip install 'pyscf==2.14.0'
# Optional paired GPU stack (venv only; does not uninstall ~/.local):
.venv/bin/pip install 'gpu4pyscf-cuda12x==1.8.1'
```

`--system-site-packages` keeps NumPy/SciPy/CuPy from the user site visible while the
venv’s packages shadow older user-site wheels. `pip` reported “Not uninstalling …
outside environment” for both pyscf and gpu4pyscf — **`~/.local` stayed intact**.

```bash
export PYTHONPATH=src
.venv/bin/python tests/manual/bench_osc_cpu_extract.py
sbatch tests/manual/bench_osc_pyscf214.slurm
sbatch tests/manual/probe_gpu4pyscf_pyscf214.slurm
```

## Versions

| Package | Live user (`~/.local`) | Worktree `.venv` (after setup) |
|---------|------------------------|--------------------------------|
| numpy   | 2.4.4                  | 2.4.4 (user site) |
| pyscf   | **2.10.0** (untouched) | **2.14.0** |
| gpu4pyscf-cuda12x | **1.4.3** (untouched) | **1.8.1** (optional; for paired GPU smoke) |

## Results

### `lib.einsum` on NumPy 2.4.4
- PySCF **2.10.0**: FAIL (`ValueError: not enough values to unpack (expected 4, got 3)`)
- PySCF **2.14.0**: **PASS** (`ij,jk,kl->ik` and `xpq,pi,qj->xij`)

### Stock CPU `td.oscillator_strength()` (PySCF 2.14 only)
- **PASS**; matches `oscillator_strength_cpu` (~1e-14 rel).

### gpu4pyscf **1.4.3** + PySCF **2.14.0** (GPU V100, job 4572330)
- Imports OK
- `to_gpu()` **FAIL**: `AttributeError: 'RKS' object has no attribute 'cphf_grids'`
- `gpu4pyscf.dft.RKS.kernel()` **FAIL**: `TypeError: eigh() ... 4 were given`
- **Not deployable** as a drop-in upgrade for current opsins GPU stack.

### gpu4pyscf **1.8.1** + PySCF **2.14.0** (GPU V100, job 4572332)
- `to_gpu()` OK; native `gpu4pyscf.dft.RKS` OK
- TDDFT OK
- **Stock `td.oscillator_strength()` OK on GPU** (~2–5 ms)
- `oscillator_strength_cpu` matches (~1e-15 rel, ~1–2 ms)

## Recommendation

1. **Do not** upgrade live `~/.local` PySCF to 2.14 while opsins still run gpu4pyscf **1.4.3** — GPU SCF breaks.
2. **Keep `oscillator_strength_cpu`** as the production fix on the current 2.10 + 1.4.3 stack.
3. Future paired upgrade path: PySCF **2.14** + gpu4pyscf **≥1.8.1** restores stock GPU osc; still keep the CPU helper as cheap defense-in-depth.

`.venv/` is gitignored; do not commit wheels.
