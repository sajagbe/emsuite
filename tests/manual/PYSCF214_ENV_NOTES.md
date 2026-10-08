# Optional PySCF 2.14 + gpu4pyscf ≥1.8.1 env

Purpose: validate the NumPy ≥2.4 `lib.einsum` fix (PySCF PR #3099 / release 2.14.0)
and paired GPU stack **without** changing the live user install
(`~/.local` PySCF 2.10.0 / gpu4pyscf 1.4.3) used by production jobs.

Default emsuite (this package) already ships `oscillator_strength_cpu` so
oscillator strengths work on the live 2.10 + 1.4.3 stack. The paired upgrade
restores stock GPU `td.oscillator_strength()` as well.

## Create / use (repo-local `.venv`)

```bash
cd /data/PHO_WORK/sajagbe2/packages/emsuite
python -m venv --system-site-packages .venv
.venv/bin/pip install -U pip wheel setuptools
.venv/bin/pip install 'pyscf==2.14.0'
.venv/bin/pip install 'gpu4pyscf-cuda12x==1.8.1'
```

`--system-site-packages` keeps NumPy/SciPy/CuPy from the user site visible while the
venv’s packages shadow older user-site wheels.

```bash
export PYTHONPATH=src
.venv/bin/python tests/manual/bench_osc_cpu_extract.py
sbatch tests/manual/bench_osc_pyscf214.slurm
sbatch tests/manual/probe_gpu4pyscf_pyscf214.slurm
sbatch tests/manual/integrated_pyscf214_stack.slurm
```

## Versions

| Package | Live user (`~/.local`) | Optional `.venv` |
|---------|------------------------|------------------|
| numpy   | 2.4.4                  | 2.4.4 (user site) |
| pyscf   | **2.10.0**             | **2.14.0** |
| gpu4pyscf-cuda12x | **1.4.3**     | **1.8.1** |

## Results (V100 probes)

### `lib.einsum` on NumPy 2.4.4
- PySCF **2.10.0**: FAIL (`ValueError: not enough values to unpack`)
- PySCF **2.14.0**: **PASS**

### gpu4pyscf **1.4.3** + PySCF **2.14.0**
- GPU SCF **FAIL** (`cphf_grids` / `eigh` arity) — not a drop-in upgrade.

### gpu4pyscf **1.8.1** + PySCF **2.14.0**
- SCF / TDDFT / stock GPU osc **OK**; matches `oscillator_strength_cpu` (~1e-15).

## Recommendation

1. **Do not** upgrade live PySCF to 2.14 while production still runs gpu4pyscf **1.4.3**.
2. Keep `oscillator_strength_cpu` as the default path (and defense-in-depth after a paired upgrade).
3. Future paired upgrade: PySCF **2.14** + gpu4pyscf **≥1.8.1**.

`.venv/` is gitignored; do not commit wheels.
