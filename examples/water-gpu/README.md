# Water GPU tuning (SMILES → surface → tuning)

End-to-end test of EMSuite on **water** using a GPU node. Starts from SMILES `O`,
builds a homogeneous charged VDW surface, then runs electrostatic tuning maps.

## Prerequisites

On the GPU compute node:

```bash
pip install -e "/path/to/emsuite[gpu,dev]"
python -c "import gpu4pyscf.dft; from emsuite.core import check_gpu_info; assert check_gpu_info() >= 1"
```

## Option A — CLI (two steps)

```bash
cd examples/water-gpu
chmod +x run_gpu.sh

./run_gpu.sh              # smoke: homo/lumo/gap (~30 surface points)
./run_gpu.sh --full       # full CodexTest property set (slower)
```

Or manually on an allocated GPU:

```bash
emsuite -s surface.in
emsuite -t tuning_smoke.in    # or tuning.in for full run
```

## Option B — Python API (one script)

```bash
./run_gpu.sh --api          # smoke
./run_gpu.sh --api --full   # full property set
```

Or directly:

```bash
python run_api.py
python run_api.py --full
```

Equivalent inline API:

```python
from emsuite import SurfaceInput, TuningInput

surf = SurfaceInput.from_config(
    input_type="SMILES",
    input_data="O",
    output_surf="Water.surf",
    optimized_xyz="Water.xyz",
    surface_charge=0.10,
    optimize=True,
    optimize_method="uff",
).run()

TuningInput.from_config(
    molecule="Water.xyz",
    surface_file=surf.path,
    properties=["homo", "lumo", "gap"],
    parallel=True,
    num_procs=1,
).run()
```

## Outputs

- `Water.xyz` — UFF-optimized geometry from SMILES
- `Water.surf` — VDW surface with uniform +0.10 e probe charge
- `results_Water_<timestamp>/` — MOL2 maps, CSV summary, logs

## Notes

- `num_procs = 1` targets a single-GPU Ray worker (matches one `srun --gres=gpu:1`).
- Surface from SMILES will **not** match the historical `Water.surf` in `CodexTest.md`
  byte-for-byte (different geometry source). For regression against that run, use the
  copied `Water.xyz` / `Water.surf` inputs instead of SMILES.
- Full run (`tuning.in`) includes `ie`/`ea` and needs anion/cation SCF — expect
  ~1–3 hours on one GPU depending on queue hardware.
