# LF Proto pre-production checklist

All computation runs through **SLURM batch jobs (`sbatch`) only**. Do not use `srun`, interactive allocations, or bare `emsuite` on login nodes for production or smoke work.

## Before submission

- [ ] `export LF_PROTO_ROOT=/path/to/lf-proto`
- [ ] `export EMSUITE_ROOT=/path/to/emsuite`
- [ ] Bootstrap work directory: `bash "$EMSUITE_ROOT/scripts/lf_proto/bootstrap_workdir.sh"`
- [ ] Validate inputs and SLURM scripts: `cd "$LF_PROTO_ROOT" && ./validate.sh`
- [ ] Optional smoke test: `./validate.sh --run-smoke` (submits 3 short GPU jobs via `sbatch`)
- [ ] Confirm `prep/LF.surf` exists (or will be created by `submit_all.sh` via `sbatch prep/run_surface.slurm`)

## SLURM policy

| Step | Command | Notes |
|------|---------|-------|
| Surface prep | `cd prep && sbatch run_surface.slurm` | qCPU120, CPU-only |
| Smoke tests | `./validate.sh --run-smoke` | 3× `run_smoke.slurm` via sbatch |
| Production batch | `./submit_all.sh` | Surface (if needed) + 14 production jobs, all sbatch |

**Never use:** `srun`, `srun --pty`, `run_gpu.sh --local`, or `emsuite` directly on a login node.

## After smoke (if run)

- [ ] All smoke jobs show `COMPLETED` in `sacct`
- [ ] Smoke logs contain `Using 4 parallel processes on GPU`
- [ ] `sacct` shows `BatchFlag=1` for each smoke job (confirms batch submission)

## Production submission

- [ ] `./submit_all.sh` from `$LF_PROTO_ROOT` (submits 14 jobs; surface prep first if `LF.surf` missing)
- [ ] Record job IDs printed by `submit_all.sh`
