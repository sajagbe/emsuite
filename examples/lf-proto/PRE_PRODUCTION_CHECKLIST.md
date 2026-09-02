# LF Proto pre-production checklist

All computation runs through **SLURM batch jobs (`sbatch`) only**. Do not use `srun`, interactive allocations, or bare `emsuite` on login nodes for production or smoke work.

## Before submission

- [ ] `export LF_PROTO_ROOT=/path/to/lf-proto`
- [ ] `export EMSUITE_ROOT=/path/to/emsuite`
- [ ] Bootstrap work directory: `bash "$EMSUITE_ROOT/scripts/lf_proto/bootstrap_workdir.sh"`
- [ ] Validate inputs and SLURM scripts: `cd "$LF_PROTO_ROOT" && ./validate.sh`
- [ ] Confirm Step 3b (workdir contract) passes — each `run.slurm` must run in its calc subdir (`singlet/`, `potential/AT1/`, etc.)
- [ ] Optional smoke test: `./validate.sh --run-smoke` (submits 3 short GPU jobs via `sbatch`)
- [ ] Confirm `prep/LF.surf` exists (or will be created by `submit_all.sh` via `sbatch prep/run_surface.slurm`)

## SLURM policy

| Step | Command | Notes |
|------|---------|-------|
| Surface prep | `cd prep && sbatch run_surface.slurm` | qCPU120, CPU-only |
| Smoke tests | `./validate.sh --run-smoke` | 3× `run_smoke.slurm` via sbatch |
| Production batch | `./submit_all.sh` | Surface (if needed) + 14 production jobs, all sbatch |

**Never use:** `srun`, `srun --pty`, `run_gpu.sh --local`, or `emsuite` directly on a login node.

### Why `sbatch --test-only` is not enough

Step 3 runs `sbatch --test-only` from each calc directory. That only validates partition, account, and resource requests — **it does not execute the script body**. A production submit that calls `sbatch /full/path/to/singlet/run.slurm` from the lf-proto root sets `SLURM_SUBMIT_DIR` to the root, so jobs fail instantly with wrong WorkDir and missing `tuning.in` / `potential.in` / `coupled.in`.

**Required:** `submit_all.sh` must use `(cd "$calc_dir" && sbatch run.slurm)` for every job. Each `run.slurm` should also `cd "${SLURM_SUBMIT_DIR:-...}"` as a belt-and-suspenders guard. Step 3b enforces both.

## After smoke (if run)

- [ ] All smoke jobs show `COMPLETED` in `sacct`
- [ ] Smoke logs contain `Using 4 parallel processes on GPU`
- [ ] `sacct` shows `BatchFlag=1` for each smoke job (confirms batch submission)

## Production submission

- [ ] `./submit_all.sh` from `$LF_PROTO_ROOT` (submits 14 jobs; surface prep first if `LF.surf` missing)
- [ ] Record job IDs printed by `submit_all.sh`

## Production sign-off

**Launched:** 2026-09-02 ~18:56 UTC (workdir fix)  
**Approved by:** User ("go")

**Job IDs:** 4235846, 4235847, 4235848, 4235849, 4235850, 4235851, 4235852, 4235853, 4235854, 4235855, 4235856, 4235857, 4235858, 4235859

**Submit errors:** None. Jobs RUNNING >60s; emsuite started successfully (GPU preflight + input patching confirmed in slurm logs).

**Previous failed batch (wrong WorkDir):** 4235653–4235666 — instant FAILED; root cause fixed in submit_all.sh.

