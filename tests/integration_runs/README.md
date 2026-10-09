# Integration run logs

Slow integration tests can archive a full, traceable record of every run here.

## Consistency suite (cleanup / refactor gates)

Use this when changing shared plumbing (dead-code removal, CLI/input factories,
result types). It runs the core slow channel matrix, archives workspaces, and
records numeric fingerprints for before/after compare.

Numeric fingerprints for potential/tuning use the committed fixture
`tests/integration/fixtures/methane.surf` because the external `vsg` tool does
not emit bit-identical VDW point sets across runs. Surface-*generation* tests
still call `vsg` but only record structural assertions.

```bash
# Phase 0 / new oracle on current main
python scripts/run_consistency_suite.py --baseline

# After each cleanup step
python scripts/run_consistency_suite.py
# → auto-compares fingerprints to tests/integration_runs/baseline_fingerprints.json
```

## Run a full audited suite

```bash
python scripts/run_consistency_suite.py
```

(Older docs referred to `run_slow_integration_audit.py`; the consistency runner
supersedes it for cleanup gates and also archives via `EMSUITE_INTEGRATION_RUN_DIR`.)

Each run creates a timestamped folder, for example `consistency-2026-06-21T153045Z/`, containing:

| Artifact | Purpose |
|----------|---------|
| `RUN_MANIFEST.md` | Human-readable index: feature ↔ test ↔ pass/fail |
| `run_meta.json` | Git commit, EMSuite version, pytest command |
| `pytest.log` | Full pytest console log |
| `junit.xml` | Machine-readable results |
| `<test_name>/manifest.json` | Feature id, code paths, outcome, duration |
| `<test_name>/workspace/` | Copy of the test working directory (inputs, outputs, logs) |
| `<test_name>/assertions.json` | Checks performed by the test |
| `<test_name>/failure.txt` | Present only when a test fails |

Run folders are gitignored; this README is tracked.

## Run tests without archiving

```bash
uv run pytest tests/integration -v -m slow
```

Archiving is enabled only when `EMSUITE_INTEGRATION_RUN_DIR` is set (the audit script does this automatically).

## GPU channel validation

Requires a CUDA GPU (gpu4pyscf + CuPy). From the repo root:

```bash
chmod +x scripts/run_gpu_integration.sh
./scripts/run_gpu_integration.sh
```

This allocates `srun --partition=qDEV --gres=gpu:1` (override with `SLURM_PARTITION`, etc.), runs preflight, then:

```bash
python -m pytest tests/integration/test_gpu_channels.py -v -m gpu
```

Artifacts land in `tests/integration_runs/gpu-<timestamp>/` (gitignored). If you are already on a GPU node:

```bash
./scripts/run_gpu_integration.sh --local
```
