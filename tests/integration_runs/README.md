# Integration run logs

Slow integration tests can archive a full, traceable record of every run here.

## Run a full audited suite

```bash
python scripts/run_slow_integration_audit.py
```

Each run creates a timestamped folder, for example `2026-06-21T153045Z/`, containing:

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
