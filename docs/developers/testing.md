---
icon: lucide/flask-conical
---

# Testing and Validation

## Fast tests

```bash
uv run --extra dev pytest tests/unit tests/regression -q
```

Unit tests cover validation, typed inputs/results, parsing, property scheduling,
and pure transformations. Regression tests protect stable scientific and file
semantics without requiring every expensive backend.

## Integration tests

```bash
uv run --extra dev pytest tests/integration -q
```

Some integration tests invoke PySCF, APBS, Ray, PDB2PQR, or platform resources.
Use markers and focused paths while developing.

## Scientific reproduction

A defensible comparison records:

- source commit and clean/dirty status;
- Python, PySCF, NumPy, and SciPy versions;
- input hashes;
- backend and worker configuration;
- convergence and failed-point counts;
- exact structural comparison plus bounded numeric differences.

Ignore timestamps and other presentation metadata unless they affect recovery.

## Documentation tests

```bash
uv sync --extra docs
uv run zensical build --clean --strict
```

Validate code examples against the current CLI and typed API. Avoid executing
expensive chemistry solely to test page rendering.
