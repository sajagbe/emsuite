# Engineering Guide

EMSuite refoundation follows two reference guides:

- [uv-cookiecutter](https://github.com/jevandezande/uv-cookiecutter) — project layout, `uv` workflow, Ruff, pytest, pre-commit, GitHub Actions
- [How to Make a Great Open-Source Scientific Project](https://rowansci.com/blog/how-to-make-a-great-open-source-scientific-project) — principles for minimal, packaged, clean, tested scientific Python

## Package structure

```
src/emsuite/
  __init__.py        # version + public Input/Result/API exports
  api.py             # keyword API → Input.run()
  geometry.py        # Geometry.from_xyz
  inputs.py          # SurfaceInput, PotentialInput, TuningInput, CoupledInput
  results.py         # SurfaceResult, PotentialResult, TuningResult, CoupledResult
  config/            # parser, schemas, resolve
  cli/               # main.py (-s -t -p -c)
  core/              # hardware, molecule, qmmm, excited, io
  engines/           # base.py, pyscf_engine.py
  surface/           # io, vdw, optimize, generate, runner
  tuning/
    properties/      # registry, ground_state, excited_state, thermo
    parallel.py      # Ray workers
    logging.py, output.py, resume.py, config_io.py, runner.py
  potential/         # apbs, pqr, charges, dx, gauss, occupancy, runner
  coupled/           # in-memory potential → tuning
```

## Testing

```bash
uv sync --extra dev
uv run pytest tests/unit tests/regression -v    # freeze line
uv run pytest tests/integration -v -m slow      # optional APBS/PySCF
uv run pre-commit run --all-files
```

CI (`.github/workflows/tests.yml`) runs unit + regression on every push; integration on all pushes.

## uv-cookiecutter alignment

| Feature | Status |
|---------|--------|
| `src/` layout, `uv`, Ruff, pytest, pre-commit, GHA | Done |
| Codecov | Planned |
| `ty` / mypy in CI | Planned |

## What not to do

- Do not commit local research work — use gitignored [`dev/`](../dev/README.md).
- Do not grow `tuning/runner.py` with new properties — extend `tuning/properties/`.
- Do not confuse `calc_type='combined'` with the `coupled` channel.

See [ROADMAP.md](ROADMAP.md) for the v2.x property backlog.
