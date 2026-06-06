# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

EMSuite is a Python CLI for computational chemistry (electrostatic tuning maps). There is no web server, database, or Docker stack. Development is: install the package, then run `emsuite -s` (surface generation) and `emsuite -t` (tuning calculations).

### PATH

`pip install -e .` installs the `emsuite` and `vsg` binaries to `~/.local/bin`. Ensure this is on `PATH` before running commands:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Install

From the repo root (see `README.md` and `docs/installation.rst`):

```bash
pip install -e .          # CPU / editable dev install
pip install -e ".[gpu]"   # optional NVIDIA GPU acceleration (CUDA 12.x)
```

Python **3.11+** is required (`.python-version` pins 3.13; 3.12 works). A `uv.lock` exists but the documented install path is `pip`.

### Running the application

Two-stage workflow:

```bash
emsuite -s surface.in   # generate .surf from SMILES or XYZ
emsuite -t tuning.in    # QM/MM tuning over surface points
```

Templates: `templates/surface.in`, `templates/tuning.in`. Full example inputs/outputs: `examples/tuning/CCO2-exe/`. Sample XYZ geometries: `molecules/`.

### Services

| Component | Notes |
|-----------|--------|
| **EMSuite CLI** | Only required runtime; entry point `emsuite` |
| **Ray** | Started automatically when `parallel = True` in `tuning.in`; set `parallel = False` to avoid Ray overhead for small/local runs |
| **`vsg` binary** | Bundled via `vdw-surfgen`; must be on PATH (installed with the package) |

No separate servers to start.

### Lint / tests

This repo has **no** configured linter (ruff/flake8/mypy) and **no** automated test suite (`tests/` is absent). Reasonable sanity checks:

```bash
python3 -m compileall -q src/emsuite
python3 -c "import emsuite"
emsuite -h
```

### Docs (optional)

```bash
cd docs && pip install -r requirements.txt && make html
```

### Gotchas

- **Tuning completion** calls an external Office quote API (`core.print_office_quote()`). Surface generation does not. Tuning runs need outbound HTTPS for a clean exit; failures raise after calculations finish.
- **GPU**: Without `[gpu]` extra, runs use CPU PySCF only (expected in cloud VMs without NVIDIA GPUs).
- **Runtime**: Tuning cost scales with surface point count × properties. For quick smoke tests use a small molecule (e.g. water `O`), low `surface_density`, `parallel = False`, and a minimal basis (e.g. `sto-3g`).
- **Outputs**: Tuning writes timestamped `results_<molecule>_<timestamp>/` directories in the working directory.
