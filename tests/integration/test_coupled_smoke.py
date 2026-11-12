"""Coupled potential → tuning smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite import CoupledInput

from .helpers import record_assertions

COUPLED_IN = """\
molecule = 'acetate.xyz'
pdb = 'complex.pdb'
surface_file = 'acetate.surf'
output_surf = 'coupled.surf'
potential_method = 'apbs'
off = ['ACT']
properties = ['homo', 'lumo']
basis_set = 'sto-3g'
method = 'dft'
functional = 'b3lyp'
charge = -1
spin = 0
calc_type = 'separate'
parallel = False
"""


@pytest.mark.slow
def test_coupled_pipeline(pdb_complex: Path) -> None:
    (pdb_complex / "coupled.in").write_text(COUPLED_IN)
    result = CoupledInput.from_file("coupled.in").run()
    results = Path(result.tuning.results_dir)
    assert results.is_dir()
    assert not list(pdb_complex.glob("coupled_*.in"))
    assert Path(result.potential.path).is_file()
    assert Path(result.potential.path).with_suffix(".mol2").is_file()
    assert (Path(result.run_dir) / "run.json").is_file()

    record_assertions(
        pdb_complex,
        coupled_results_dir=str(results),
        properties=["homo", "lumo"],
    )


@pytest.mark.slow
def test_coupled_generates_surface_then_runs_apbs_and_tuning(pdb_complex: Path) -> None:
    """Real surface generator, APBS and small CPU HF calculation, no supplied surface."""
    import json

    import numpy as np

    from emsuite.surface.io import surface_quantity

    original = (pdb_complex / "acetate.xyz").read_bytes()
    result = CoupledInput(
        molecule="acetate.xyz",
        pdb="complex.pdb",
        off=["ACT"],
        surface_density=0.005,
        properties=("homo",),
        basis_set="sto-3g",
        method="hf",
        # Same acetate anion as COUPLED_IN and test_gpu_coupled_parallel.
        charge=-1,
        calc_type="combined",
        parallel=False,
    ).run()
    assert result.surface is not None
    assert len(result.surface.coords) > 0 and np.isfinite(result.potential.values).all()
    np.testing.assert_array_equal(result.surface.coords, result.potential.coords)
    assert surface_quantity(result.potential.path) == "charge"
    assert (pdb_complex / "acetate.xyz").read_bytes() == original
    assert Path.cwd() == pdb_complex
    manifest = json.loads((Path(result.run_dir) / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert Path(result.tuning.results_dir, "acetate_tuning_summary.csv").is_file()
