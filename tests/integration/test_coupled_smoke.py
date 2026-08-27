"""Coupled potential → tuning smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite.coupled import run_coupled_calculation

from .helpers import METHANE_SURFACE_IN, record_assertions

COUPLED_IN = """\
molecule = 'methane.xyz'
output_surf = 'coupled.surf'
surface_density = 0.5
potential_method = 'apbs'
properties = ['homo', 'lumo']
basis_set = 'sto-3g'
method = 'dft'
functional = 'b3lyp'
charge = 0
spin = 0
calc_type = 'separate'
parallel = False
"""

SURFACE_IN = METHANE_SURFACE_IN


@pytest.mark.slow
def test_coupled_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    from emsuite.surface import run_surface_calculation

    (tmp_path / "surface.in").write_text(SURFACE_IN)
    run_surface_calculation("surface.in")
    (tmp_path / "coupled.in").write_text(COUPLED_IN)
    run_coupled_calculation("coupled.in")
    results = list(tmp_path.glob("results_methane_*"))
    assert results
    assert not list(tmp_path.glob("coupled_*.in"))

    record_assertions(
        tmp_path,
        coupled_results_dir=str(results[-1]),
        properties=["homo", "lumo"],
    )
