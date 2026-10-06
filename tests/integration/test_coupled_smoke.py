"""Coupled potential → tuning smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite import CoupledInput

from .helpers import install_methane_surf, record_assertions, write_methane_xyz

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


@pytest.mark.slow
def test_coupled_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_methane_xyz(tmp_path)
    install_methane_surf(tmp_path)
    (tmp_path / "coupled.in").write_text(COUPLED_IN)
    CoupledInput.from_file("coupled.in").run()
    results = list(tmp_path.glob("results_methane_*"))
    assert results
    assert not list(tmp_path.glob("coupled_*.in"))

    record_assertions(
        tmp_path,
        coupled_results_dir=str(results[-1]),
        properties=["homo", "lumo"],
    )
