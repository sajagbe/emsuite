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
    CoupledInput.from_file("coupled.in").run()
    results = list(pdb_complex.glob("results_acetate_*"))
    assert results
    assert not list(pdb_complex.glob("coupled_*.in"))
    assert (pdb_complex / "coupled.surf").is_file() and (pdb_complex / "coupled.mol2").is_file()

    record_assertions(
        pdb_complex,
        coupled_results_dir=str(results[-1]),
        properties=["homo", "lumo"],
    )
