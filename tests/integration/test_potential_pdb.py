"""Potential channel on a PDB: Amber for known residues, RESP for the unknown one."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from emsuite import PotentialInput

from .helpers import fingerprint_surf, record_assertions

POTENTIAL_IN = """\
pdb = 'complex.pdb'
surface_file = 'acetate.surf'
output_surf = 'acetate_potential.surf'
quantity = 'potential'
off = ['ACT']
"""


@pytest.mark.slow
def test_potential_from_input_file(pdb_complex: Path) -> None:
    """The acetate is off (charge 0, radius kept): no quantum calculation is needed."""
    (pdb_complex / "potential.in").write_text(POTENTIAL_IN)
    result = PotentialInput.from_file("potential.in").run()
    assert result.quantity == "potential"
    assert len(result.values) >= 10 and np.all(np.isfinite(result.values))
    for suffix in (".surf", ".mol2", ".csv", ".pqr"):
        assert Path("acetate_potential").with_suffix(suffix).is_file()
    rows = list(csv.DictReader(open("acetate_potential_sources.csv")))
    assert [row["source"] for row in rows[:3]] == [
        "Amber template: ACE",
        "Amber template: ALA",
        "Amber template: NME",
    ]
    assert rows[3]["residue"] == "ACT" and "no RESP run" in rows[3]["source"]
    assert rows[3]["atoms_off"] == "7"
    record_assertions(
        pdb_complex,
        channel="potential",
        quantity="potential",
        surface_points=len(result.values),
        surf=fingerprint_surf(result.path),
    )


@pytest.mark.slow
def test_potential_with_resp_for_unknown_residue(pdb_complex: Path) -> None:
    """The acetate keeps its charges: RESP runs on it (small basis to keep the test fast)."""
    result = PotentialInput(
        pdb="complex.pdb",
        surface_file="acetate.surf",
        output_surf="acetate_resp.surf",
        unknown_charge={"ACT": -1},
        unknown_basis="sto-3g",
    ).run()
    assert np.all(np.isfinite(result.values))
    rows = list(csv.DictReader(open("acetate_resp_sources.csv")))
    assert rows[3]["source"].startswith("RESP HF/STO-3G")
    assert float(rows[3]["net_charge_e"]) == pytest.approx(-1.0, abs=1e-3)
    # an anion on its own surface: the potential is negative everywhere
    assert result.values.max() < 0


@pytest.mark.slow
def test_off_and_removed_give_different_maps(pdb_complex: Path) -> None:
    common = dict(pdb="complex.pdb", surface_file="acetate.surf")
    off = PotentialInput(output_surf="off.surf", off=["ACT"], **common).run()
    removed = PotentialInput(output_surf="removed.surf", removed=["ACT"], **common).run()
    pqr = {name: Path(name).with_suffix(".pqr").read_text().split() for name in ("off", "removed")}
    assert pqr["off"][-1] == "1.2000" and pqr["removed"][-1] == "0.0000"
    assert not np.allclose(off.values, removed.values)


@pytest.mark.slow
def test_gauss_charge_and_grid_spacing(pdb_complex: Path) -> None:
    result = PotentialInput(
        pdb="complex.pdb",
        surface_file="acetate.surf",
        output_surf="acetate_charge.surf",
        quantity="charge",
        off=["ACT"],
        grid_spacing=0.4,
    ).run()
    assert result.quantity == "charge"
    assert np.all(np.isfinite(result.values))
