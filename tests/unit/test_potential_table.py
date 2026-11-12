"""Potential channel: Amber assignment, unknown residues, off switches (no HF, no APBS)."""

from pathlib import Path

import numpy as np
import pytest

from emsuite.config.schemas import ConfigValidationError
from emsuite.potential.amber import assign_amber
from emsuite.potential.spec import DEFAULT_FORCEFIELD, PotentialSpec
from emsuite.potential.structure import load_structure
from emsuite.potential.table import build_parameter_table

DATA = Path(__file__).parent / "data" / "potential"
MIX = str(DATA / "ala_acetate.pdb")  # ACE-ALA-NME (known) + acetate ACT (unknown)
TWO = str(DATA / "ala_two_acetates.pdb")  # lone zwitterion ALA + two acetates (all unknown)
SEP = str(DATA / "phosphoserine.pdb")  # ACE-SEP-NME: unknown residue bonded to known ones


class FakeResp:
    """Stands in for the HF/RESP calculation: spreads the total charge evenly."""

    def __init__(self) -> None:
        self.labels: list[str] = []

    def __call__(self, atoms, charge, spin, label):
        self.labels.append(label)
        return np.full(len(atoms), charge / len(atoms))


def _spec(pdb: str, **kwargs) -> PotentialSpec:
    return PotentialSpec(pdb=pdb, surface_file="unused.surf", **kwargs)


def test_amber_values_for_capped_alanine():
    structure = load_structure(MIX)
    amber = assign_amber(structure, DEFAULT_FORCEFIELD)
    by_name = {
        (structure.res_names[i], structure.atom_names[i]): (amber.charges[i], amber.radii[i])
        for i in range(structure.n_atoms)
    }
    assert by_name[("ALA", "N")] == pytest.approx((-0.4157, 1.824), abs=1e-4)
    assert by_name[("ALA", "H")] == pytest.approx((0.2719, 0.6), abs=1e-4)
    assert by_name[("ALA", "CA")] == pytest.approx((0.0337, 1.908), abs=1e-4)
    assert by_name[("ACE", "CH3")] == pytest.approx((-0.3662, 1.908), abs=1e-4)
    # the acetyl cap fits both the protein ACE and the GLYCAM ACX template: ACE must win
    assert sorted(amber.templates.values()) == ["ACE", "ALA", "NME"]
    assert [structure.residue_label(r) for r in amber.unmatched] == ["ACT:A:4"]
    assert np.isnan(amber.charges[structure.residue_atoms(amber.unmatched[0])]).all()


def test_table_combines_amber_and_resp():
    resp = FakeResp()
    table = build_parameter_table(_spec(MIX, unknown_charge={"ACT": -1}), resp=resp)
    s = table.structure
    act = np.array([name == "ACT" for name in s.res_names])
    assert resp.labels == ["ACT:A:4"] and table.hf_runs == 1
    assert table.charges[act].sum() == pytest.approx(-1.0)
    assert table.charges[~act].sum() == pytest.approx(0.0, abs=1e-6)
    radii = {s.elements[i]: table.radii[i] for i in np.flatnonzero(act)}
    assert radii == pytest.approx({"C": 1.70, "O": 1.52, "H": 1.20})
    sources = {row["residue"]: row["source"] for row in table.residue_rows()}
    assert sources["ALA"] == "Amber template: ALA"
    assert sources["ACT"].startswith("RESP HF/6-31G*")


def test_missing_charge_stops_before_any_hf():
    resp = FakeResp()
    with pytest.raises(ConfigValidationError, match="unknown_charge = {'ACT'"):
        build_parameter_table(_spec(MIX), resp=resp)
    assert resp.labels == []


def test_impossible_charge_and_spin_stop():
    with pytest.raises(ConfigValidationError, match="cannot have spin 0"):
        build_parameter_table(_spec(MIX, unknown_charge={"ACT": 0}), resp=FakeResp())


def test_unknown_residue_bonded_to_protein_stops():
    with pytest.raises(ConfigValidationError, match="not supported yet"):
        build_parameter_table(_spec(SEP, unknown_charge={"SEP": -1}), resp=FakeResp())


def test_off_residue_keeps_radius_and_needs_no_hf():
    resp = FakeResp()
    table = build_parameter_table(_spec(MIX, off=["ACT", "ALA:A:2@N"]), resp=resp)
    s = table.structure
    act = np.array([name == "ACT" for name in s.res_names])
    assert resp.labels == []
    assert (table.charges[act] == 0).all()
    assert {s.elements[i]: table.radii[i] for i in np.flatnonzero(act)} == pytest.approx(
        {"C": 1.70, "O": 1.52, "H": 1.20}
    )
    n = next(i for i in range(s.n_atoms) if (s.res_names[i], s.atom_names[i]) == ("ALA", "N"))
    assert table.charges[n] == 0 and table.radii[n] == pytest.approx(1.824, abs=1e-4)
    assert table.off.sum() == 8 and table.removed.sum() == 0


def test_removed_residue_gets_zero_radius_and_no_hf():
    resp = FakeResp()
    table = build_parameter_table(_spec(MIX, removed=["ACT"]), resp=resp)
    act = np.array([name == "ACT" for name in table.structure.res_names])
    assert resp.labels == []
    assert (table.charges[act] == 0).all() and (table.radii[act] == 0).all()
    assert (table.radii[~act] > 0).all()
    assert table.removed.sum() == 7 and table.off.sum() == 0


def test_removed_wins_over_off():
    table = build_parameter_table(_spec(MIX, off=["ACT"], removed=["ACT"]), resp=FakeResp())
    act = np.array([name == "ACT" for name in table.structure.res_names])
    assert (table.radii[act] == 0).all() and table.off.sum() == 0


def test_switches_on_single_atom_and_known_residue():
    table = build_parameter_table(
        _spec(MIX, unknown_charge={"ACT": -1}, removed=["ACT@O3"], off=["ALA:A:2"]),
        resp=FakeResp(),
    )
    s = table.structure
    gone = {(s.res_names[i], s.atom_names[i]) for i in np.flatnonzero(table.removed)}
    assert gone == {("ACT", "O3")}
    assert table.off.sum() == 10
    assert (table.charges[table.off | table.removed] == 0).all()
    assert (table.radii[table.removed] == 0).all() and (table.radii[table.off] > 0).all()


def test_off_entry_matching_nothing_stops():
    with pytest.raises(ConfigValidationError, match="matches no atom"):
        build_parameter_table(_spec(MIX, off=["XYZ", "ACT"]), resp=FakeResp())


@pytest.mark.parametrize(("mode", "runs"), [("each", 3), ("one", 2)])
def test_resp_copies(mode: str, runs: int):
    resp = FakeResp()
    table = build_parameter_table(
        _spec(TWO, unknown_charge={"ACT": -1, "ALA": 0}, resp_copies=mode), resp=resp
    )
    assert len(resp.labels) == runs
    reused = [row for row in table.residue_rows() if "reused" in str(row["source"])]
    assert len(reused) == (1 if mode == "one" else 0)
    assert table.charges.sum() == pytest.approx(-2.0)


def test_missing_hydrogens_stop_then_add_hydrogens(tmp_path: Path):
    stripped = tmp_path / "no_h.pdb"
    keep = [
        line
        for line in Path(MIX).read_text().splitlines(keepends=True)
        if not (line.startswith(("ATOM", "HETATM")) and line[76:78].strip() == "H")
        or line[17:20] == "ACT"
    ]
    stripped.write_text("".join(keep))
    with pytest.raises(ConfigValidationError, match="probably missing hydrogens"):
        build_parameter_table(_spec(str(stripped), off=["ACT"]), resp=FakeResp())
    table = build_parameter_table(
        _spec(str(stripped), off=["ACT"], add_hydrogens=True), resp=FakeResp()
    )
    assert table.structure.hydrogens_added == 12
    assert table.structure.n_atoms == load_structure(MIX).n_atoms
    assert table.charges.sum() == pytest.approx(0.0, abs=1e-6)


def test_pqr_and_sources_files(tmp_path: Path):
    table = build_parameter_table(_spec(MIX, off=["ACT"]), resp=FakeResp())
    pqr = table.write_pqr(tmp_path / "out.pqr").read_text().splitlines()
    assert len(pqr) == table.structure.n_atoms
    first = pqr[0].split()
    assert first[:6] == ["ATOM", "1", "CH3", "ACE", "A", "1"] and len(first) == 11
    assert float(first[-2]) == pytest.approx(-0.3662) and float(first[-1]) == pytest.approx(1.908)
    sources = table.write_sources(tmp_path / "sources.csv").read_text().splitlines()
    assert sources[0] == ("residue,chain,number,atoms,source,atoms_off,atoms_removed,net_charge_e")
    assert len(sources) == 5
