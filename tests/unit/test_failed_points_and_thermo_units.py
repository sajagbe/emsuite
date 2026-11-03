"""Failed surface points must not crash output; thermo props use documented units."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from emsuite.physconst import HARTREE_TO_EV, HARTREE_TO_KCAL
from emsuite.tuning.output import create_output_files, normalize_effects
from emsuite.tuning.properties.thermo import calculate_thermo_properties


def test_normalize_effects_skips_failed_points():
    effects = [{"a_effect": -1.0}, None, {"a_effect": 3.0}]
    normalized, params = normalize_effects(effects, ["a_effect"])
    assert params["a_effect"] == (-1.0, 3.0)
    assert normalized[1] is None
    assert normalized[0]["a_effect"] == pytest.approx(-1.0)
    assert normalized[2]["a_effect"] == pytest.approx(1.0)


def test_create_output_files_with_failed_point(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    coords = np.zeros((3, 3))
    effects = [{"s1_exe_effect": 0.1}, None, {"s1_exe_effect": -0.2}]
    create_output_files(coords, effects, "mol", ["exe"], {"s1_exe": 2.0})
    rows = list(csv.DictReader(open(tmp_path / "mol_tuning_summary.csv")))
    assert len(rows) == 3
    assert rows[1]["s1_exe_effect"] == ""  # failed point left blank, not 0.0
    assert float(rows[2]["s1_exe_effect"]) == pytest.approx(-0.2)
    assert (tmp_path / "mol_s1_exe.mol2").exists()


def test_thermo_units_match_readme():
    mf = SimpleNamespace(e_tot=-1.0)
    cation = SimpleNamespace(e_tot=-0.7)  # ie = 0.3 Ha
    anion = SimpleNamespace(e_tot=-1.1)  # ea = 0.1 Ha
    props = ["ie", "ea", "cp", "eng", "hard", "efl", "nfl"]
    r = calculate_thermo_properties(mf, anion, cation, props)
    assert r["ie"] == pytest.approx(0.3 * HARTREE_TO_KCAL)
    assert r["ea"] == pytest.approx(0.1 * HARTREE_TO_KCAL)
    assert r["cp"] == pytest.approx(-0.2 * HARTREE_TO_KCAL)
    assert r["eng"] == pytest.approx(0.2 * HARTREE_TO_EV)
    assert r["hard"] == pytest.approx(0.1 * HARTREE_TO_EV)
    efl_ev = (0.2**2 / (2 * 0.1)) * HARTREE_TO_EV
    assert r["efl"] == pytest.approx(efl_ev)
    assert r["nfl"] == pytest.approx(1 / efl_ev)
