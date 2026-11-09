"""A failed surface point must not lose the map; thermo baselines come out in kcal/mol.

Real QM (methane, sto-3g, serial). One point is forced to fail by wrapping the
per-point calculator, which is what happened for real in jobs 4635222/31/35.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from emsuite import TuningInput
from emsuite.tuning import runner

from .helpers import install_methane_surf, write_methane_xyz

TUNING_IN = """\
molecule = 'methane.xyz'
surface_file = 'methane.surf'
properties = ['homo', 'ie', 'ea', 'cp', 'hard']
basis_set = 'sto-3g'
method = 'dft'
functional = 'b3lyp'
charge = 0
spin = 0
solvent = None
calc_type = 'separate'
parallel = False
"""

FAIL_AT_CALL = 3  # 1-based call index of the point forced to fail


@pytest.mark.slow
def test_failed_point_keeps_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_methane_xyz(tmp_path)
    surf = Path(install_methane_surf(tmp_path))
    n_points = len(surf.read_text().strip().splitlines()) - 1
    (tmp_path / "tuning.in").write_text(TUNING_IN)

    real = runner.calculate_surface_effect_at_point
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == FAIL_AT_CALL:
            raise RuntimeError("forced failure for test")
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "calculate_surface_effect_at_point", flaky)

    TuningInput.from_file("tuning.in").run()  # must not raise

    results_dir = sorted(tmp_path.glob("results_methane_*"))[-1]
    rows = list(csv.DictReader((results_dir / "methane_tuning_summary.csv").open()))
    assert len(rows) == n_points
    failed = FAIL_AT_CALL - 1
    assert rows[failed]["homo_effect"] == ""
    assert all(r["homo_effect"] != "" for i, r in enumerate(rows) if i != failed)
    assert (results_dir / "methane_homo_normalized.mol2").is_file()

    # Thermo baselines in documented units. Methane B3LYP/sto-3g vertical IE is
    # ~0.45-0.55 Ha, i.e. ~280-350 kcal/mol; Hartree-valued output would be < 1.
    ie = float(rows[0]["ie_baseline"])
    assert 200 < ie < 450, ie
    # Hardness (IE-EA)/2 in eV; sto-3g methane gives ~15 eV (unbound anion).
    hard_ev = float(rows[0]["hard_baseline"])
    assert 5 < hard_ev < 25, hard_ev
