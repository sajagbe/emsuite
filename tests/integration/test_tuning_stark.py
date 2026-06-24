"""v1.2 Stark effect integration test."""

from __future__ import annotations

import pytest

from emsuite.surface import run_surface_calculation
from emsuite.tuning import main as run_tuning

from .helpers import latest_results_dir, prepare_methane_surface, record_assertions, tuning_in


@pytest.mark.slow
def test_tuning_stark_gap(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prepare_methane_surface(tmp_path)
    run_surface_calculation("surface.in")

    (tmp_path / "tuning.in").write_text(tuning_in(["stark_gap"]))
    run_tuning("tuning.in")

    results_dir = latest_results_dir(tmp_path)
    summary = (results_dir / "methane_tuning_summary.csv").read_text()
    assert "stark_gap_effect" in summary
    assert (results_dir / "methane_stark_gap.mol2").is_file()

    record_assertions(
        tmp_path,
        stark_gap_effect_in_csv=True,
        stark_gap_mol2=True,
        results_dir=str(results_dir),
    )
