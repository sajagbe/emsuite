"""v1.2 water probe interaction integration test."""

from __future__ import annotations

import pytest

from emsuite.surface import run_surface_calculation
from emsuite.tuning import main as run_tuning

from .helpers import latest_results_dir, prepare_methane_surface, record_assertions, tuning_in


@pytest.mark.slow
def test_tuning_water_probe_interaction(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prepare_methane_surface(tmp_path)
    run_surface_calculation("surface.in")

    (tmp_path / "tuning.in").write_text(tuning_in(["h2o"]))
    run_tuning("tuning.in")

    results_dir = latest_results_dir(tmp_path)
    summary = (results_dir / "methane_tuning_summary.csv").read_text()
    assert "h2o_effect" in summary
    assert (results_dir / "methane_h2o.mol2").is_file()

    record_assertions(
        tmp_path,
        h2o_effect_in_csv=True,
        h2o_mol2=True,
        results_dir=str(results_dir),
    )
