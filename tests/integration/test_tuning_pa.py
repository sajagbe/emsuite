"""v1.2 proton affinity integration test."""

from __future__ import annotations

import pytest

from emsuite.surface import run_surface_calculation
from emsuite.tuning import main as run_tuning

from .helpers import latest_results_dir, prepare_methane_surface, record_assertions, tuning_in


@pytest.mark.slow
def test_tuning_proton_affinity(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prepare_methane_surface(tmp_path)
    run_surface_calculation("surface.in")

    (tmp_path / "tuning.in").write_text(tuning_in(["pa"]))
    run_tuning("tuning.in")

    results_dir = latest_results_dir(tmp_path)
    summary = (results_dir / "methane_tuning_summary.csv").read_text()
    assert "pa_effect" in summary
    assert (results_dir / "methane_pa.mol2").is_file()

    record_assertions(
        tmp_path,
        pa_effect_in_csv=True,
        pa_mol2=True,
        results_dir=str(results_dir),
    )
