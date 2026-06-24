"""v1.2 vibrational frequency integration test."""

from __future__ import annotations

import pytest

from emsuite.surface import run_surface_calculation
from emsuite.tuning import main as run_tuning

from .helpers import latest_results_dir, prepare_methane_surface, record_assertions, tuning_in


@pytest.mark.slow
def test_tuning_fundamental_frequency(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prepare_methane_surface(tmp_path)
    run_surface_calculation("surface.in")

    (tmp_path / "tuning.in").write_text(tuning_in(["freq"]))
    run_tuning("tuning.in")

    results_dir = latest_results_dir(tmp_path)
    summary = (results_dir / "methane_tuning_summary.csv").read_text()
    assert "freq_effect" in summary
    assert (results_dir / "methane_freq.mol2").is_file()

    log_summary = (results_dir / "logs" / "calculation_summary.out").read_text()
    assert "freq" in log_summary.lower()

    record_assertions(
        tmp_path,
        freq_effect_in_csv=True,
        freq_mol2=True,
        raw_freq_logged=True,
        results_dir=str(results_dir),
    )
