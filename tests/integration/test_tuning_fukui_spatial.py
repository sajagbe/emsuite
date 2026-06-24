"""v1.2 spatial Fukui surface-map integration test."""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite.surface import run_surface_calculation
from emsuite.tuning import main as run_tuning

from .helpers import (
    latest_results_dir,
    prepare_methane_surface,
    record_assertions,
    tuning_in,
)


@pytest.mark.slow
def test_tuning_fukui_spatial_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prepare_methane_surface(tmp_path)
    run_surface_calculation("surface.in")

    (tmp_path / "tuning.in").write_text(
        tuning_in(
            ["fukui_spa_plus", "fukui_spa_minus"],
            extra_lines="fukui_projection = 'nearest'\n",
        )
    )
    run_tuning("tuning.in")

    results_dir = latest_results_dir(tmp_path)
    summary = (results_dir / "methane_tuning_summary.csv").read_text()
    assert "fukui_spa_plus_effect" in summary
    assert "fukui_spa_minus_effect" in summary
    assert (results_dir / "methane_fukui_spa_plus.mol2").is_file()
    assert (results_dir / "methane_fukui_spa_minus_normalized.mol2").is_file()

    record_assertions(
        tmp_path,
        fukui_spa_plus_mol2=True,
        fukui_spa_minus_mol2=True,
        csv_has_spatial_effects=True,
        results_dir=str(results_dir),
    )
