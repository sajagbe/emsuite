"""v1.2 transition-state barrier integration test."""

from __future__ import annotations

import shutil

import pytest

from emsuite.surface import run_surface_calculation
from emsuite.tuning import main as run_tuning

from .helpers import latest_results_dir, prepare_methane_surface, record_assertions, tuning_in


@pytest.mark.slow
def test_tuning_ts_barrier_global(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prepare_methane_surface(tmp_path)
    run_surface_calculation("surface.in")

    # Use identical geometry as a smoke TS; barrier should be ~0 kcal/mol.
    shutil.copy("methane.xyz", "methane_ts.xyz")

    (tmp_path / "tuning.in").write_text(
        tuning_in(["homo", "ts_barrier"], extra_lines="ts_xyz = 'methane_ts.xyz'\n")
    )
    run_tuning("tuning.in")

    results_dir = latest_results_dir(tmp_path)
    log_summary = (results_dir / "logs" / "calculation_summary.out").read_text()
    assert "ts_barrier" in log_summary

    record_assertions(
        tmp_path,
        ts_barrier_in_summary_log=True,
        ts_xyz_used="methane_ts.xyz",
        results_dir=str(results_dir),
    )
