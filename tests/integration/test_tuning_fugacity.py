"""v1.2 fugacity extension integration test."""

from __future__ import annotations

import pytest

from emsuite.surface import run_surface_calculation
from emsuite.tuning import main as run_tuning

from .helpers import latest_results_dir, prepare_methane_surface, record_assertions, tuning_in


@pytest.mark.slow
def test_tuning_fugacity_extensions(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prepare_methane_surface(tmp_path)
    run_surface_calculation("surface.in")

    (tmp_path / "tuning.in").write_text(tuning_in(["efl_fug", "nfl_fug", "eng_fug"]))
    run_tuning("tuning.in")

    results_dir = latest_results_dir(tmp_path)
    summary = (results_dir / "methane_tuning_summary.csv").read_text()
    for prop in ("efl_fug", "nfl_fug", "eng_fug"):
        assert f"{prop}_effect" in summary
        assert (results_dir / f"methane_{prop}.mol2").is_file()

    record_assertions(
        tmp_path,
        fugacity_props=["efl_fug", "nfl_fug", "eng_fug"],
        all_mol2_created=True,
        results_dir=str(results_dir),
    )
