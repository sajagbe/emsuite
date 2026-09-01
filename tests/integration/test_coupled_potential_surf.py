"""CoupledInput.potential_surf skips potential recompute, reused across calc_type."""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite.inputs import CoupledInput, SurfaceInput

from .helpers import METHANE_SURFACE_IN, record_assertions

SURFACE_IN = METHANE_SURFACE_IN


@pytest.mark.slow
def test_coupled_reuses_potential_surf_across_calc_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "surface.in").write_text(SURFACE_IN)
    surf = SurfaceInput.from_file(tmp_path / "surface.in").run()
    assert Path(surf.path).is_file()

    # No protein/ligand_atoms/potential_method given — would fail potential-channel
    # validation if it ran. potential_surf skips that entirely.
    common = dict(
        molecule="methane.xyz",
        potential_surf=surf.path,
        properties=["homo", "lumo"],
        basis_set="sto-3g",
        parallel=False,
    )

    separate = CoupledInput.from_config(calc_type="separate", **common).run()
    combined = CoupledInput.from_config(calc_type="combined", **common).run()

    assert separate.potential.path == surf.path
    assert combined.potential.path == surf.path
    assert not list(tmp_path.glob("coupled_*.in"))
    # Potential recompute would have written its own coupled.surf/csv; confirm absence.
    assert not (tmp_path / "coupled.surf").exists()

    for result in (separate, combined):
        assert result.tuning.results_dir
        assert Path(result.tuning.results_dir).is_dir()
    # results_dir is a second-precision timestamp (tuning/output.py), not calc_type-qualified,
    # so don't assert inequality here — just that both runs produced real results.

    record_assertions(
        tmp_path,
        separate_results_dir=separate.tuning.results_dir,
        combined_results_dir=combined.tuning.results_dir,
        potential_surf_reused=surf.path,
    )
