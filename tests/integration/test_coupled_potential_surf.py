"""CoupledInput.potential_surf skips potential recompute, reused across calc_type."""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite.inputs import CoupledInput

from .helpers import install_methane_surf, record_assertions, write_methane_xyz


@pytest.mark.slow
def test_coupled_reuses_potential_surf_across_calc_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_methane_xyz(tmp_path)
    surf_path = install_methane_surf(tmp_path)

    # No protein/ligand_atoms/potential_method given — would fail potential-channel
    # validation if it ran. potential_surf skips that entirely.
    common = dict(
        molecule="methane.xyz",
        potential_surf=str(surf_path),
        properties=["homo", "lumo"],
        basis_set="sto-3g",
        parallel=False,
    )

    separate = CoupledInput(calc_type="separate", **common).run()
    combined = CoupledInput(calc_type="combined", **common).run()

    assert separate.potential.path == str(surf_path)
    assert combined.potential.path == str(surf_path)
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
        potential_surf_reused=str(surf_path),
    )
