"""Consistency-matrix cases missing from older slow smokes.

Covers: heterogeneous surface, APBS Gauss-law charge (CPU), tuning combined.
Numeric cases use the committed ``fixtures/methane.surf`` because ``vsg`` point
sets are not bit-stable across runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emsuite import SurfaceInput
from emsuite.inputs import PotentialInput, TuningInput
from emsuite.surface.io import load_surf, save_surf

from .helpers import (
    METHANE_SURFACE_HETERO_IN,
    fingerprint_csv,
    fingerprint_surf,
    install_methane_surf,
    latest_results_dir,
    record_assertions,
    write_methane_xyz,
)


@pytest.mark.slow
def test_surface_heterogenous_generates_edit_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Heterogeneous VDW surface: zero placeholder charges + edit-header comment."""
    monkeypatch.chdir(tmp_path)
    write_methane_xyz(tmp_path)
    (tmp_path / "surface.in").write_text(METHANE_SURFACE_HETERO_IN)
    result = SurfaceInput.from_file("surface.in").run()
    surf_path = result.path
    assert Path(surf_path).is_file()

    header = Path(surf_path).read_text().splitlines()[0]
    assert "EDIT CHARGES" in header

    coords, values = load_surf(surf_path)
    assert len(coords) >= 10
    assert np.allclose(values, 0.0)

    # Non-uniform edit must round-trip (user workflow before tuning/coupled).
    # .surf I/O formats charges to 6 decimals — compare at that precision.
    edited = np.linspace(-0.05, 0.05, num=len(values)).round(6)
    save_surf(coords, edited, surf_path, heterogenous=True)
    _, again = load_surf(surf_path)
    assert np.allclose(again, edited, atol=1e-6)

    record_assertions(
        tmp_path,
        channel="surface",
        surface_type="heterogenous",
        # vsg coords are nondeterministic — only assert structural fields here.
        n_points=int(len(values)),
        edited_values_sum=float(np.sum(edited)),
    )


@pytest.mark.slow
def test_potential_apbs_gauss_charge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU APBS → Gauss-law surface charges (quantity=charge)."""
    monkeypatch.chdir(tmp_path)
    write_methane_xyz(tmp_path)
    surf = install_methane_surf(tmp_path)

    result = PotentialInput(
        molecule="methane.xyz",
        surface_file=str(surf),
        output_surf="methane_charge.surf",
        method="apbs",
        quantity="charge",
    ).run()
    assert Path(result.path).is_file()
    assert result.quantity == "charge"
    assert np.all(np.isfinite(result.values))
    assert len(result.values) >= 10

    record_assertions(
        tmp_path,
        channel="potential",
        quantity="charge",
        surf=fingerprint_surf(result.path),
    )


@pytest.mark.slow
def test_tuning_combined_methane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone tuning with calc_type=combined on methane (one effect for all points)."""
    monkeypatch.chdir(tmp_path)
    write_methane_xyz(tmp_path)
    install_methane_surf(tmp_path)

    TuningInput(
        molecule="methane.xyz",
        surface_file="methane.surf",
        properties=["homo", "lumo", "gap"],
        basis_set="sto-3g",
        method="dft",
        functional="b3lyp",
        calc_type="combined",
        parallel=False,
    ).run()

    results_dir = latest_results_dir(tmp_path)
    summary_csv = results_dir / "methane_tuning_summary.csv"
    assert summary_csv.is_file()
    csv_lines = summary_csv.read_text().strip().splitlines()
    # Combined emits a single data row (mean coordinate / total charge).
    assert len(csv_lines) == 2

    for prop in ("homo", "lumo", "gap"):
        assert (results_dir / f"methane_{prop}.mol2").is_file()
        assert (results_dir / f"methane_{prop}_normalized.mol2").is_file()

    record_assertions(
        tmp_path,
        channel="tuning",
        calc_type="combined",
        properties=["homo", "lumo", "gap"],
        results_dir=str(results_dir),
        summary=fingerprint_csv(summary_csv),
    )
