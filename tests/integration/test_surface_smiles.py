"""Surface generation from SMILES integration test."""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite.surface import run_surface_calculation

from .helpers import METHANE_SURFACE_IN, record_assertions

SURFACE_IN = METHANE_SURFACE_IN.replace("methane.surf", "smiles_surface.surf").replace(
    "methane.xyz", "smiles_methane.xyz"
)


@pytest.mark.slow
def test_surface_smiles_generates_surf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "surface.in").write_text(SURFACE_IN)
    surf_path = run_surface_calculation("surface.in")
    assert Path(surf_path).is_file()
    lines = Path(surf_path).read_text().strip().splitlines()
    assert len(lines) >= 11

    record_assertions(
        tmp_path,
        surf_path=surf_path,
        surface_points=len(lines) - 1,
        optimized_xyz="smiles_methane.xyz",
    )
