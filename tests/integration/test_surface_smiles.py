"""Surface generation from SMILES integration test."""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite import SurfaceInput

from .helpers import METHANE_SURFACE_IN, record_assertions

SURFACE_IN = (
    METHANE_SURFACE_IN.replace("methane.surf", "smiles_surface.surf")
    .replace(
        "input_type = 'XYZ'\ninput_data = 'methane.xyz'",
        "input_type = 'SMILES'\ninput_data = 'C'",
    )
    .replace(
        "optimize = False",
        "optimize = True\noptimize_method = 'uff'\noptimized_xyz = 'smiles_methane.xyz'",
    )
)


@pytest.mark.slow
def test_surface_smiles_generates_surf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "surface.in").write_text(SURFACE_IN)
    result = SurfaceInput.from_file("surface.in").run()
    assert Path(result.path).is_file()
    lines = Path(result.path).read_text().strip().splitlines()
    assert len(lines) >= 11

    record_assertions(
        tmp_path,
        surf_path=result.path,
        surface_points=len(lines) - 1,
        optimized_xyz="smiles_methane.xyz",
        surface_type="homogenous",
        # SMILES→UFF geometry is nondeterministic; do not fingerprint coords/values.
    )
