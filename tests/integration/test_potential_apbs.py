"""Potential channel APBS smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emsuite.potential import run_potential_calculation
from emsuite.surface import run_surface_calculation

from .helpers import METHANE_SURFACE_IN, record_assertions

SURFACE_IN = METHANE_SURFACE_IN

POTENTIAL_IN = """\
molecule = 'methane.xyz'
surface_file = 'methane.surf'
output_surf = 'methane_potential.surf'
method = 'apbs'
quantity = 'potential'
"""


@pytest.mark.slow
def test_potential_apbs_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "surface.in").write_text(SURFACE_IN)
    run_surface_calculation("surface.in")
    (tmp_path / "potential.in").write_text(POTENTIAL_IN)
    surf_path = run_potential_calculation("potential.in")
    assert Path(surf_path).is_file()
    data = np.loadtxt(surf_path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    potentials = data[:, 3]
    assert potentials.shape[0] >= 10
    assert np.all(np.isfinite(potentials))

    record_assertions(
        tmp_path,
        method="apbs",
        quantity="potential",
        surface_points=int(potentials.shape[0]),
        potentials_finite=True,
        output_surf=surf_path,
    )
