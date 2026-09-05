"""Potential channel APBS smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emsuite.potential import run_potential_calculation
from emsuite.potential.apbs import run_apbs_grids
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


# pdb2pqr --whitespace format: record serial name resname resseq x y z charge radius
EXTERNAL_PQR = """\
ATOM 1 C MOL 1 0.000 0.000 0.000 -0.0776 1.7000
ATOM 2 H MOL 1 1.089 0.000 0.000 0.0194 1.2000
ATOM 3 H MOL 1 -0.363 1.028 0.000 0.0194 1.2000
ATOM 4 H MOL 1 -0.363 -0.514 0.891 0.0194 1.2000
ATOM 5 H MOL 1 -0.363 -0.514 -0.891 0.0194 1.2000
"""


@pytest.mark.slow
def test_run_apbs_grids_accepts_external_pqr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_apbs_grids(pqr_path=...) skips write_pqr entirely (Section C, pdb2pqr path)."""
    monkeypatch.chdir(tmp_path)
    pqr = tmp_path / "external.pqr"
    pqr.write_text(EXTERNAL_PQR)

    box_coords = np.array(
        [[0.0, 0.0, 0.0], [1.089, 0.0, 0.0], [-0.363, 1.028, 0.0], [-0.363, -0.514, 0.891]]
    )
    grids = run_apbs_grids(pqr_path=pqr, box_coords=box_coords, workdir=tmp_path / "apbs_work")
    assert grids.potential.data.shape == (65, 65, 65)
    assert np.all(np.isfinite(grids.potential.data))
