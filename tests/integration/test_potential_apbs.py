"""APBS runner smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emsuite.potential.apbs import run_apbs_grids

# whitespace-separated PQR: record serial name resname resseq x y z charge radius
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
    """run_apbs_grids(pqr_path=...) uses a ready PQR file and skips write_pqr."""
    monkeypatch.chdir(tmp_path)
    pqr = tmp_path / "external.pqr"
    pqr.write_text(EXTERNAL_PQR)

    box_coords = np.array(
        [[0.0, 0.0, 0.0], [1.089, 0.0, 0.0], [-0.363, 1.028, 0.0], [-0.363, -0.514, 0.891]]
    )
    grids = run_apbs_grids(pqr_path=pqr, box_coords=box_coords, workdir=tmp_path / "apbs_work")
    assert grids.potential.data.shape == (65, 65, 65)
    assert np.all(np.isfinite(grids.potential.data))
