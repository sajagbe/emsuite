"""Exercise requested APBS outputs without executing the scientific solver."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from emsuite.potential import apbs


@pytest.mark.parametrize("dielectric", [False, True])
def test_requested_grids_and_timings(tmp_path, monkeypatch, dielectric):
    pqr = tmp_path / "input.pqr"
    pqr.write_text("unused by fake solver")
    header = """object 1 class gridpositions counts 2 2 2
origin 0 0 0
delta 1 0 0
delta 0 1 0
delta 0 0 1
object 3 class array type double rank 0 items 8 data follows
1 2 3 4 5 6 7 8
"""

    def solver(args, cwd, **kwargs):
        text = Path(args[1]).read_text()
        assert "write pot dx potential" in text
        for axis in "xyz":
            assert (f"write diel{axis}" in text) == dielectric
        for name in ["potential"] + ([f"potential_diel{x}" for x in "xyz"] if dielectric else []):
            (cwd / f"{name}.dx").write_text(header)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(apbs.subprocess, "run", solver)
    grids = apbs.run_apbs_grids(
        pqr_path=pqr,
        box_coords=np.zeros((1, 3)),
        workdir=tmp_path / "work",
        include_dielectric=dielectric,
    )
    assert grids.potential.data.shape == (2, 2, 2)
    assert (grids.dielx is not None) == dielectric
    assert set(grids.timings) == {"apbs_execute_and_write_s", "read_potential_s", "cleanup_s"} | (
        {f"read_diel{x}_s" for x in "xyz"} if dielectric else set()
    )
    assert all(value >= 0 for value in grids.timings.values())


@pytest.mark.parametrize("quantity", ["potential", "charge"])
@pytest.mark.parametrize("pqr", [None, "saved.pqr"])
def test_channel_requests_maps_for_quantity(monkeypatch, quantity, pqr):
    from emsuite.potential import runner
    from emsuite.potential.dx import DxGrid

    grid = DxGrid(np.ones((3, 3, 3)), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def grids(**kwargs):
        assert kwargs["include_dielectric"] == (quantity == "charge")
        assert kwargs["grid_spacing"] == 0.3
        return apbs.ApbsGrids(grid, grid, grid, grid)

    monkeypatch.setattr(runner, "run_apbs_grids", grids)
    values = runner._apbs_values(
        np.array([[1.0, 1.0, 1.0]]),
        [],
        [],
        np.zeros((1, 3)),
        2.0,
        78.54,
        quantity,
        pqr_path=pqr,
        grid_spacing=0.3,
    )
    np.testing.assert_allclose(values, [1.0 if quantity == "potential" else 0.0])
