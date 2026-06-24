"""v1.2 potential bond-axis scan integration test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emsuite.potential import run_potential_calculation

from .helpers import record_assertions, write_methane_xyz

POTENTIAL_BOND_IN = """\
molecule = 'methane.xyz'
output_surf = 'bond_scan.surf'
method = 'coulomb'
bond_scan_atoms = [0, 1]
bond_scan_steps = 7
bond_scan_span = 2.0
"""


@pytest.mark.slow
def test_potential_bond_axis_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_methane_xyz(tmp_path)
    (tmp_path / "potential.in").write_text(POTENTIAL_BOND_IN)

    surf_path = run_potential_calculation("potential.in")
    assert Path(surf_path).is_file()

    data = np.loadtxt(surf_path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape[0] == 7
    assert np.all(np.isfinite(data[:, 3]))

    csv_path = Path("bond_scan.csv")
    assert csv_path.is_file()

    record_assertions(
        tmp_path,
        bond_scan_points=7,
        potentials_finite=True,
        output_surf=str(surf_path),
        csv_rows=7,
    )
