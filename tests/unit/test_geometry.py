"""Geometry XYZ round-trip."""

from pathlib import Path

import numpy as np

from emsuite.geometry import Geometry

XYZ = """\
2
test
H  0.0000000000  0.0000000000  0.0000000000
H  0.7400000000  0.0000000000  0.0000000000
"""


def test_geometry_xyz_roundtrip(tmp_path: Path):
    src = tmp_path / "h2.xyz"
    src.write_text(XYZ)
    geom = Geometry.from_xyz(src)
    assert geom.symbols == ("H", "H")
    assert geom.coords.shape == (2, 3)
    out = tmp_path / "out.xyz"
    geom.to_xyz(out)
    again = Geometry.from_xyz(out)
    assert again.symbols == geom.symbols
    np.testing.assert_allclose(again.coords, geom.coords)
