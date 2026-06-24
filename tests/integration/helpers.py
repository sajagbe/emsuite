"""Shared helpers for integration tests."""

from __future__ import annotations

import json
from pathlib import Path

METHANE_SURFACE_IN = """\
input_type = 'SMILES'
input_data = 'C'
surface_density = 0.5
surface_scale = 1.0
surface_type = 'homogenous'
surface_charge = 0.1
output_surf = 'methane.surf'
optimize = True
optimize_method = 'uff'
optimized_xyz = 'methane.xyz'
"""

METHANE_XYZ = """\
5
methane
C    0.000000    0.000000    0.000000
H    1.089000    0.000000    0.000000
H   -0.363000    1.028000    0.000000
H   -0.363000   -0.514000    0.891000
H   -0.363000   -0.514000   -0.891000
"""


def prepare_methane_surface(tmp_path: Path) -> tuple[Path, Path]:
    """Write surface.in and return (surf_in, expected_xyz)."""
    surf_in = tmp_path / "surface.in"
    surf_in.write_text(METHANE_SURFACE_IN)
    return surf_in, tmp_path / "methane.xyz"


def write_methane_xyz(tmp_path: Path) -> Path:
    xyz = tmp_path / "methane.xyz"
    xyz.write_text(METHANE_XYZ)
    return xyz


def tuning_in(
    properties: list[str],
    *,
    surface_file: str = "methane.surf",
    extra_lines: str = "",
) -> str:
    props_repr = repr(properties)
    return f"""\
molecule = 'methane.xyz'
surface_file = '{surface_file}'
properties = {props_repr}
basis_set = 'sto-3g'
method = 'dft'
functional = 'b3lyp'
charge = 0
spin = 0
solvent = None
calc_type = 'separate'
parallel = False
{extra_lines}"""


def record_assertions(tmp_path: Path, **checks: object) -> None:
    """Persist assertion summary for the integration audit runner."""
    path = tmp_path / ".integration_assertions.json"
    path.write_text(json.dumps(checks, indent=2, default=str))


def latest_results_dir(tmp_path: Path, molecule: str = "methane") -> Path:
    dirs = sorted(tmp_path.glob(f"results_{molecule}_*"))
    assert dirs, f"no results_{molecule}_* directory under {tmp_path}"
    return dirs[-1]
