"""APBS Poisson-Boltzmann potential sampling."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import apbs_binary
import numpy as np

from .pqr import read_xyz, write_pqr


def _write_apbs_input(
    pqr_name: str,
    output_prefix: str,
    pdie: float,
    sdie: float,
    path: Path,
) -> Path:
    content = f"""read
  mol pqr {pqr_name}
end
elec
  mg-auto
  dime 33 33 33
  cglen 16 16 16
  fglen 10 10 10
  cgcent mol 1
  fgcent mol 1
  mol 1
  lpbe
  pdie {pdie}
  sdie {sdie}
  chgm spl2
  srfm smol
  srad 1.4
  swin 0.3
  temp 298.15
  calcenergy total
  calcforce no
  write pot dx {output_prefix}
end
quit
"""
    apbs_in = path / "apbs.in"
    apbs_in.write_text(content)
    return apbs_in


def run_apbs_potential(
    xyz_path: str,
    surface_coords: np.ndarray,
    charges: np.ndarray | None = None,
    pdie: float = 2.0,
    sdie: float = 78.54,
    workdir: str | Path | None = None,
) -> np.ndarray:
    """
    Run APBS and sample electrostatic potential at surface coordinates.

    Raises RuntimeError when APBS fails; callers may fall back to Coulomb ESP.
    """
    atoms = read_xyz(xyz_path)
    if charges is None:
        charges = np.zeros(len(atoms))

    if workdir is None:
        tmp = tempfile.TemporaryDirectory()
        work_path = Path(tmp.name)
    else:
        work_path = Path(workdir)
        tmp = None

    try:
        pqr_path = write_pqr(atoms, charges.tolist(), work_path / "input.pqr")
        prefix = "potential"
        apbs_in = _write_apbs_input(pqr_path.name, prefix, pdie, sdie, work_path)

        result = subprocess.run(
            [apbs_binary.APBS_BIN_PATH, str(apbs_in)],
            cwd=work_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"APBS failed (exit {result.returncode}): {result.stderr[-500:]}")

        dx_files = list(work_path.glob(f"{prefix}*.dx"))
        if not dx_files:
            raise RuntimeError("APBS did not produce a DX potential file")

        coords_file = work_path / "surface_coords.txt"
        np.savetxt(coords_file, surface_coords, fmt="%.6f")

        multivalue_out = work_path / "multivalue_out.txt"
        subprocess.run(
            [
                apbs_binary.MULTIVALUE_BIN_PATH,
                str(coords_file),
                str(dx_files[0]),
                str(multivalue_out),
            ],
            cwd=work_path,
            capture_output=True,
            text=True,
            check=True,
        )

        values = np.loadtxt(multivalue_out)
        if values.ndim == 0:
            return np.array([float(values)])
        if values.ndim == 2:
            return values[:, -1]
        return values
    finally:
        if tmp is not None:
            tmp.cleanup()
