"""APBS Poisson-Boltzmann grids (potential + dielectric)."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import apbs_binary
import numpy as np

from emsuite.geometry import read_xyz

from .dx import DxGrid, parse_dx
from .pqr import write_pqr


@dataclass(frozen=True)
class ApbsGrids:
    potential: DxGrid
    dielx: DxGrid
    diely: DxGrid
    dielz: DxGrid


def _box_lengths(
    atom_coords: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    extent = atom_coords.max(axis=0) - atom_coords.min(axis=0)
    box = float(max(float(np.max(extent)) + 12.0, 16.0))
    cglen = (box, box, box)
    fglen = (0.8 * box, 0.8 * box, 0.8 * box)
    return cglen, fglen


def _write_apbs_input(
    pqr_name: str,
    prefix: str,
    pdie: float,
    sdie: float,
    cglen: tuple[float, float, float],
    fglen: tuple[float, float, float],
    dime: tuple[int, int, int],
    path: Path,
) -> Path:
    content = f"""read
  mol pqr {pqr_name}
end
elec
  mg-auto
  dime {dime[0]} {dime[1]} {dime[2]}
  cglen {cglen[0]:.3f} {cglen[1]:.3f} {cglen[2]:.3f}
  fglen {fglen[0]:.3f} {fglen[1]:.3f} {fglen[2]:.3f}
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
  write pot dx {prefix}
  write dielx dx {prefix}_dielx
  write diely dx {prefix}_diely
  write dielz dx {prefix}_dielz
end
quit
"""
    apbs_in = path / "apbs.in"
    apbs_in.write_text(content)
    return apbs_in


def _dx_file(work_path: Path, stem: str) -> Path:
    exact = work_path / f"{stem}.dx"
    if exact.is_file():
        return exact
    if stem.endswith(("_dielx", "_diely", "_dielz")):
        matches = list(work_path.glob(f"{stem}*.dx"))
    else:
        matches = [path for path in work_path.glob(f"{stem}*.dx") if "_diel" not in path.name]
    if not matches:
        raise RuntimeError(f"APBS did not produce a DX file named '{stem}'")
    return matches[0]


def run_apbs_grids(
    xyz_path: str | None = None,
    charges: np.ndarray | None = None,
    pdie: float = 2.0,
    sdie: float = 78.54,
    dime: tuple[int, int, int] = (65, 65, 65),
    workdir: str | Path | None = None,
    atoms: list[tuple[str, float, float, float]] | None = None,
    box_coords: np.ndarray | None = None,
) -> ApbsGrids:
    """Run APBS and return potential plus dielectric grids."""
    if atoms is None:
        if xyz_path is None:
            raise ValueError("run_apbs_grids requires xyz_path or atoms")
        atoms = read_xyz(xyz_path)
    if charges is None:
        charges = np.zeros(len(atoms))
    atom_coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
    extent_coords = atom_coords if box_coords is None else np.asarray(box_coords, dtype=float)
    cglen, fglen = _box_lengths(extent_coords)

    if workdir is None:
        tmp = tempfile.TemporaryDirectory()
        work_path = Path(tmp.name)
    else:
        work_path = Path(workdir)
        tmp = None

    try:
        pqr_path = write_pqr(atoms, charges.tolist(), work_path / "input.pqr")
        prefix = "potential"
        apbs_in = _write_apbs_input(
            pqr_path.name, prefix, pdie, sdie, cglen, fglen, dime, work_path
        )
        result = subprocess.run(
            [apbs_binary.APBS_BIN_PATH, str(apbs_in)],
            cwd=work_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"APBS failed (exit {result.returncode}): {result.stderr[-500:]}")

        return ApbsGrids(
            potential=parse_dx(_dx_file(work_path, prefix)),
            dielx=parse_dx(_dx_file(work_path, f"{prefix}_dielx")),
            diely=parse_dx(_dx_file(work_path, f"{prefix}_diely")),
            dielz=parse_dx(_dx_file(work_path, f"{prefix}_dielz")),
        )
    finally:
        if tmp is not None:
            tmp.cleanup()
