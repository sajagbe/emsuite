"""APBS Poisson-Boltzmann grids (potential + dielectric)."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import apbs_binary
import numpy as np

from emsuite.geometry import read_xyz

from .dx import DxGrid, parse_dx
from .pqr import write_pqr


@dataclass(frozen=True)
class ApbsGrids:
    potential: DxGrid
    dielx: DxGrid | None = None
    diely: DxGrid | None = None
    dielz: DxGrid | None = None
    timings: dict[str, float] = field(default_factory=dict)


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
    include_dielectric: bool = True,
) -> Path:
    dielectric_output = ""
    if include_dielectric:
        dielectric_output = "".join(
            f"  write diel{axis} dx {prefix}_diel{axis}\n" for axis in "xyz"
        )
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
  bcfl mdh
  pdie {pdie}
  sdie {sdie}
  chgm spl2
  srfm smol
  srad 1.4
  swin 0.3
  sdens 10.0
  temp 298.15
  calcenergy total
  calcforce no
  write pot dx {prefix}
{dielectric_output}end
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


def dime_for_spacing(fine_length: float, spacing: float) -> int:
    """Smallest APBS multigrid size (32 k + 1) whose fine-grid spacing is at most ``spacing`` Å."""
    if not spacing > 0:
        raise ValueError("grid spacing must be positive")
    k = max(2, int(np.ceil(fine_length / spacing / 32.0)))
    return 32 * k + 1


def run_apbs_grids(
    xyz_path: str | None = None,
    charges: np.ndarray | None = None,
    pdie: float = 2.0,
    sdie: float = 78.54,
    dime: tuple[int, int, int] = (65, 65, 65),
    workdir: str | Path | None = None,
    atoms: list[tuple[str, float, float, float]] | None = None,
    box_coords: np.ndarray | None = None,
    pqr_path: str | Path | None = None,
    grid_spacing: float | None = None,
    include_dielectric: bool = True,
) -> ApbsGrids:
    """Run APBS; omit dielectric-map I/O when ``include_dielectric=False``.

    Timings separate execution (including APBS's file writing), each grid read,
    and temporary-file cleanup. The subprocess boundary cannot isolate the
    solver from its own writing. The low-level default retains all maps.

    ``pqr_path``, if given, points APBS at an already-written PQR (e.g. from
    pdb2pqr) directly — ``atoms``/``charges``/``write_pqr`` are skipped
    entirely. ``box_coords`` is then required (there's no atoms list here to
    derive the box extent from).
    """
    if pqr_path is not None:
        if box_coords is None:
            raise ValueError("run_apbs_grids requires box_coords when pqr_path is given")
        extent_coords = np.asarray(box_coords, dtype=float)
    else:
        if atoms is None:
            if xyz_path is None:
                raise ValueError("run_apbs_grids requires xyz_path or atoms")
            atoms = read_xyz(xyz_path)
        if charges is None:
            charges = np.zeros(len(atoms))
        atom_coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
        extent_coords = atom_coords if box_coords is None else np.asarray(box_coords, dtype=float)
    cglen, fglen = _box_lengths(extent_coords)
    if grid_spacing is not None:
        n = dime_for_spacing(fglen[0], grid_spacing)
        dime = (n, n, n)
        print(f"APBS grid: {n} points per side, fine spacing {fglen[0] / (n - 1):.3f} Å")

    if workdir is None:
        tmp = tempfile.TemporaryDirectory()
        work_path = Path(tmp.name)
    else:
        work_path = Path(workdir).resolve()
        work_path.mkdir(parents=True, exist_ok=True)
        tmp = None

    timings = {}
    try:
        if pqr_path is not None:
            input_pqr = Path(pqr_path).resolve()
        else:
            input_pqr = write_pqr(atoms, charges.tolist(), work_path / "input.pqr")
        prefix = "potential"
        apbs_in = _write_apbs_input(
            str(input_pqr),
            prefix,
            pdie,
            sdie,
            cglen,
            fglen,
            dime,
            work_path,
            include_dielectric=include_dielectric,
        )
        start = perf_counter()
        result = subprocess.run(
            [apbs_binary.APBS_BIN_PATH, str(apbs_in)],
            cwd=work_path,
            capture_output=True,
            text=True,
            check=False,
        )
        timings["apbs_execute_and_write_s"] = perf_counter() - start
        if result.returncode != 0:
            raise RuntimeError(f"APBS failed (exit {result.returncode}): {result.stderr[-500:]}")

        def read_grid(stem: str, name: str) -> DxGrid:
            start = perf_counter()
            grid = parse_dx(_dx_file(work_path, stem))
            timings[f"read_{name}_s"] = perf_counter() - start
            return grid

        potential = read_grid(prefix, "potential")
        dielectric = {}
        if include_dielectric:
            dielectric = {
                f"diel{axis}": read_grid(f"{prefix}_diel{axis}", f"diel{axis}") for axis in "xyz"
            }
        return ApbsGrids(potential=potential, timings=timings, **dielectric)
    finally:
        start = perf_counter()
        if tmp is not None:
            tmp.cleanup()
        timings["cleanup_s"] = perf_counter() - start
