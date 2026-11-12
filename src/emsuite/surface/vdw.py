"""VDW surface generation using the installed vdw-surfgen environment."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np


def get_vdw_surface_coordinates(xyz_file, surface_density=1.0, surface_scale=1.0):
    """Generate points without moving the input or touching caller scratch files.

    Use this Python interpreter and a unique scratch directory, independently
    of console-script PATH configuration. Only our own temporary files are removed.
    """
    for name, value in (("surface_density", surface_density), ("surface_scale", surface_scale)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    source = Path(xyz_file).resolve(strict=True)
    lines = source.read_text().splitlines()
    try:
        count = int(lines[0])
        rows = [line.split() for line in lines[2 : 2 + count]]
        coords = np.array([[float(x) for x in row[1:4]] for row in rows])
        if (
            count <= 0
            or len(rows) != count
            or coords.shape != (count, 3)
            or not np.isfinite(coords).all()
        ):
            raise ValueError("incomplete or nonfinite XYZ coordinates")
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid XYZ geometry in {source}: {exc}") from exc

    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="emsuite-surface-") as scratch:
        work = Path(scratch)
        copied = work / source.name
        shutil.copyfile(source, copied)
        start = perf_counter()
        ret = subprocess.run(
            [
                sys.executable,
                "-c",
                "from vdw_surfgen.cli import cli_entry; cli_entry()",
                str(copied),
                "-d",
                str(surface_density),
                "-s",
                str(surface_scale),
                "-t",
            ],
            cwd=work,
            capture_output=True,
            text=True,
        )
        generation_s = perf_counter() - start
        if ret.returncode != 0:
            raise RuntimeError(f"vdw-surfgen failed using {sys.executable}: {ret.stderr[-2000:]}")
        output = work / f"{source.stem}_vdw_surface.txt"
        if not output.is_file():
            raise FileNotFoundError(
                f"vdw-surfgen did not write {output.name} in its scratch directory"
            )
        start = perf_counter()
        if output.stat().st_size == 0:
            raise ValueError("vdw-surfgen produced an empty surface")
        points = np.loadtxt(output, dtype=float, ndmin=2)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError("vdw-surfgen surface must contain finite x, y, z coordinates")
        read_s = perf_counter() - start
    print(
        f"Surface timings (seconds): generation={generation_s:.6f}, "
        f"read={read_s:.6f}, total={perf_counter() - started:.6f}"
    )
    return points
