"""Parse APBS OpenDX grid files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DxGrid:
    """Cell-centered (or APBS-native) scalar grid."""

    data: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.data.shape)


def parse_dx(path: str | Path) -> DxGrid:
    """Load an APBS OpenDX file into a ``(nx, ny, nz)`` array (x slowest, z fastest)."""
    counts: list[int] | None = None
    origin: list[float] | None = None
    deltas: list[list[float]] = []
    data = None

    # Read the ASCII header without an iterator, leaving a seekable stream for
    # NumPy's compiled numeric reader. Avoid one Python float per grid point.
    with Path(path).open(encoding="utf-8") as handle:
        while raw := handle.readline():
            line = raw.strip()
            if line.startswith("object 1 class gridpositions counts"):
                counts = [int(tok) for tok in line.split()[-3:]]
            elif line.startswith("origin"):
                origin = [float(tok) for tok in line.split()[1:4]]
            elif line.startswith("delta"):
                deltas.append([float(tok) for tok in line.split()[1:4]])
            elif "data follows" in line:
                if counts is None or any(n <= 0 for n in counts):
                    raise ValueError(f"DX file missing valid grid counts: {path}")
                data = np.fromfile(handle, dtype=float, count=int(np.prod(counts)), sep=" ")
                break

    if counts is None or origin is None or len(deltas) != 3:
        raise ValueError(f"DX file missing grid metadata: {path}")

    nx, ny, nz = counts
    expected = nx * ny * nz
    if data is None or data.size < expected:
        found = 0 if data is None else data.size
        raise ValueError(f"DX file {path} has {found} values, expected {expected}")

    data = data.reshape(nx, ny, nz)
    spacing = (deltas[0][0], deltas[1][1], deltas[2][2])
    return DxGrid(data=data, origin=(origin[0], origin[1], origin[2]), spacing=spacing)
