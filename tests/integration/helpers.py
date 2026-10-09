"""Shared helpers for integration tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"

METHANE_SURFACE_IN = """\
input_type = 'XYZ'
input_data = 'methane.xyz'
surface_density = 0.5
surface_scale = 1.0
surface_type = 'homogenous'
surface_charge = 0.1
output_surf = 'methane.surf'
optimize = False
"""

METHANE_SURFACE_HETERO_IN = """\
input_type = 'XYZ'
input_data = 'methane.xyz'
surface_density = 0.5
surface_scale = 1.0
surface_type = 'heterogenous'
output_surf = 'methane_hetero.surf'
optimize = False
"""

METHANE_XYZ = (FIXTURES / "methane.xyz").read_text()


def write_methane_xyz(tmp_path: Path) -> Path:
    xyz = tmp_path / "methane.xyz"
    xyz.write_text(METHANE_XYZ)
    return xyz


def install_methane_surf(tmp_path: Path, name: str = "methane.surf") -> Path:
    """Copy the committed VDW fixture (stable across runs; ``vsg`` itself is not)."""
    dest = tmp_path / name
    shutil.copy2(FIXTURES / "methane.surf", dest)
    return dest


def record_assertions(tmp_path: Path, **checks: object) -> None:
    """Persist assertion summary for the integration audit runner."""
    path = tmp_path / ".integration_assertions.json"
    path.write_text(json.dumps(checks, indent=2, default=str))


def latest_results_dir(tmp_path: Path, molecule: str = "methane") -> Path:
    dirs = sorted(tmp_path.glob(f"results_{molecule}_*"))
    assert dirs, f"no results_{molecule}_* directory under {tmp_path}"
    return dirs[-1]


def fingerprint_array(values: np.ndarray, *, decimals: int = 8) -> str:
    """Stable hash of rounded numeric values for cross-run consistency checks."""
    rounded = np.asarray(values, dtype=float).round(decimals)
    payload = rounded.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_surf(path: str | Path, *, decimals: int = 8) -> dict[str, object]:
    from emsuite.surface.io import load_surf

    coords, values = load_surf(str(path))
    return {
        "n_points": int(len(values)),
        "coords_sha256": fingerprint_array(coords, decimals=decimals),
        "values_sha256": fingerprint_array(values, decimals=decimals),
        "values_sum": float(np.round(np.sum(values), decimals)),
    }


def fingerprint_csv(path: str | Path, *, decimals: int = 8) -> dict[str, object]:
    text = Path(path).read_text().strip().splitlines()
    if not text:
        return {"n_rows": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    header, *rows = text
    numeric = []
    for row in rows:
        parts = row.split(",")
        for part in parts:
            try:
                numeric.append(float(part))
            except ValueError:
                continue
    return {
        "n_rows": len(rows),
        "header": header,
        "numeric_sha256": fingerprint_array(np.asarray(numeric, dtype=float), decimals=decimals),
    }
