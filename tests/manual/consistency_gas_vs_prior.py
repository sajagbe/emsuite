#!/usr/bin/env python3
"""Gas-phase water combined exe/osc smoke vs prior successful gas results.

Compares a fresh TuningInput(combined, exe+osc) fingerprint against the
archived gas-phase water combined singlet summary under
tests/integration_runs/recreate-maps-.../water/combined/singlet/.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import importlib.util

from emsuite.inputs import TuningInput

_helpers_path = _ROOT / "tests" / "integration" / "helpers.py"
_spec = importlib.util.spec_from_file_location("emsuite_int_helpers", _helpers_path)
assert _spec and _spec.loader
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
fingerprint_csv = _helpers.fingerprint_csv
latest_results_dir = _helpers.latest_results_dir


_PRIOR = (
    _ROOT
    / "tests/integration_runs/recreate-maps-2026-10-07T035624Z/water/combined/singlet"
)
_WATER_XYZ = _ROOT / "tests/integration_runs/recreate-maps-2026-10-07T035624Z/water/Water.xyz"
_WATER_SURF = _ROOT / "tests/integration_runs/recreate-maps-2026-10-07T035624Z/water/Water.surf"


def _find_prior_summary() -> Path:
    matches = sorted(_PRIOR.glob("results_Water_*/Water_tuning_summary.csv"))
    # Prefer the successful one with property columns.
    for path in reversed(matches):
        header = path.read_text().splitlines()[0]
        if "s1_exe" in header and "s1_osc" in header:
            return path
    raise FileNotFoundError(f"no prior exe/osc summary under {_PRIOR}")


def main() -> int:
    if not _WATER_XYZ.is_file() or not _WATER_SURF.is_file():
        print("SKIP: prior water fixtures missing")
        return 0

    prior = _find_prior_summary()
    prior_fp = fingerprint_csv(prior, decimals=5)
    print("prior:", prior)
    print("prior fingerprint:", prior_fp)

    with tempfile.TemporaryDirectory(prefix="emsuite_consist_") as tmp:
        tmp_path = Path(tmp)
        os.chdir(tmp_path)
        xyz = tmp_path / "Water.xyz"
        surf = tmp_path / "Water.surf"
        xyz.write_bytes(_WATER_XYZ.read_bytes())
        surf.write_bytes(_WATER_SURF.read_bytes())

        # Keep CUDA to one device for combined stability.
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[
            0
        ]

        TuningInput(
            molecule=str(xyz),
            surface_file=str(surf),
            properties=("exe", "osc"),
            basis_set="6-31G*",
            method="dft",
            functional="b3lyp",
            solvent=None,
            calc_type="combined",
            parallel=False,
            state_of_interest=3,
            triplet=False,
        ).run()

        results_dir = latest_results_dir(tmp_path, molecule="Water")
        summary = results_dir / "Water_tuning_summary.csv"
        assert summary.is_file()
        fresh_fp = fingerprint_csv(summary, decimals=5)
        print("fresh fingerprint:", fresh_fp)

        # Compare baselines / effects for s1–s3 exe+osc (header-aligned numeric hash).
        if fresh_fp["numeric_sha256"] == prior_fp["numeric_sha256"]:
            print("PASS: exact fingerprint match at 5 decimals")
            return 0

        # Soft compare: load shared columns and check max relative drift.
        import csv

        def _row_map(path: Path) -> dict[str, float]:
            with path.open() as fh:
                reader = csv.DictReader(fh)
                row = next(reader)
            return {k: float(v) for k, v in row.items() if k not in {"point_index", "x", "y", "z"}}

        a, b = _row_map(prior), _row_map(summary)
        keys = sorted(set(a) & set(b))
        rels = []
        for k in keys:
            denom = max(abs(a[k]), 1e-12)
            rels.append(abs(a[k] - b[k]) / denom)
        max_rel = float(max(rels)) if rels else float("inf")
        print(f"shared columns={len(keys)} max_rel={max_rel:.3e}")
        if max_rel < 1e-4:
            print("PASS: within 1e-4 relative of prior gas combined")
            return 0
        print("FAIL: drifted beyond tolerance vs prior")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
