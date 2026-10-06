#!/usr/bin/env python3
"""Run the EMSuite consistency suite and optionally promote a baseline.

Usage (from packages/emsuite)::

    # Capture / refresh the oracle used across cleanup steps
    python scripts/run_consistency_suite.py --baseline

    # Re-check after a cleanup step (archives under tests/integration_runs/)
    python scripts/run_consistency_suite.py

    # Diff fingerprints in assertions.json between two archived runs
    python scripts/run_consistency_suite.py --compare \\
        tests/integration_runs/baseline \\
        tests/integration_runs/<run-id>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "tests" / "integration_runs"
BASELINE = RUNS / "baseline"
BASELINE_FINGERPRINTS = RUNS / "baseline_fingerprints.json"

# Core slow cases + consistency-matrix gaps (CPU). GPU stays optional/separate.
CONSISTENCY_NODEIDS = [
    "tests/integration/test_surface_smiles.py::test_surface_smiles_generates_surf",
    "tests/integration/test_smoke_methane.py::test_methane_surface_to_tuning_smoke",
    "tests/integration/test_potential_apbs.py::test_potential_apbs_map",
    "tests/integration/test_consistency_matrix.py::test_surface_heterogenous_generates_edit_header",
    "tests/integration/test_consistency_matrix.py::test_potential_apbs_gauss_charge",
    "tests/integration/test_consistency_matrix.py::test_tuning_combined_methane",
    "tests/integration/test_coupled_smoke.py::test_coupled_pipeline",
    "tests/integration/test_coupled_potential_surf.py::test_coupled_reuses_potential_surf_across_calc_types",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _collect_fingerprints(run_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for assertions in sorted(run_dir.glob("*/assertions.json")):
        key = assertions.parent.name
        out[key] = json.loads(assertions.read_text())
    return out


def _compare_fingerprints(base: dict[str, object], cand: dict[str, object]) -> int:
    if not base:
        print("Baseline fingerprints empty.", file=sys.stderr)
        return 2
    if not cand:
        print("Candidate fingerprints empty.", file=sys.stderr)
        return 2

    missing = sorted(set(base) - set(cand))
    extra = sorted(set(cand) - set(base))
    drifts: list[str] = []

    for key in sorted(set(base) & set(cand)):
        b, c = base[key], cand[key]
        if not isinstance(b, dict) or not isinstance(c, dict):
            if b != c:
                drifts.append(f"{key}: baseline={b!r} candidate={c!r}")
            continue
        for field in ("surf", "summary", "quantity", "calc_type", "surface_type", "properties"):
            if field in b and field in c and b[field] != c[field]:
                drifts.append(f"{key}.{field}: baseline={b[field]!r} candidate={c[field]!r}")

    if missing:
        print("Missing cases vs baseline:", ", ".join(missing))
    if extra:
        print("Extra cases vs baseline:", ", ".join(extra))
    if drifts:
        print("Fingerprint drifts:")
        for line in drifts:
            print(" ", line)

    if missing or drifts:
        return 1
    print(f"OK: fingerprints match baseline ({len(base)} cases).")
    return 0


def _compare(baseline_dir: Path, candidate_dir: Path) -> int:
    return _compare_fingerprints(_collect_fingerprints(baseline_dir), _collect_fingerprints(candidate_dir))

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=f"After a green run, copy artifacts to {BASELINE}",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_DIR", "CANDIDATE_DIR"),
        help="Compare assertions.json fingerprints only (no pytest).",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args after -- passed to pytest",
    )
    args = parser.parse_args(argv)

    if args.compare:
        return _compare(Path(args.compare[0]), Path(args.compare[1]))

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    run_dir = RUNS / f"consistency-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["EMSUITE_INTEGRATION_RUN_DIR"] = str(run_dir)

    if shutil.which("uv"):
        cmd = [
            "uv",
            "run",
            "pytest",
            "-v",
            "-m",
            "slow",
            "--tb=short",
            f"--junitxml={run_dir / 'junit.xml'}",
            *CONSISTENCY_NODEIDS,
            *args.pytest_args,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-m",
            "slow",
            "--tb=short",
            f"--junitxml={run_dir / 'junit.xml'}",
            *CONSISTENCY_NODEIDS,
            *args.pytest_args,
        ]
    meta = {
        "timestamp_utc": stamp,
        "git_sha": _git_sha(),
        "command": cmd,
        "nodeids": CONSISTENCY_NODEIDS,
        "promote_baseline": bool(args.baseline),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("Running:", " ".join(cmd))
    print("Artifacts:", run_dir)
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    (run_dir / "pytest_exit_code.txt").write_text(str(proc.returncode))

    fingerprints = _collect_fingerprints(run_dir)
    (run_dir / "fingerprints.json").write_text(json.dumps(fingerprints, indent=2))

    if proc.returncode != 0:
        print(f"Suite failed (exit {proc.returncode}); baseline not updated.", file=sys.stderr)
        return proc.returncode

    if args.baseline:
        if BASELINE.exists():
            shutil.rmtree(BASELINE)
        shutil.copytree(run_dir, BASELINE)
        BASELINE_FINGERPRINTS.write_text(json.dumps(fingerprints, indent=2, sort_keys=True) + "\n")
        print(f"Baseline updated → {BASELINE}")
        print(f"Committed oracle → {BASELINE_FINGERPRINTS}")
    elif BASELINE_FINGERPRINTS.is_file():
        print(f"Comparing against {BASELINE_FINGERPRINTS}…")
        base = json.loads(BASELINE_FINGERPRINTS.read_text())
        return _compare_fingerprints(base, fingerprints)
    elif BASELINE.exists():
        print("Comparing against existing baseline directory…")
        return _compare(BASELINE, run_dir)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
