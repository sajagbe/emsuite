#!/usr/bin/env python3
"""Standalone path existence checker for the lf-proto work directory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SYSTEMS = ("AT1", "AT2", "AS1", "AS2", "CR1", "CR2")


def check(path: Path, label: str) -> tuple[bool, str]:
    if path.exists():
        return True, f"OK   {label}: {path}"
    return False, f"MISS {label}: {path}"


def expected_paths(root: Path) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = [
        ("prep/LF.xyz", root / "prep" / "LF.xyz"),
        ("prep/surface.in", root / "prep" / "surface.in"),
        ("prep/LF.surf", root / "prep" / "LF.surf"),
        ("lf-homogeneous/singlet/LF.xyz", root / "lf-homogeneous" / "singlet" / "LF.xyz"),
        ("lf-homogeneous/singlet/LF.surf", root / "lf-homogeneous" / "singlet" / "LF.surf"),
        ("lf-homogeneous/singlet/tuning.in", root / "lf-homogeneous" / "singlet" / "tuning.in"),
        ("lf-homogeneous/triplet/LF.xyz", root / "lf-homogeneous" / "triplet" / "LF.xyz"),
        ("lf-homogeneous/triplet/LF.surf", root / "lf-homogeneous" / "triplet" / "LF.surf"),
        ("lf-homogeneous/triplet/tuning.in", root / "lf-homogeneous" / "triplet" / "tuning.in"),
    ]

    for state in ("singlet", "triplet"):
        paths.extend(
            [
                (f"lf-homogeneous/{state}/run.slurm", root / "lf-homogeneous" / state / "run.slurm"),
                (
                    f"lf-homogeneous/{state}/tuning_smoke.in",
                    root / "lf-homogeneous" / state / "tuning_smoke.in",
                ),
                (
                    f"lf-homogeneous/{state}/run_smoke.slurm",
                    root / "lf-homogeneous" / state / "run_smoke.slurm",
                ),
            ]
        )

    for sys in SYSTEMS:
        frame = root / "lov-protein" / "frames" / sys
        paths.extend(
            [
                (f"frames/{sys}/ligand.xyz", frame / "ligand.xyz"),
                (f"frames/{sys}/protein.xyz", frame / "protein.xyz"),
                (f"frames/{sys}/metadata.json", frame / "metadata.json"),
                (
                    f"potential/{sys}/potential.in",
                    root / "lov-protein" / "potential" / sys / "potential.in",
                ),
                (
                    f"potential/{sys}/run.slurm",
                    root / "lov-protein" / "potential" / sys / "run.slurm",
                ),
                (
                    f"coupled/{sys}/coupled.in",
                    root / "lov-protein" / "coupled" / sys / "coupled.in",
                ),
                (
                    f"coupled/{sys}/run.slurm",
                    root / "lov-protein" / "coupled" / sys / "run.slurm",
                ),
            ]
        )

    paths.extend(
        [
            ("coupled/AT1/coupled_smoke.in", root / "lov-protein" / "coupled" / "AT1" / "coupled_smoke.in"),
            ("coupled/AT1/run_smoke.slurm", root / "lov-protein" / "coupled" / "AT1" / "run_smoke.slurm"),
            ("submit_all.sh", root / "submit_all.sh"),
            ("validate.sh", root / "validate.sh"),
        ]
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(
            os.environ.get(
                "LF_PROTO_ROOT",
                "/data/PHO_WORK/sajagbe2/QMMM/LOVCalculations/lf-proto",
            )
        ),
        help="lf-proto work directory (default: $LF_PROTO_ROOT)",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        print(f"FAIL: work directory not found: {root}", file=sys.stderr)
        return 1

    print(f"Checking expected paths under {root}\n")
    passed = 0
    failed = 0
    for label, path in expected_paths(root):
        ok, line = check(path, label)
        print(line)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nSummary: {passed} found, {failed} missing, {passed + failed} checked")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
