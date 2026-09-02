#!/usr/bin/env python3
"""Validate all .in files under the lf-proto work directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emsuite.config import parse_assignments
from emsuite.inputs import CoupledInput, PotentialInput, SurfaceInput, TuningInput

PATH_ATTRS: dict[type, tuple[str, ...]] = {
    SurfaceInput: ("input_data", "optimized_xyz"),
    TuningInput: ("molecule", "surface_file"),
    PotentialInput: ("molecule", "surface_file", "protein", "ligand", "ligand_mol2"),
    CoupledInput: (
        "molecule",
        "surface_file",
        "protein",
        "ligand",
        "ligand_mol2",
        "potential_surf",
    ),
}


def guess_input_class(path: Path) -> type:
    name = path.name.lower()
    if "surface" in name and "potential" not in name and "coupled" not in name:
        return SurfaceInput
    if "coupled" in name:
        return CoupledInput
    if "potential" in name:
        return PotentialInput
    if "tuning" in name:
        return TuningInput

    keys = set(parse_assignments(path.read_text()).keys())
    if "input_type" in keys:
        return SurfaceInput
    if "potential_quantity" in keys or "potential_method" in keys:
        return CoupledInput
    if "quantity" in keys and "properties" not in keys:
        return PotentialInput
    if "surface_file" in keys and "properties" in keys:
        return TuningInput
    raise ValueError(f"cannot determine input type for {path}")


def resolve_path_fields(obj: object, job_dir: Path) -> list[tuple[str, Path]]:
    cls = type(obj)
    found: list[tuple[str, Path]] = []
    for attr in PATH_ATTRS.get(cls, ()):
        value = getattr(obj, attr, None)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = (job_dir / path).resolve()
        found.append((attr, path))
    return found


def validate_file(path: Path) -> tuple[bool, list[str]]:
    messages: list[str] = []
    try:
        cls = guess_input_class(path)
        obj = cls.from_file(path)
        messages.append(f"  parsed as {cls.__name__}")
    except Exception as exc:
        return False, [f"  parse error: {exc}"]

    missing: list[str] = []
    for attr, resolved in resolve_path_fields(obj, path.parent):
        if resolved.exists():
            messages.append(f"  {attr}: OK ({resolved})")
        else:
            missing.append(f"  {attr}: MISSING ({resolved})")

    if missing:
        messages.extend(missing)
        return False, messages
    return True, messages


def find_in_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.in"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(
            __import__("os").environ.get(
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

    in_files = find_in_files(root)
    if not in_files:
        print(f"FAIL: no .in files under {root}", file=sys.stderr)
        return 1

    passed = 0
    failed = 0
    print(f"Validating {len(in_files)} input file(s) under {root}\n")

    for path in in_files:
        rel = path.relative_to(root)
        ok, messages = validate_file(path)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {rel}")
        for line in messages:
            print(line)
        print()
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"Summary: {passed} passed, {failed} failed, {len(in_files)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
