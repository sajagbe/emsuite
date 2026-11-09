"""Build the fix-validation cases from reference (pre-drift) runs.

Each tuning case reruns a subset of surface points from a reference map with the
reference's own tuning.in, once on GPU and once on CPU, so the new results can be
compared point by point with the old ones (analyze.py).

Baseline cases rerun only the neutral/cation/anion baselines several times, to
check that the IE/EA baseline no longer drifts between identical runs.

Run from this folder:  python prepare.py   (writes cases/ and tasks_*.txt)
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IR = ROOT / "tests/integration_runs"
CAMP = IR / "opsins/campaigns/pose-restrain-2026-10-11"

# name, source dir (xyz/surf/tuning.in), oracle results dir, molecule tag, n points (None = all)
TUNING_CASES = [
    (
        "water_full",
        ROOT / "examples/water-gpu",
        ROOT / "examples/water-gpu/results_Water_2026-09-02_00-39-19",
        "Water",
        None,
    ),
    (
        "lf_gas",
        IR / "recreate-maps-2026-10-07T035624Z/lf-gas/6-31Gs/separate/singlet",
        None,
        "LF",
        8,
    ),
    ("rPSB_gas", CAMP / "gas/rPSB/separate_singlet", None, "rPSB", 6),
    ("SB_gas", CAMP / "gas/SB/separate_singlet", None, "SB", 6),
    ("rPSB_nhexane", CAMP / "solvents/rPSB/n-hexane/singlet", None, "rPSB", 6),
    ("rPSB_dmso", CAMP / "solvents/rPSB/dimethylsulfoxide/singlet", None, "rPSB", 6),
    ("SB_water", CAMP / "solvents/SB/water/singlet", None, "SB", 6),
]

# Baseline-only repeats: systems whose earlier IE/EA baselines drifted or are references.
BASELINE_CASES = [
    ("rPSB_gas", CAMP / "gas/rPSB/separate_singlet"),
    ("SB_gas", CAMP / "gas/SB/separate_singlet"),
    ("rPSB_nhexane", CAMP / "solvents/rPSB/n-hexane/singlet"),
    ("rPSB_chloroform", CAMP / "solvents/rPSB/chloroform/singlet"),
    ("rPSB_acetonitrile", CAMP / "solvents/rPSB/acetonitrile/singlet"),
    ("SB_dmso", CAMP / "solvents/SB/dimethylsulfoxide/singlet"),
]
REPEATS = {"gpu": 5, "cpu": 2}


def latest_results(d: Path) -> Path:
    rs = sorted(d.glob("results_*"), key=lambda p: p.stat().st_mtime)
    if not rs:
        raise FileNotFoundError(f"no results_* under {d}")
    return rs[-1]


def resolve_inputs(src: Path) -> tuple[Path, Path, str]:
    """Return absolute xyz, surf paths and tuning.in text for a source run dir."""
    text = (src / "tuning.in").read_text()

    def val(key: str) -> str:
        m = re.search(rf"^{key}\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if not m:
            raise KeyError(f"{key} not in {src}/tuning.in")
        return m.group(1)

    xyz = (src / val("molecule")).resolve()
    surf = (src / val("surface_file")).resolve()
    if not surf.exists():  # campaign runs copy geometry/<tag>.surf beside tuning.in
        surf = src / Path(val("surface_file")).name
    return xyz, surf, text


def set_key(text: str, key: str, value: str) -> str:
    new, n = re.subn(rf"^{key}\s*=.*$", f"{key} = {value}", text, flags=re.M)
    return new if n else new.rstrip("\n") + f"\n{key} = {value}\n"


def pick_points(rows: list[dict], n: int | None) -> list[int]:
    idx = list(range(len(rows)))
    if n is None or n >= len(rows):
        return idx

    def col(name: str) -> list[float] | None:
        if name not in rows[0]:
            return None
        try:
            return [float(r[name]) for r in rows]
        except ValueError:
            return None

    chosen: list[int] = []
    for name in ("s1_exe_effect", "gse_effect", "homo_effect"):
        v = col(name)
        if v:
            for i in (v.index(min(v)), v.index(max(v))):
                if i not in chosen:
                    chosen.append(i)
    step = len(rows) / max(1, n - len(chosen) + 1)
    k = 1
    while len(chosen) < n:
        i = int(round(k * step)) % len(rows)
        if i not in chosen:
            chosen.append(i)
        k += 1
    return sorted(chosen[:n])


def write_subset_surf(surf: Path, points: list[int], dest: Path, ref_rows: list[dict]) -> None:
    lines = surf.read_text().splitlines()
    header, body = lines[0], [ln for ln in lines[1:] if ln.strip()]
    out = [header]
    for i in points:
        x, y, z = (float(t) for t in body[i].split()[:3])
        r = ref_rows[i]
        if max(abs(x - float(r["x"])), abs(y - float(r["y"])), abs(z - float(r["z"]))) > 1e-4:
            raise SystemExit(f"{surf}: point {i} does not match reference CSV")
        out.append(body[i])
    dest.write_text("\n".join(out) + "\n")


def main() -> None:
    cases = HERE / "cases"
    if cases.exists():
        raise SystemExit(f"{cases} exists; move it aside before re-preparing")
    tasks = {"gpu4": [], "gpu1": [], "cpu": []}
    manifest = {"tuning": {}, "baseline": {}}

    for name, src, oracle, tag, npts in TUNING_CASES:
        oracle = oracle or latest_results(src)
        ref_csv = next(oracle.glob("*_tuning_summary.csv"))
        rows = list(csv.DictReader(ref_csv.open()))
        points = pick_points(rows, npts)
        xyz, surf, text = resolve_inputs(src)
        for dev in ("gpu", "cpu"):
            d = cases / name / dev
            d.mkdir(parents=True)
            shutil.copy2(xyz, d / f"{tag}.xyz")
            write_subset_surf(surf, points, d / f"{tag}.surf", rows)
            t = set_key(text, "molecule", f"'{tag}.xyz'")
            t = set_key(t, "surface_file", f"'{tag}.surf'")
            if dev == "gpu":
                big = tag in ("rPSB", "SB")
                t = set_key(t, "parallel", "True")
                t = set_key(t, "num_procs", "4" if big else "1")
                tasks["gpu4" if big else "gpu1"].append(f"tune {d.relative_to(HERE)}")
            else:
                t = set_key(t, "parallel", "False")
                t = set_key(t, "num_procs", "1")
                tasks["cpu"].append(f"tune {d.relative_to(HERE)}")
            (d / "tuning.in").write_text(t)
        manifest["tuning"][name] = {
            "reference_csv": str(ref_csv),
            "reference_points": points,
            "source": str(src),
        }

    for name, src in BASELINE_CASES:
        xyz, _surf, text = resolve_inputs(src)
        tag = xyz.stem
        ref_csv = next(latest_results(src).glob("*_tuning_summary.csv"))
        for dev in ("gpu", "cpu"):
            d = cases / "baseline" / name / dev
            d.mkdir(parents=True)
            shutil.copy2(xyz, d / f"{tag}.xyz")
            (d / "tuning.in").write_text(set_key(text, "molecule", f"'{tag}.xyz'"))
            tasks["gpu1" if dev == "gpu" else "cpu"].append(
                f"baseline {d.relative_to(HERE)} {REPEATS[dev]}"
            )
        manifest["baseline"][name] = {"reference_csv": str(ref_csv), "source": str(src)}

    tasks["cpu"].append("pytest tests/integration/test_failed_point_recovery.py")
    for k, v in tasks.items():
        (HERE / f"tasks_{k}.txt").write_text("\n".join(v) + "\n")
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print({k: len(v) for k, v in tasks.items()})


if __name__ == "__main__":
    main()
