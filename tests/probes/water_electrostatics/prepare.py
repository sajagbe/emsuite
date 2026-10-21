#!/usr/bin/env python3
"""Build q-scan and few-charge superposition jobs from the existing water map."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(
    "/data/PHO_WORK/sajagbe2/packages/emsuite/tests/integration_runs/"
    "recreate-maps-2026-10-07T035624Z/water"
)
MAP_CSV = SRC / "separate/singlet/results_Water_2026-10-07_04-18-19/Water_tuning_summary.csv"
Q_SCAN = (-0.50, -0.25, -0.10, -0.05, 0.05, 0.10, 0.25, 0.50)
PROPS = "['gse', 'exe', 'dm']"


def load_map(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for raw in csv.DictReader(f):
            r = {k.strip(): v.strip() for k, v in raw.items()}
            r["point_index"] = int(float(r["point_index"]))
            r["xyz"] = (float(r["x"]), float(r["y"]), float(r["z"]))
            r["s1"] = float(r["s1_exe_effect"])
            rows.append(r)
    if len(rows) != 32:
        raise SystemExit(f"expected 32 map rows, got {len(rows)} in {path}")
    return rows


def pick_sentinels(rows: list[dict]) -> dict[str, dict]:
    red = min(rows, key=lambda r: r["s1"])
    blue = max(rows, key=lambda r: r["s1"])
    used = {red["point_index"], blue["point_index"]}
    zero = min((r for r in rows if r["point_index"] not in used), key=lambda r: abs(r["s1"]))
    return {"RED": red, "BLUE": blue, "ZERO": zero}


def qtag(q: float) -> str:
    return f"q{q:+.2f}"


def write_surf(path: Path, points: list[tuple[tuple[float, float, float], float]]) -> None:
    lines = ["x          y          z          q"]
    for (x, y, z), q in points:
        lines.append(f"{x:<10.6f} {y:<10.6f} {z:<10.6f} {q:.6f}")
    path.write_text("\n".join(lines) + "\n")


def write_tuning(path: Path, calc_type: str) -> None:
    path.write_text(
        "\n".join(
            [
                "molecule = 'Water.xyz'",
                "surface_file = 'Water.surf'",
                "charge = 0",
                "spin = 0",
                "basis_set = '6-31G*'",
                "method = 'dft'",
                "functional = 'b3lyp'",
                "solvent = None",
                f"calc_type = '{calc_type}'",
                f"properties = {PROPS}",
                "state_of_interest = 1",
                "triplet = False",
                "parallel = False",
                "num_procs = 1",
                "",
            ]
        )
    )


def make_job(rel: str, points: list[tuple[tuple[float, float, float], float]], calc_type: str) -> Path:
    job = HERE / rel
    job.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "Water.xyz", job / "Water.xyz")
    write_surf(job / "Water.surf", points)
    write_tuning(job / "tuning.in", calc_type)
    return job


def main() -> None:
    sentinels = pick_sentinels(load_map(MAP_CSV))
    jobs: list[str] = []
    meta = {
        "source_map": str(MAP_CSV),
        "source_xyz": str(SRC / "Water.xyz"),
        "source_surf": str(SRC / "Water.surf"),
        "sentinels": {
            name: {
                "point_index": int(r["point_index"]),
                "xyz": list(r["xyz"]),
                "s1_exe_effect_q0.10": r["s1"],
            }
            for name, r in sentinels.items()
        },
    }

    for name, row in sentinels.items():
        for q in Q_SCAN:
            rel = f"jobs/qscan/{name}_{qtag(q)}"
            make_job(rel, [(row["xyz"], q)], "separate")
            jobs.append(rel)

    red, blue, zero = sentinels["RED"], sentinels["BLUE"], sentinels["ZERO"]
    combined = {
        "jobs/superpose/AB_q+0.10_combined": (
            [(red["xyz"], 0.10), (blue["xyz"], 0.10)],
        ),
        "jobs/superpose/ABC_q+0.10_combined": (
            [(red["xyz"], 0.10), (blue["xyz"], 0.10), (zero["xyz"], 0.10)],
        ),
        "jobs/superpose/AB_q+0.50_combined": (
            [(red["xyz"], 0.50), (blue["xyz"], 0.50)],
        ),
    }
    for rel, (pts,) in combined.items():
        make_job(rel, pts, "combined")
        jobs.append(rel)

    (HERE / "jobs.txt").write_text("\n".join(jobs) + "\n")
    (HERE / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(jobs)} jobs")
    for name, r in sentinels.items():
        print(f"  {name:4s} idx={r['point_index']:2d}  S1@0.10={r['s1']:+.4f} eV  xyz={r['xyz']}")
    print("next: sbatch run.slurm")


if __name__ == "__main__":
    main()
