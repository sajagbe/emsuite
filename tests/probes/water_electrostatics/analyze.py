#!/usr/bin/env python3
"""Full-shell preview now; q-scan and few-charge plots when jobs finish."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SRC = Path(
    "/data/PHO_WORK/sajagbe2/packages/emsuite/tests/integration_runs/"
    "recreate-maps-2026-10-07T035624Z/water"
)
SEP_CSV = SRC / "separate/singlet/results_Water_2026-10-07_04-18-19/Water_tuning_summary.csv"
CMB_CSV = SRC / "combined/singlet/results_Water_2026-10-07_21-21-02/Water_tuning_summary.csv"
PROPS = ("gse_effect", "s1_exe_effect", "dm_effect")


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for raw in csv.DictReader(f):
            rows.append({k.strip(): v.strip() for k, v in raw.items()})
    return rows


def latest_summary(job: Path) -> Path | None:
    found = sorted(job.glob("results_Water_*/Water_tuning_summary.csv"))
    return found[-1] if found else None


def effect(rows: list[dict], col: str) -> list[float]:
    return [float(r[col]) for r in rows]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def full_shell() -> None:
    sep, cmb = load_csv(SEP_CSV), load_csv(CMB_CSV)
    rows = []
    for col in [k for k in sep[0] if k.endswith("_effect") and "normalized" not in k]:
        if col not in cmb[0]:
            continue
        s = float(np.sum(effect(sep, col)))
        c = float(effect(cmb, col)[0])
        rows.append(
            {
                "property": col,
                "n_separate": len(sep),
                "sum_separate": f"{s:.8g}",
                "combined": f"{c:.8g}",
                "abs_diff": f"{abs(s - c):.8g}",
                "ratio": f"{(s / c) if c else 'nan'}",
            }
        )
    write_csv(
        OUT / "full_shell.csv",
        rows,
        ["property", "n_separate", "sum_separate", "combined", "abs_diff", "ratio"],
    )
    print(f"full-shell: {len(sep)} separate vs 1 combined -> {OUT / 'full_shell.csv'}")
    for r in rows:
        if r["property"] in PROPS:
            print(
                f"  {r['property']:16s} sum={r['sum_separate']:>12}  "
                f"cmb={r['combined']:>12}  ratio={r['ratio']}"
            )


def qscan_and_sign(manifest: dict) -> None:
    rows = []
    for name in ("RED", "BLUE", "ZERO"):
        for qdir in sorted((HERE / "jobs/qscan").glob(f"{name}_q*")):
            summary = latest_summary(qdir)
            if summary is None:
                continue
            rec = load_csv(summary)[0]
            q = float(qdir.name.split("_q")[1])
            rows.append(
                {
                    "label": name,
                    "q": q,
                    "point_index": manifest["sentinels"][name]["point_index"],
                    "gse_effect": float(rec["gse_effect"]),
                    "s1_exe_effect": float(rec["s1_exe_effect"]),
                    "dm_effect": float(rec["dm_effect"]),
                    "results": str(summary),
                }
            )
    if not rows:
        print("q-scan: no finished jobs yet")
        return
    write_csv(
        OUT / "qscan.csv",
        [{k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
        ["label", "q", "point_index", "gse_effect", "s1_exe_effect", "dm_effect", "results"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for name in ("RED", "BLUE", "ZERO"):
        pts = sorted((r for r in rows if r["label"] == name), key=lambda r: r["q"])
        if not pts:
            continue
        qs = [r["q"] for r in pts]
        ys = [r["s1_exe_effect"] for r in pts]
        axes[0].plot(qs, ys, "o-", label=name)
    axes[0].axhline(0.0, color="0.5", lw=0.6)
    axes[0].axvline(0.0, color="0.5", lw=0.6)
    axes[0].set_xlabel("q / e")
    axes[0].set_ylabel("S1 ΔP / eV")
    axes[0].set_title("q-scan")
    axes[0].legend(frameon=False)

    for name in ("RED", "BLUE", "ZERO"):
        byq = {r["q"]: r["s1_exe_effect"] for r in rows if r["label"] == name}
        xs, ys = [], []
        for q, yp in byq.items():
            if q > 0 and -q in byq:
                xs.append(yp)
                ys.append(-byq[-q])
        if xs:
            axes[1].plot(xs, ys, "o", label=name)
    lims = axes[1].get_xlim() + axes[1].get_ylim()
    lo, hi = min(lims), max(lims)
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[1].set_xlabel("ΔP(+q) / eV")
    axes[1].set_ylabel("−ΔP(−q) / eV")
    axes[1].set_title("sign reversal")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "qscan.png", dpi=150)
    plt.close(fig)
    print(f"q-scan: {len(rows)} rows -> {OUT / 'qscan.csv'}")


def superpose(manifest: dict) -> None:
    sent = manifest["sentinels"]

    def one_pt(label: str, q: float) -> dict | None:
        summary = latest_summary(HERE / f"jobs/qscan/{label}_q{q:+.2f}")
        return load_csv(summary)[0] if summary else None

    specs = [
        ("AB_q+0.10", ("RED", "BLUE"), 0.10, "jobs/superpose/AB_q+0.10_combined"),
        ("ABC_q+0.10", ("RED", "BLUE", "ZERO"), 0.10, "jobs/superpose/ABC_q+0.10_combined"),
        ("AB_q+0.50", ("RED", "BLUE"), 0.50, "jobs/superpose/AB_q+0.50_combined"),
    ]
    rows = []
    for name, labels, q, rel in specs:
        cmb_path = latest_summary(HERE / rel)
        parts = [one_pt(lab, q) for lab in labels]
        if cmb_path is None or any(p is None for p in parts):
            continue
        cmb = load_csv(cmb_path)[0]
        rec = {"case": name, "q": q, "labels": "+".join(labels)}
        for col, short in (("gse_effect", "gse"), ("s1_exe_effect", "s1"), ("dm_effect", "dm")):
            s = sum(float(p[col]) for p in parts)  # type: ignore[index]
            c = float(cmb[col])
            rec[f"{short}_sum"] = s
            rec[f"{short}_combined"] = c
            rec[f"{short}_diff"] = s - c
        rows.append(rec)
    if not rows:
        print("superpose: no finished combined jobs yet")
        return
    fields = list(rows[0].keys())
    write_csv(
        OUT / "superpose.csv",
        [{k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
        fields,
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    xs = [r["s1_combined"] for r in rows]
    ys = [r["s1_sum"] for r in rows]
    ax.plot(xs, ys, "o")
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(r["case"], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    lo, hi = min(xs + ys), max(xs + ys)
    pad = 0.05 * (hi - lo if hi != lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
    ax.set_xlabel("combined S1 ΔP / eV")
    ax.set_ylabel("sum of separate S1 ΔP / eV")
    ax.set_title("few-charge superposition")
    fig.tight_layout()
    fig.savefig(OUT / "superpose.png", dpi=150)
    plt.close(fig)
    print(f"superpose: {len(rows)} cases -> {OUT / 'superpose.csv'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full_shell()
    man_path = HERE / "manifest.json"
    if not man_path.exists():
        print("no manifest.json; run prepare.py")
        return
    manifest = json.loads(man_path.read_text())
    qscan_and_sign(manifest)
    superpose(manifest)


if __name__ == "__main__":
    main()
