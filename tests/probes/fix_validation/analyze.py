"""Compare fix-validation results with pre-drift reference runs; write REPORT.md.

Checks
  A. Tuning cases: new GPU and CPU maps vs the reference map at the same points,
     and new CPU vs new GPU. Reference thermo values (written in Hartree before the
     unit fix) are converted to kcal/mol / eV / 1/eV first, and are only compared
     when the reference IE/EA baseline was physical (|IE|, |EA| < 1 Ha).
  B. Baseline repeats: spread across identical repeats, CPU vs GPU, physical range,
     and agreement with a sane reference baseline where one exists.

Run from this folder after the jobs finish:  python analyze.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from emsuite.physconst import HARTREE_TO_EV, HARTREE_TO_KCAL

HERE = Path(__file__).resolve().parent
CAMP = HERE.parents[1] / "integration_runs/opsins/campaigns/pose-restrain-2026-10-11"

KCAL = ("ie", "ea", "cp")
EV = ("eng", "hard", "efl")
THERMO = KCAL + EV + ("nfl",)
ONE_HA_KCAL = HARTREE_TO_KCAL

# Extra sane reference baselines (Hartree), same geometry/method/solvent as the case.
EXTRA_REF = {
    "SB_gas": CAMP / "protein/tuning/4gkc/SB/separate_singlet",  # gas baseline, sane IE
}


def base(col: str) -> str:
    for suf in ("_effect", "_baseline"):
        if col.endswith(suf):
            return col[: -len(suf)]
    return col


def to_new_units(prop: str, v: float) -> float:
    if prop in KCAL:
        return v * HARTREE_TO_KCAL
    if prop in EV:
        return v * HARTREE_TO_EV
    if prop == "nfl":
        return v / HARTREE_TO_EV
    return v


def tolerance(prop: str, col: str) -> tuple[float, str]:
    if prop == "gse":
        return (0.05, "abs") if col.endswith("_baseline") else (0.01, "abs")
    if prop == "nfl":
        return 1e-3, "rel"
    if prop in KCAL:
        return 0.05, "abs"
    if prop in EV:
        return 2e-3, "abs"
    return 1e-3, "abs"  # exe/osc/homo/lumo/gap (eV), dm, spin


def fnum(s: str) -> float | None:
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def sane_ha(row: dict) -> bool:
    ie, ea = fnum(row.get("ie_baseline")), fnum(row.get("ea_baseline"))
    return ie is not None and ea is not None and abs(ie) < 1 and abs(ea) < 1


def latest_csv(d: Path) -> Path | None:
    rs = sorted(d.glob("results_*/*_tuning_summary.csv"), key=lambda p: p.stat().st_mtime)
    return rs[-1] if rs else None


def compare(new_rows, ref_rows, ref_is_old: bool, ref_sane: bool):
    """Per-column max diff. new_rows[j] pairs with ref_rows[j]."""
    out = []
    cols = [
        c
        for c in new_rows[0]
        if c in ref_rows[0]
        and c not in ("point_index", "x", "y", "z")
        and not c.endswith("_normalized")
    ]
    for c in cols:
        prop = base(c)
        if prop in THERMO and not ref_sane:
            out.append((c, None, None, "SKIP (reference IE/EA baseline unphysical)"))
            continue
        tol, kind = tolerance(prop, c)
        worst, n = 0.0, 0
        for a, b in zip(new_rows, ref_rows, strict=True):
            x, y = fnum(a[c]), fnum(b[c])
            if x is None or y is None:
                continue
            if ref_is_old:
                y = to_new_units(prop, y)
            d = abs(x - y) / max(abs(y), 1e-12) if kind == "rel" else abs(x - y)
            worst, n = max(worst, d), n + 1
        status = "n/a" if n == 0 else ("PASS" if worst <= tol else "FAIL")
        out.append((c, worst if n else None, f"{tol:g} {kind}", status))
    return out


def section_tuning(manifest, lines, summary):
    lines.append("## A. Tuning maps vs pre-drift references\n")
    for name, m in manifest["tuning"].items():
        ref_rows_all = list(csv.DictReader(open(m["reference_csv"])))
        ref_rows = [ref_rows_all[i] for i in m["reference_points"]]
        ref_sane = sane_ha(ref_rows_all[0])
        new = {}
        for dev in ("gpu", "cpu"):
            p = latest_csv(HERE / "cases" / name / dev)
            new[dev] = list(csv.DictReader(open(p))) if p else None
        lines.append(f"### {name}\n")
        lines.append(f"Reference: `{m['reference_csv']}` points {m['reference_points']}  ")
        lines.append(f"Reference IE/EA baseline physical: **{ref_sane}**\n")
        pairs = [
            ("gpu vs reference", "gpu", None),
            ("cpu vs reference", "cpu", None),
            ("cpu vs gpu", "cpu", "gpu"),
        ]
        for label, a, b in pairs:
            if new[a] is None or (b and new[b] is None):
                lines.append(f"- {label}: **MISSING results**\n")
                summary.append((name, label, "MISSING"))
                continue
            rows_b = new[b] if b else ref_rows
            res = compare(new[a], rows_b, ref_is_old=b is None, ref_sane=ref_sane or bool(b))
            fails = [r for r in res if r[3] == "FAIL"]
            summary.append((name, label, "FAIL" if fails else "PASS"))
            lines.append(f"**{label}: {'FAIL' if fails else 'PASS'}** ({len(res)} columns)\n")
            lines.append("| column | max diff | tolerance | status |\n|---|---|---|---|")
            for c, w, tol, st in res:
                if st != "PASS" or c.endswith("_baseline") or "exe" in c or c.startswith("ie"):
                    ws = "" if w is None else f"{w:.3g}"
                    lines.append(f"| {c} | {ws} | {tol or ''} | {st} |")
            lines.append("")
        for dev in ("gpu", "cpu"):
            if new[dev]:
                ie = fnum(new[dev][0].get("ie_baseline"))
                ea = fnum(new[dev][0].get("ea_baseline"))
                if ie is not None and ea is not None:
                    ok = abs(ie) < ONE_HA_KCAL and abs(ea) < ONE_HA_KCAL
                    summary.append(
                        (name, f"{dev} IE/EA baseline physical", "PASS" if ok else "FAIL")
                    )
                    lines.append(
                        f"- new {dev} baseline: IE {ie:.3f}, EA {ea:.3f} kcal/mol "
                        f"({'physical' if ok else 'UNPHYSICAL'})"
                    )
        lines.append("")


def section_baseline(manifest, lines, summary):
    lines.append("## B. Baseline repeats (IE/EA drift)\n")
    lines.append(
        "| system | device | repeats ok | IE kcal/mol (min..max) | EA kcal/mol (min..max) "
        "| spread IE/EA | reference IE/EA | rejected states | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    per_sys = {}
    for name, m in manifest["baseline"].items():
        ref_row = next(csv.DictReader(open(m["reference_csv"])))
        ref = {}
        for k in ("ie", "ea"):
            v = fnum(ref_row.get(f"{k}_baseline"))
            if v is not None and abs(v) < 1:
                ref[k] = v * HARTREE_TO_KCAL
        if name in EXTRA_REF and (p := latest_csv(EXTRA_REF[name])):
            r2 = next(csv.DictReader(open(p)))
            for k in ("ie", "ea"):
                v = fnum(r2.get(f"{k}_baseline"))
                if k not in ref and v is not None and abs(v) < 1:
                    ref[k] = v * HARTREE_TO_KCAL
        for dev in ("gpu", "cpu"):
            f = HERE / "cases/baseline" / name / dev / "baseline_repeats.jsonl"
            recs = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
            ok = [r for r in recs if r.get("ok") and "ie" in r["raw"] and "ea" in r["raw"]]
            if not ok:
                lines.append(f"| {name} | {dev} | 0/{len(recs)} | | | | | | **MISSING** |")
                summary.append((f"baseline {name}", dev, "MISSING"))
                continue
            ie = [r["raw"]["ie"] for r in ok]
            ea = [r["raw"]["ea"] for r in ok]
            spread = (max(ie) - min(ie), max(ea) - min(ea))
            phys = all(abs(v) < ONE_HA_KCAL for v in ie + ea)
            match = all(
                abs(sum(v) / len(v) - ref[k]) < 0.5 for k, v in (("ie", ie), ("ea", ea)) if k in ref
            )
            rej = sum(len(r["rejected"]) for r in ok)
            verdict = "PASS" if phys and spread[0] < 0.05 and spread[1] < 0.05 and match else "FAIL"
            per_sys.setdefault(name, {})[dev] = (sum(ie) / len(ie), sum(ea) / len(ea))
            refs = "/".join(f"{ref[k]:.2f}" if k in ref else "-" for k in ("ie", "ea"))
            lines.append(
                f"| {name} | {dev} | {len(ok)}/{len(recs)} | {min(ie):.3f}..{max(ie):.3f} | "
                f"{min(ea):.3f}..{max(ea):.3f} | {spread[0]:.2g}/{spread[1]:.2g} | {refs} | {rej} "
                f"| **{verdict}** |"
            )
            summary.append((f"baseline {name}", dev, verdict))
    lines.append(
        "\nReference match tolerance 0.5 kcal/mol; repeat spread tolerance 0.05 kcal/mol.\n"
    )
    lines.append("CPU vs GPU mean baseline difference (kcal/mol):\n")
    for name, d in per_sys.items():
        if "gpu" in d and "cpu" in d:
            die, dea = d["cpu"][0] - d["gpu"][0], d["cpu"][1] - d["gpu"][1]
            st = "PASS" if abs(die) < 0.05 and abs(dea) < 0.05 else "FAIL"
            summary.append((f"baseline {name}", "cpu vs gpu", st))
            lines.append(f"- {name}: IE {die:+.4f}, EA {dea:+.4f} → {st}")
    lines.append("")


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    lines = ["# Fix validation report\n"]
    head = (HERE / "code_under_test.head").read_text().strip()
    lines.append(f"Code under test: HEAD `{head}` + `code_under_test.diff`\n")
    summary: list[tuple[str, str, str]] = []
    body: list[str] = []
    section_tuning(manifest, body, summary)
    section_baseline(manifest, body, summary)
    n_fail = sum(s == "FAIL" for *_, s in summary)
    n_miss = sum(s == "MISSING" for *_, s in summary)
    lines.append(
        f"**Summary: {len(summary) - n_fail - n_miss} PASS, {n_fail} FAIL, {n_miss} MISSING**\n"
    )
    lines.append("| case | check | result |\n|---|---|---|")
    lines += [f"| {a} | {b} | {c} |" for a, b, c in summary]
    lines.append("")
    (HERE / "REPORT.md").write_text("\n".join(lines + body) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
