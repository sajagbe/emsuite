#!/usr/bin/env python3
"""Distance sweep: is dm a tuning effect or MM-included dipole?

See README.md in this directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto, qmmm

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CAMPAIGN = Path(
    "/data/PHO_WORK/sajagbe2/packages/emsuite/tests/integration_runs/"
    "recreate-maps-2026-10-07T035624Z"
)
WATER_XYZ = CAMPAIGN / "water" / "Water.xyz"
WATER_CSV = (
    CAMPAIGN
    / "water"
    / "separate"
    / "singlet"
    / "results_Water_2026-10-07_04-18-19"
    / "Water_tuning_summary.csv"
)

# Outward offsets (Å) from the native surface point along COM→point.
# Include δ=0 (native surface) plus the requested 1.0–3.0 @ 0.1 Å.
OFFSETS_A = [0.0] + [round(x, 1) for x in np.arange(1.0, 3.0 + 1e-9, 0.1)]
Q_MM = 0.10  # same as campaign surface_charge
BASIS = "6-31G*"
XC = "b3lyp"


def load_xyz(path: Path) -> str:
    lines = path.read_text().strip().splitlines()
    if lines[0].strip().isdigit():
        nat = int(lines[0])
        body = lines[2 : 2 + nat]
    else:
        body = [ln for ln in lines if ln.strip()]
    return ";".join(" ".join(ln.split()[:4]) for ln in body)


def load_extreme_points(csv_path: Path) -> list[dict]:
    rows: list[dict[str, str]] = []
    with csv_path.open() as f:
        for raw in csv.DictReader(f):
            rows.append({k.strip(): v.strip() for k, v in raw.items()})
    if not rows:
        raise SystemExit(f"empty {csv_path}")

    def eff(r: dict[str, str]) -> float:
        return float(r["dm_effect"])

    rmax = max(rows, key=eff)
    rmin = min(rows, key=eff)
    out = []
    for label, r in (("MAX", rmax), ("MIN", rmin)):
        out.append(
            {
                "label": label,
                "point_index": int(float(r["point_index"])),
                "dm_effect_map": float(r["dm_effect"]),
                "xyz": np.array(
                    [float(r["x"]), float(r["y"]), float(r["z"])], dtype=float
                ),
            }
        )
    return out


def mol_com(mol: gto.Mole) -> np.ndarray:
    """Mass-weighted COM in Å (PySCF atom_coords are Bohr → convert)."""
    from pyscf.data.nist import BOHR

    coords_a = mol.atom_coords() * BOHR  # Bohr → Å
    masses = np.array([mol.atom_mass_list()[i] for i in range(mol.natm)], dtype=float)
    return (masses[:, None] * coords_a).sum(axis=0) / masses.sum()


def dm_norm(mf) -> float:
    """Same as emsuite: ||dip_moment()|| in Debye."""
    return float(np.linalg.norm(mf.dip_moment(verbose=0)))


def dm_vector(mf) -> np.ndarray:
    return np.asarray(mf.dip_moment(verbose=0), dtype=float)


def run_sweep(point: dict) -> list[dict]:
    mol = gto.M(atom=load_xyz(WATER_XYZ), basis=BASIS, charge=0, spin=0, verbose=0)
    mol.build()
    com = mol_com(mol)
    r0 = point["xyz"]
    vec = r0 - com
    dist0 = float(np.linalg.norm(vec))
    uhat = vec / dist0

    print(f"\n=== point {point['label']} idx={point['point_index']} ===")
    print(f"surface xyz (Å) = {r0}")
    print(f"COM (Å)         = {com}")
    print(f"|r0-COM| (Å)    = {dist0:.4f}")
    print(f"map dm_effect   = {point['dm_effect_map']:.6f}")

    mf0 = dft.RKS(mol)
    mf0.xc = XC
    mf0.kernel()
    dm0 = dm_norm(mf0)
    mu0 = dm_vector(mf0)
    print(f"vacuum |μ| (D)  = {dm0:.6f}")

    rows: list[dict] = []
    for delta in OFFSETS_A:
        r = r0 + delta * uhat  # outward along COM→surface ray
        dist_com = float(np.linalg.norm(r - com))
        mf = dft.RKS(mol)
        mf.xc = XC
        mf = qmmm.mm_charge(mf, r.reshape(1, 3), np.array([Q_MM]))
        mf.kernel()
        if not mf.converged:
            print(f"  WARN: SCF not converged at δ={delta}")
        mu = dm_vector(mf)
        dm = float(np.linalg.norm(mu))
        # Counterfactual: add classical MM dipole contribution in Debye
        # μ_MM = q * (r - origin)_Bohr * AU2DEBYE; PySCF origin default = (0,0,0) Bohr
        from pyscf.data import nist

        r_bohr = r / nist.BOHR
        mu_mm = Q_MM * r_bohr * nist.AU2DEBYE
        mu_incl = mu + mu_mm
        dm_incl = float(np.linalg.norm(mu_incl))
        row = {
            "label": point["label"],
            "point_index": point["point_index"],
            "delta_A": delta,
            "dist_com_A": dist_com,
            "x": float(r[0]),
            "y": float(r[1]),
            "z": float(r[2]),
            "dm_vacuum": dm0,
            "dm_wsc": dm,
            "dm_effect": dm - dm0,  # signed, same spirit as emsuite effect
            "dm_effect_abs": abs(dm - dm0),
            "dm_incl_mm": dm_incl,
            "dm_incl_effect": dm_incl - dm0,
            "scf_converged": bool(mf.converged),
            "mu0_xyz": mu0.tolist(),
            "mu_xyz": mu.tolist(),
        }
        rows.append(row)
        print(
            f"  δ={delta:4.1f}  r_com={dist_com:5.3f}  "
            f"|μ|={dm:8.5f}  Δμ={dm - dm0:+8.5f}  |Δμ|={abs(dm - dm0):8.5f}  "
            f"|μ+q·r|={dm_incl:8.5f}  Δ(incl)={dm_incl - dm0:+8.5f}"
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "label",
        "point_index",
        "delta_A",
        "dist_com_A",
        "x",
        "y",
        "z",
        "dm_vacuum",
        "dm_wsc",
        "dm_effect",
        "dm_effect_abs",
        "dm_incl_mm",
        "dm_incl_effect",
        "scf_converged",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_rows(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for ax, key, title, ylabel in (
        (
            axes[0],
            "dm_effect_abs",
            "emsuite-style |Δμ| (QM-only dip_moment)",
            "|μ_wsc − μ₀| (Debye)",
        ),
        (
            axes[1],
            "dm_incl_effect",
            "counterfactual Δ|μ| if MM added to μ",
            "‖μ_QM + q·r‖ − μ₀  (Debye)",
        ),
    ):
        for label, color in (("MAX", "#1565C0"), ("MIN", "#C62828")):
            sub = [r for r in rows if r["label"] == label]
            xs = [r["delta_A"] for r in sub]
            ys = [r[key] if key != "dm_incl_effect" else abs(r[key]) for r in sub]
            # for incl panel show absolute deviation of the included dipole from vacuum
            if key == "dm_incl_effect":
                ys = [abs(r["dm_incl_mm"] - r["dm_vacuum"]) for r in sub]
            ax.plot(xs, ys, "o-", color=color, ms=4, lw=1.2, label=label)
        ax.axvline(0.0, color="#9E9E9E", lw=0.8, ls="--")
        ax.set_xlabel("outward offset δ from surface point (Å)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle(
        f"Water B3LYP/{BASIS}, q={Q_MM}: dipole vs charge distance\n"
        "Left shrinks with δ ⇒ tuning; right grows ⇒ MM-in-μ artifact",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {path.with_suffix('.svg')}")
    print(f"wrote {path.with_suffix('.pdf')}")


def verdict(rows: list[dict]) -> dict:
    """Compare |Δμ| at δ=1.0 vs δ=3.0 for each label."""
    out = {}
    for label in ("MAX", "MIN"):
        sub = {r["delta_A"]: r for r in rows if r["label"] == label}
        a1 = sub[1.0]["dm_effect_abs"]
        a3 = sub[3.0]["dm_effect_abs"]
        i1 = abs(sub[1.0]["dm_incl_mm"] - sub[1.0]["dm_vacuum"])
        i3 = abs(sub[3.0]["dm_incl_mm"] - sub[3.0]["dm_vacuum"])
        out[label] = {
            "abs_effect_d1": a1,
            "abs_effect_d3": a3,
            "qm_only_shrinks": a3 < a1,
            "incl_d1": i1,
            "incl_d3": i3,
            "incl_grows": i3 > i1,
            "interpretation": (
                "TUNING (QM-only |Δμ| shrinks with distance)"
                if a3 < a1
                else "NOT shrinking — inspect carefully"
            ),
        }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    points = load_extreme_points(WATER_CSV)
    all_rows: list[dict] = []
    for pt in points:
        all_rows.extend(run_sweep(pt))

    csv_path = OUT / "dm_distance_sweep.csv"
    write_csv(all_rows, csv_path)
    print(f"wrote {csv_path}")

    plot_rows(all_rows, OUT / "dm_distance_sweep")
    v = verdict(all_rows)
    (OUT / "verdict.json").write_text(json.dumps(v, indent=2) + "\n")
    print("\n=== VERDICT ===")
    print(json.dumps(v, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
