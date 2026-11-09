"""How much do the ligand radii change emsuite's APBS surface maps?

For each ligand (alone in water, as in the potential / coupled modules):
  1. HF/6-31G* RESP charges (CPU port, single stage) are computed once.
  2. One vsg surface is generated (same points for every setting).
  3. APBS is run with emsuite's own settings (run_apbs_grids: mg-auto, dime 65,
     lpbe, smol, srad 1.4, sdie 78.54) for each charge/radius setting below, at
     pdie 2 (emsuite default) and pdie 1 (the interior dielectric ZAP9 was fitted with).
  4. Both surface quantities emsuite maps are read off the same grids:
     potential (kT/e) and Gauss-law surface charge (e).

Settings (charges + radii):
  current        Gasteiger + Mantina   (what emsuite does today for a lone ligand)
  b3lyp_mantina  B3LYP/6-31G* RESP + Mantina (charges from results_acidsgcn001.json)
  hf_mantina     HF/6-31G* RESP + Mantina     <- reference for the radius comparison
  hf_bondi       HF/6-31G* RESP + Bondi
  hf_zap9_2008   HF/6-31G* RESP + ZAP9 (Nicholls 2008 Table 2)
  hf_zap9_2010   HF/6-31G* RESP + ZAP9 (Nicholls 2010 rules)

Outputs, per ligand, in apbs_compare/<name>/:
  <setting>_pdie<k>.pqr        the exact PQR APBS read
  points.csv                   every surface point, every setting (potential and charge)
  hf_resp_charges.txt          the HF RESP charges used
and apbs_compare/<name>_summary.txt with the comparison tables.

Usage: python apbs_radii_compare.py [LF SB rPSB]
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import resp_cpu
from compare_gpu import MOLECULES, build_mol
from pyscf import scf
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from zap9_radii import radii_for

from emsuite.potential.apbs import run_apbs_grids
from emsuite.potential.charges import partial_charges_from_xyz
from emsuite.potential.gauss import charges_at_points, potential_at_points
from emsuite.surface import load_surf
from emsuite.surface.generate import generate_surface

HERE = Path(__file__).resolve().parent
OUT = HERE / "apbs_compare"
REF = "hf_mantina"
PDIES = (2.0, 1.0)


def radius_sets(xyz: Path, charge: int) -> tuple[list[str], dict[str, np.ndarray]]:
    mol = Chem.MolFromXYZFile(str(xyz))
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    cols = {"mantina": [], "bondi": [], "zap9_2008": [], "zap9_2010": []}
    for a in mol.GetAtoms():
        mant, bon, _p2p, z08, z10, _note = radii_for(a)
        cols["mantina"].append(mant)
        cols["bondi"].append(bon)
        cols["zap9_2008"].append(z08)
        cols["zap9_2010"].append(z10)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    return symbols, {k: np.asarray(v) for k, v in cols.items()}


def write_pqr(path: Path, symbols, coords, charges, radii) -> Path:
    with path.open("w") as f:
        for i, (s, (x, y, z), q, r) in enumerate(
            zip(symbols, coords, charges, radii, strict=True), 1
        ):
            f.write(
                f"ATOM  {i:5d} {s:>2s}{i:<2d} LIG     1    {x:8.3f}{y:8.3f}{z:8.3f} {q:8.4f} {r:7.4f}\n"
            )
    return path


def surface_points(xyz: Path, work: Path) -> np.ndarray:
    cwd = os.getcwd()
    os.chdir(work)
    try:
        surf = generate_surface(
            input_type="XYZ",
            input_data=str(xyz),
            output_surf="surface.surf",
            surface_density=0.5,
            surface_scale=1.0,
            surface_type="homogenous",
            surface_charge=0.0,
            optimize=False,
        )
        coords, _ = load_surf(surf)
    finally:
        os.chdir(cwd)
    return coords


def stats_vs(ref: np.ndarray, x: np.ndarray) -> dict:
    d = x - ref
    return {
        "mean": float(x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
        "rms_diff": float(np.sqrt(np.mean(d**2))),
        "max_abs_diff": float(np.max(np.abs(d))),
        "rel_rms_diff": float(np.sqrt(np.mean(d**2)) / np.sqrt(np.mean(ref**2))),
        "corr": float(np.corrcoef(ref, x)[0, 1]),
    }


def run(name: str) -> None:
    xyz, charge = MOLECULES[name]
    work = OUT / name
    work.mkdir(parents=True, exist_ok=True)
    log = [
        f"{name}  charge {charge}  host {socket.gethostname()}  {time.strftime('%Y-%m-%d %H:%M')}"
    ]

    # charges
    mol = build_mol(xyz, charge)
    t = time.time()
    mf = scf.RHF(mol).run()
    q_hf = resp_cpu.resp_solve(mol, mf.make_rdm1())
    log.append(
        f"HF/6-31G* SCF E = {mf.e_tot:.8f} Ha  converged={mf.converged}  ({time.time() - t:.0f} s)"
    )
    prev = {r["molecule"]: r for r in json.loads((HERE / "results_acidsgcn001.json").read_text())}
    q_b3 = np.asarray(prev[name]["cpu_chain_resp"])
    _atoms, q_gast = partial_charges_from_xyz(str(xyz), charge=charge)

    symbols, radii = radius_sets(xyz, charge)
    coords = mol.atom_coords(unit="Angstrom")
    (work / "hf_resp_charges.txt").write_text(
        "atom   HF/6-31G* RESP   B3LYP/6-31G* RESP   Gasteiger\n"
        + "".join(
            f"{s}{i + 1:<5}{a:>14.5f}{b:>18.5f}{c:>14.5f}\n"
            for i, (s, a, b, c) in enumerate(zip(symbols, q_hf, q_b3, q_gast, strict=True))
        )
        + f"sum   {q_hf.sum():>14.5f}{q_b3.sum():>18.5f}{q_gast.sum():>14.5f}\n"
    )

    settings = {
        "current": (q_gast, radii["mantina"]),
        "b3lyp_mantina": (q_b3, radii["mantina"]),
        "hf_mantina": (q_hf, radii["mantina"]),
        "hf_bondi": (q_hf, radii["bondi"]),
        "hf_zap9_2008": (q_hf, radii["zap9_2008"]),
        "hf_zap9_2010": (q_hf, radii["zap9_2010"]),
    }

    pts = surface_points(xyz, work)
    log.append(f"surface points: {len(pts)} (vsg, density 0.5, scale 1.0)")

    table = {}
    columns = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}
    for pdie in PDIES:
        for key, (q, r) in settings.items():
            tag = f"{key}_pdie{int(pdie)}"
            pqr = write_pqr(work / f"{tag}.pqr", symbols, coords, q, r)
            t = time.time()
            g = run_apbs_grids(
                pqr_path=pqr, box_coords=coords, pdie=pdie, sdie=78.54, workdir=work / f"apbs_{tag}"
            )
            pot = potential_at_points(g.potential, pts)
            chg = charges_at_points(g.potential, g.dielx, g.diely, g.dielz, pts)
            columns[f"pot_{tag}"] = pot
            columns[f"chg_{tag}"] = chg
            table[tag] = (pot, chg)
            print(f"{name} {tag}: {time.time() - t:.1f} s", flush=True)

    hdr = ",".join(columns)
    np.savetxt(
        work / "points.csv",
        np.column_stack(list(columns.values())),
        delimiter=",",
        header=hdr,
        comments="",
        fmt="%.6e",
    )

    for qty, idx, unit in (("POTENTIAL", 0, "kT/e"), ("GAUSS-LAW SURFACE CHARGE", 1, "e")):
        for pdie in PDIES:
            ref = table[f"{REF}_pdie{int(pdie)}"][idx]
            log += [
                "",
                f"{qty} ({unit}) at pdie {int(pdie)}; differences are vs {REF}_pdie{int(pdie)}",
                f"{'setting':<16}{'mean':>10}{'min':>10}{'max':>10}{'rms diff':>10}{'max|diff|':>11}{'rel rms':>9}{'corr':>8}",
            ]
            fmt = ">10.4f" if idx == 0 else ">10.2e"
            for key in settings:
                s = stats_vs(ref, table[f"{key}_pdie{int(pdie)}"][idx])
                log.append(
                    f"{key:<16}{s['mean']:{fmt}}{s['min']:{fmt}}{s['max']:{fmt}}{s['rms_diff']:{fmt}}"
                    f" {s['max_abs_diff']:{fmt}}{s['rel_rms_diff']:>9.1%}{s['corr']:>8.4f}"
                )
        # pdie effect on the same setting
        log += ["", f"{qty}: pdie 1 vs pdie 2 for the same setting (rel rms diff)"]
        for key in settings:
            a, b = table[f"{key}_pdie2"][idx], table[f"{key}_pdie1"][idx]
            log.append(f"{key:<16}{stats_vs(a, b)['rel_rms_diff']:>9.1%}")

    log += ["", "1 kT/e = 25.69 mV at 298.15 K. rel rms = rms(diff) / rms(reference)."]
    summary = OUT / f"{name}_summary.txt"
    summary.write_text("\n".join(log) + "\n")
    print("\n".join(log), flush=True)
    print("wrote", summary)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for n in sys.argv[1:] or ["LF", "SB", "rPSB"]:
        run(n)
