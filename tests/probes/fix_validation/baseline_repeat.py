"""Repeat the tuning baseline (neutral, anion, cation; solvated if the run is) N times.

Usage: python baseline_repeat.py CASE_DIR N

Uses the same runner functions as `emsuite -t` (create_molecule_objects,
apply_solvation, calculate_all_properties). Writes CASE_DIR/baseline_repeats.jsonl,
one JSON record per repeat: energies, chosen spins, <S^2>, convergence flags,
rejected spin states, and the thermo/orbital properties.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

from emsuite import core
from emsuite.inputs import TuningInput
from emsuite.tuning.properties import calculate_all_properties, setup_calculation
from emsuite.tuning.runner import apply_solvation, create_molecule_objects

PROPS = ["gse", "homo", "lumo", "gap", "ie", "ea", "cp", "eng", "hard", "efl", "nfl"]


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def describe(mf):
    if mf is None:
        return None
    rec = {"e_tot": float(mf.e_tot), "converged": bool(mf.converged), "spin": int(mf.mol.spin)}
    if mf.mol.spin > 0:
        try:
            rec["s2"] = float(mf.spin_square()[0])
        except Exception as e:  # noqa: BLE001
            rec["s2_error"] = str(e)
    return rec


def main(case_dir: str, n: int) -> None:
    case = Path(case_dir).resolve()
    os.chdir(case)
    p = TuningInput.from_file("tuning.in").to_dict()
    xyz = str((case / p["molecule"]).resolve())
    props, req = setup_calculation(PROPS)
    gpu = (core.check_gpu_info() or 0) > 0
    out = case / "baseline_repeats.jsonl"
    for r in range(n):
        rep = case / f"rep{r}"
        rep.mkdir(exist_ok=True)
        os.chdir(rep)
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = Tee(old, buf)
        t0 = time.time()
        try:
            neutral, anion, cation, _td = create_molecule_objects(
                xyz,
                p["basis_set"],
                p["spin"],
                p["method"],
                p["functional"],
                p["charge"],
                gpu,
                req,
                p["state_of_interest"],
                p["triplet"],
            )
            gas = {
                "neutral": describe(neutral),
                "anion": describe(anion),
                "cation": describe(cation),
            }
            solv = None
            if p["solvent"]:
                neutral, anion, cation, *_ = apply_solvation(
                    [neutral, anion, cation, None, None, None, None, None],
                    p["solvent"],
                    p["state_of_interest"],
                    p["triplet"],
                    req,
                )
                solv = {
                    "neutral": describe(neutral),
                    "anion": describe(anion),
                    "cation": describe(cation),
                }
            raw = calculate_all_properties(
                neutral,
                anion_mf=anion,
                cation_mf=cation,
                td_obj=None,
                triplet=p["triplet"],
                props_to_calc=props,
            )
            rec = {
                "repeat": r,
                "ok": True,
                "gpu": gpu,
                "solvent": p["solvent"],
                "gas": gas,
                "solvated": solv,
                "raw": raw,
            }
        except Exception as e:  # noqa: BLE001
            rec = {"repeat": r, "ok": False, "error": repr(e)}
        finally:
            sys.stdout = old
        log = buf.getvalue()
        rec["rejected"] = [ln.strip() for ln in log.splitlines() if "REJECTED" in ln]
        rec["soscf_lines"] = sum(
            "SOSCF" in ln or "not converged" in ln.lower() for ln in log.splitlines()
        )
        rec["seconds"] = round(time.time() - t0, 1)
        with out.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"[repeat {r}] ok={rec['ok']} {rec.get('raw', rec.get('error'))}", flush=True)
        os.chdir(case)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
