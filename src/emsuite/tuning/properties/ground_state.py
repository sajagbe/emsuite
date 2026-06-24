"""Ground-state property calculators."""

from __future__ import annotations

import numpy as np

from emsuite.core import find_homo_lumo_and_gap

from ..constants import HARTREE_TO_EV, HARTREE_TO_KCAL

GROUND_STATE_PROPS = ("gse", "homo", "lumo", "gap", "dm", "spin")


def calculate_ground_state_properties(mf, props_to_calc: list[str]) -> dict:
    results: dict = {}
    if "gse" in props_to_calc:
        results["gse"] = mf.e_tot * HARTREE_TO_KCAL

    if any(p in props_to_calc for p in ("homo", "lumo", "gap")):
        homo, lumo, gap = [x * HARTREE_TO_EV for x in find_homo_lumo_and_gap(mf)]
        results.update(
            {
                p: v
                for p, v in zip(["homo", "lumo", "gap"], [homo, lumo, gap], strict=True)
                if p in props_to_calc
            }
        )

    if "dm" in props_to_calc:
        results["dm"] = float(np.linalg.norm(mf.dip_moment()))

    if "spin" in props_to_calc:
        spin_square = getattr(mf, "spin_square", None)
        if spin_square is not None:
            try:
                s2 = float(spin_square()[0])
                results["spin"] = float(np.sqrt(max(s2, 0.0)))
            except Exception:
                results["spin"] = float(getattr(mf.mol, "spin", 0))
        else:
            results["spin"] = float(getattr(mf.mol, "spin", 0))

    return results
