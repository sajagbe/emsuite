"""Extended thermodynamic / fugacity properties."""

from __future__ import annotations

import math

from ..constants import HARTREE_TO_EV


def fugacity_extensions(results: dict, props_to_calc: list[str]) -> dict[str, float]:
    """Extended nucleophilicity/electrophilicity fugacity-style indices."""
    out: dict[str, float] = {}
    if "efl_fug" in props_to_calc and "efl" in results and results["efl"] != 0:
        out["efl_fug"] = math.log(abs(results["efl"]))
    if "nfl_fug" in props_to_calc and "nfl" in results and results["nfl"] != 0:
        out["nfl_fug"] = math.log(abs(results["nfl"]))
    if "eng_fug" in props_to_calc and "eng" in results:
        out["eng_fug"] = results["eng"] * HARTREE_TO_EV
    return out
