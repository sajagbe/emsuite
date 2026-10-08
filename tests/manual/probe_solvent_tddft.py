#!/usr/bin/env python3
"""Probe why SMD/PCM solvation + TDDFT failed on the live gpu4pyscf 1.4.3 stack.

Root cause (fixed in emsuite.core.excited): routing solvent TD through
``pyscf.tdscf.TDDFT(mf)`` calls ``mf.remove_soscf()``; gpu4pyscf 1.4.3 implements
that as ``lib.logger.warn('...')`` without a logger record →
``TypeError: warn() missing 1 required positional argument: 'msg'``.

This probe confirms ``mf.TDDFT()`` (PCM-aware) works for water + water solvent.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
from pyscf import dft, gto, tdscf

from emsuite.core.excited import create_td_molecule_object
from emsuite.core.molecule import solvate_molecule
from emsuite.core.oscillator_strength import oscillator_strength_cpu


def main() -> int:
    print("numpy", np.__version__)
    print("pyscf", __import__("pyscf").__version__)
    try:
        import gpu4pyscf

        print("gpu4pyscf", getattr(gpu4pyscf, "__version__", "?"), gpu4pyscf.__file__)
    except Exception as exc:
        print("gpu4pyscf import:", type(exc).__name__, exc)

    mol = gto.M(atom="O 0 0 0; H 0 0 0.95; H 0.89 0 -0.32", basis="6-31g*", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    try:
        mf = mf.to_gpu()
        print("to_gpu OK", type(mf))
    except Exception as exc:
        print("to_gpu skipped:", type(exc).__name__, exc)

    mf.kernel()
    mf = solvate_molecule(mf, solvent="water")
    print("solvated type:", type(mf), "with_solvent:", hasattr(mf, "with_solvent"))

    print("\n=== Broken path: pyscf.tdscf.TDDFT(mf) ===")
    try:
        td_bad = tdscf.TDDFT(mf)
        td_bad.nstates = 2
        td_bad.singlet = True
        td_bad.kernel()
        print("UNEXPECTED PASS", np.asarray(td_bad.e))
    except Exception as exc:
        print("FAIL as expected:", type(exc).__name__, exc)

    print("\n=== Fixed path: mf.TDDFT() / create_td_molecule_object ===")
    try:
        td = create_td_molecule_object(mf, nstates=2, triplet=False, force_single_gpu=True)
        e = np.asarray(td.e.get() if hasattr(td.e, "get") else td.e, dtype=float)
        osc = oscillator_strength_cpu(td)
        print("PASS e=", e)
        print("PASS osc=", osc)
        assert np.isfinite(e).all() and e.size == 2
        assert np.isfinite(osc).all() and osc.size == 2
    except Exception:
        traceback.print_exc()
        print("FAIL: solvent TDDFT via mf.TDDFT path")
        return 1

    print("\nSUMMARY: solvent TDDFT works via mf.TDDFT(); avoid pyscf.tdscf.TDDFT(mf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
