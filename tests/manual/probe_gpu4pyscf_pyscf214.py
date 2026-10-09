#!/usr/bin/env python3
"""Probe gpu4pyscf vs PySCF 2.14 on a GPU node."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

print("numpy", __import__("numpy").__version__)
print("pyscf", __import__("pyscf").__version__, __import__("pyscf").__file__)
import gpu4pyscf

print("gpu4pyscf", getattr(gpu4pyscf, "__version__", "?"), gpu4pyscf.__file__)
import gpu4pyscf.dft
import gpu4pyscf.scf

print("gpu4pyscf.scf/dft import OK")

from pyscf import dft, gto, lib

# einsum sanity
lib.einsum("ij,jk,kl->ik", np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 5)))
print("lib.einsum 3-op OK")

mol = gto.M(atom="O 0 0 0; H 0 0 0.95; H 0.89 0 -0.32", basis="6-31g*", verbose=0)

# Path 1: CPU RKS -> to_gpu
print("\n=== Path 1: dft.RKS(...).to_gpu() ===")
mf = dft.RKS(mol)
mf.xc = "b3lyp"
try:
    mf_g = mf.to_gpu()
    print("to_gpu OK", type(mf_g))
    mf_g.kernel()
    print("GPU SCF E", float(mf_g.e_tot))
    td = mf_g.TDDFT()
    td.nstates = 3
    td.singlet = True
    td.kernel()
    e = td.e.get() if hasattr(td.e, "get") else td.e
    print("TDDFT e", np.asarray(e))
    t0 = time.perf_counter()
    try:
        osc = td.oscillator_strength()
        if hasattr(osc, "get"):
            osc = osc.get()
        osc = np.asarray(osc, dtype=float)
        print(f"direct osc OK ({(time.perf_counter() - t0) * 1e3:.3f} ms):", osc)
    except Exception as exc:
        print(
            f"direct osc FAIL ({(time.perf_counter() - t0) * 1e3:.3f} ms):", type(exc).__name__, exc
        )
        traceback.print_exc()
    from emsuite.core.oscillator_strength import oscillator_strength_cpu

    t0 = time.perf_counter()
    osc_cpu = oscillator_strength_cpu(td)
    print(f"cpu helper OK ({(time.perf_counter() - t0) * 1e3:.3f} ms):", osc_cpu)
except Exception as exc:
    print("Path1 FAIL:", type(exc).__name__, exc)
    traceback.print_exc()

# Path 2: construct GPU RKS directly
print("\n=== Path 2: gpu4pyscf.dft.RKS(mol) ===")
try:
    from gpu4pyscf import dft as gdft

    mf2 = gdft.RKS(mol)
    mf2.xc = "b3lyp"
    mf2.kernel()
    print("GPU SCF E", float(mf2.e_tot))
    td2 = mf2.TDDFT()
    td2.nstates = 3
    td2.singlet = True
    td2.kernel()
    e2 = td2.e.get() if hasattr(td2.e, "get") else td2.e
    print("TDDFT e", np.asarray(e2))
    t0 = time.perf_counter()
    try:
        osc2 = td2.oscillator_strength()
        if hasattr(osc2, "get"):
            osc2 = osc2.get()
        osc2 = np.asarray(osc2, dtype=float)
        print(f"direct osc OK ({(time.perf_counter() - t0) * 1e3:.3f} ms):", osc2)
        direct_ok = True
    except Exception as exc:
        print(
            f"direct osc FAIL ({(time.perf_counter() - t0) * 1e3:.3f} ms):", type(exc).__name__, exc
        )
        traceback.print_exc()
        direct_ok = False
        osc2 = None
    from emsuite.core.oscillator_strength import oscillator_strength_cpu

    t0 = time.perf_counter()
    osc_cpu2 = oscillator_strength_cpu(td2)
    print(f"cpu helper OK ({(time.perf_counter() - t0) * 1e3:.3f} ms):", osc_cpu2)
    if osc2 is not None:
        rel = np.max(np.abs(osc_cpu2 - osc2) / np.maximum(np.abs(osc2), 1e-16))
        print("rel max vs direct", rel)
except Exception as exc:
    print("Path2 FAIL:", type(exc).__name__, exc)
    traceback.print_exc()
