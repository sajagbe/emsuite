#!/usr/bin/env python3
"""Fast smoke/bench: CPU-safe oscillator strength vs direct td.oscillator_strength().

Builds a tiny RKS-B3LYP/6-31G* water molecule, runs ~3 singlet TDDFT states,
times (A) direct osc and (B) emsuite CPU extract, and checks numerical agreement
when both succeed (or vs a NumPy-backed reference when direct fails on GPU).

Usage (from repo root / worktree)::

    PYTHONPATH=src python tests/manual/bench_osc_cpu_extract.py

Optional 1-GPU SLURM smoke::

    sbatch tests/manual/bench_osc_cpu_extract.slurm
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from pyscf import dft, gto, lib

_ORIG_LIB_EINSUM = lib.einsum


def _np_einsum(subscripts, *ops, **kw):
    return np.einsum(subscripts, *ops, optimize=True)


def _lib_einsum_broken() -> bool:
    try:
        a = np.zeros((2, 3, 4))
        b = np.zeros((5, 4))
        c = np.zeros((5, 3))
        _ORIG_LIB_EINSUM("xov,pv,qo->xpq", a, b, c)
        return False
    except ValueError:
        return True


def _patch_lib_einsum() -> None:
    lib.einsum = _np_einsum
    print("NOTE: patched pyscf.lib.einsum → numpy.einsum (NumPy 2.4 compat for TD kernel)")


def _try_gpu(mf):
    try:
        import cupy as cp

        _ = cp.cuda.runtime.getDeviceCount()
        if hasattr(mf, "to_gpu"):
            return mf.to_gpu(), True
    except Exception as exc:
        print(f"GPU unavailable ({type(exc).__name__}: {exc}); using CPU")
    return mf, False


def main() -> int:
    # Login nodes with NumPy≥2.4 need the patch before TD kernel; GPU nodes
    # usually run the kernel fine via gpu4pyscf and only fail on osc extract.
    preempt_patch = _lib_einsum_broken()
    if preempt_patch:
        _patch_lib_einsum()

    mol = gto.M(
        atom="O 0 0 0; H 0 0 0.95; H 0.89 0 -0.32",
        basis="6-31g*",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf, used_gpu = _try_gpu(mf)
    mf.kernel()

    td = mf.TDDFT() if hasattr(mf, "TDDFT") else mf.TDHF()
    td.nstates = 3
    td.singlet = True
    try:
        td.kernel()
    except ValueError as exc:
        if "not enough values to unpack" in str(exc) and lib.einsum is _ORIG_LIB_EINSUM:
            _patch_lib_einsum()
            td.kernel()
        else:
            raise

    e_np = td.e.get() if hasattr(td.e, "get") else td.e
    print(f"GPU used: {used_gpu}")
    print(f"lib.einsum preempt-patched: {preempt_patch}")
    print(f"td.e type: {type(td.e)}")
    print(f"excitation energies (Ha): {np.asarray(e_np)}")

    # --- A) direct: restore original lib.einsum so GPU failure is visible ---
    lib.einsum = _ORIG_LIB_EINSUM
    t0 = time.perf_counter()
    osc_direct = None
    direct_err = None
    try:
        osc_direct = td.oscillator_strength()
        if hasattr(osc_direct, "get"):
            osc_direct = osc_direct.get()
        osc_direct = np.asarray(osc_direct, dtype=float).ravel()
    except Exception as exc:
        direct_err = exc
        traceback.print_exc()
    t_direct = time.perf_counter() - t0
    # Re-apply patch if needed for any later code
    if preempt_patch:
        lib.einsum = _np_einsum

    # --- B) CPU extract (pure numpy; independent of lib.einsum) ---
    from emsuite.core.oscillator_strength import oscillator_strength_cpu

    t0 = time.perf_counter()
    osc_cpu = oscillator_strength_cpu(td)
    t_cpu = time.perf_counter() - t0

    print(f"A) direct oscillator_strength:  {t_direct * 1e3:.3f} ms", end="")
    if direct_err is not None:
        print(f"  FAILED: {type(direct_err).__name__}: {direct_err}")
    else:
        print(f"  osc={osc_direct}")

    print(f"B) oscillator_strength_cpu:     {t_cpu * 1e3:.3f} ms  osc={osc_cpu}")

    assert osc_cpu is not None and np.isfinite(osc_cpu).all() and osc_cpu.size == 3

    if osc_direct is not None:
        rel = np.abs(osc_cpu - osc_direct) / np.maximum(np.abs(osc_direct), 1e-16)
        print(f"max relative diff (vs direct): {rel.max():.3e}")
        assert np.allclose(osc_cpu, osc_direct, rtol=1e-6, atol=1e-8)
        print("PASS: CPU extract matches direct (~1e-6 rel)")
    else:
        # Direct failed (expected on GPU / unpatched lib.einsum). Cross-check by
        # reconstructing a CPU TD with NumPy MOs and patched einsum.
        from pyscf import tdscf

        from emsuite.core.oscillator_strength import _xy_to_numpy, as_numpy

        mf_cpu = mf.to_cpu() if hasattr(mf, "to_cpu") else mf
        td_ref = tdscf.TDDFT(mf_cpu) if hasattr(mf_cpu, "xc") else tdscf.TDHF(mf_cpu)
        td_ref.singlet = td.singlet
        td_ref.e = as_numpy(td.e)
        td_ref.xy = _xy_to_numpy(td.xy)
        td_ref.converged = True
        lib.einsum = _np_einsum
        osc_ref = np.asarray(td_ref.oscillator_strength(), dtype=float).ravel()
        rel = np.abs(osc_cpu - osc_ref) / np.maximum(np.abs(osc_ref), 1e-16)
        print(f"max relative diff (vs CPU-ref): {rel.max():.3e}")
        assert np.allclose(osc_cpu, osc_ref, rtol=1e-6, atol=1e-8)
        print("PASS: direct failed as expected; CPU extract matches CPU reference")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
