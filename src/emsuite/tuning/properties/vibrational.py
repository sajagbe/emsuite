"""Vibrational frequency from Hessian."""

from __future__ import annotations

import numpy as np


def fundamental_frequency_cm1(mf) -> float:
    """
    Return the lowest real harmonic frequency (cm^-1) from the Hessian.

    Falls back to 0.0 if the Hessian calculation fails.
    """
    if mf is None:
        return 0.0
    try:
        from pyscf.hessian import thermo

        hess_mod = mf.Hessian()
        hessian = hess_mod.kernel()
        freq_info = thermo.harmonic_analysis(
            mf.mol,
            hessian,
            imaginary_freq=True,
        )
        frequencies = np.asarray(freq_info["freq_wavenumber"], dtype=float)
        real = frequencies[frequencies > 1.0]
        if real.size == 0:
            return 0.0
        return float(np.min(real))
    except Exception:
        return 0.0
