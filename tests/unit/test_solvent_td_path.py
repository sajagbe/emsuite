"""Solvent TDDFT must use mf.TDDFT(), not pyscf.tdscf.TDDFT(mf)."""

from __future__ import annotations

from pathlib import Path


def test_excited_avoids_pyscf_tdscf_factory_for_solvent():
    src = Path(__file__).resolve().parents[2] / "src" / "emsuite" / "core" / "excited.py"
    text = src.read_text()
    # The in-process branch must construct TD from the mf object itself.
    assert "td = mf.TDDFT() if hasattr(mf, \"TDDFT\") else mf.TDHF()" in text
    # Documented anti-pattern must not appear as executable code.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "tdscf.TDDFT(mf)" not in stripped or "mf_cpu" in stripped
