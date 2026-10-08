"""SMD solvent key resolution (case-sensitive PySCF keys)."""

from __future__ import annotations

import pytest

from emsuite.core.molecule import resolve_smd_solvent


def test_resolve_common_lower_keys():
    assert resolve_smd_solvent("water") == "water"
    assert resolve_smd_solvent("n-hexane") == "n-hexane"
    assert resolve_smd_solvent("dimethylsulfoxide") == "dimethylsulfoxide"


def test_resolve_nmf_preserves_canonical_case():
    key = "N-methylformamide(E/Zmixture)"
    assert resolve_smd_solvent(key) == key
    assert resolve_smd_solvent(key.lower()) == key


def test_resolve_unknown_raises():
    with pytest.raises(KeyError, match="Unknown SMD solvent"):
        resolve_smd_solvent("not-a-real-solvent")
