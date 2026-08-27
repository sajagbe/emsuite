"""Engine protocol tests."""

import pytest

from emsuite.engines import get_engine


def test_get_engine_pyscf():
    engine = get_engine("pyscf")
    assert engine.name == "pyscf"
    assert engine.is_available() is True


def test_pyscf_engine_lives_in_pyscf_module():
    from emsuite.engines.pyscf_engine import PySCFEngine

    assert PySCFEngine().name == "pyscf"


def test_unknown_engine_rejected():
    with pytest.raises(ValueError, match="only 'pyscf'"):
        get_engine("mlip")
