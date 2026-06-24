"""Engine protocol tests."""

import pytest

from emsuite.engines import MLIPEngine, get_engine


def test_mlip_engine_not_available():
    engine = MLIPEngine()
    assert engine.is_available() is False


def test_mlip_engine_raises_not_implemented():
    engine = MLIPEngine()
    with pytest.raises(NotImplementedError):
        engine.optimize_geometry("missing.xyz")


def test_get_engine_pyscf():
    engine = get_engine("pyscf")
    assert engine.name == "pyscf"
    assert engine.is_available() is True


def test_pyscf_engine_lives_in_pyscf_module():
    from emsuite.engines.pyscf_engine import PySCFEngine

    assert PySCFEngine().name == "pyscf"
