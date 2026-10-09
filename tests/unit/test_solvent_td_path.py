"""Solvent TDDFT must use mf.TDDFT()/TDHF(), not pyscf.tdscf.TDDFT(mf)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from emsuite.core.excited import create_td_molecule_object


def _mock_td() -> MagicMock:
    td = MagicMock()
    td.e = np.array([0.1, 0.2])
    return td


@pytest.fixture(autouse=True)
def _single_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


def test_solvent_dft_uses_mf_tddft():
    mf = MagicMock()
    mf.xc = "b3lyp"
    td = _mock_td()
    mf.TDDFT.return_value = td

    result = create_td_molecule_object(mf, nstates=2, force_single_gpu=True)

    mf.TDDFT.assert_called_once()
    mf.TDHF.assert_not_called()
    assert result is td
    td.kernel.assert_called_once()


def test_solvent_hf_uses_mf_tdhf():
    mf = MagicMock(spec=["xc", "TDHF"])
    mf.xc = None
    td = _mock_td()
    mf.TDHF.return_value = td

    result = create_td_molecule_object(mf, nstates=2, force_single_gpu=True)

    mf.TDHF.assert_called_once()
    assert result is td


def test_pcm_hf_with_tddft_attr_still_uses_tdhf():
    """PCM defines TDDFT() on HF wrappers; route by xc, not hasattr."""
    mf = MagicMock()
    mf.xc = None
    mf.TDHF = MagicMock()
    td = _mock_td()
    mf.TDHF.return_value = td

    create_td_molecule_object(mf, nstates=2, force_single_gpu=True)

    mf.TDHF.assert_called_once()
    mf.TDDFT.assert_not_called()


def test_dft_tdhf_fallback_when_tddft_missing_on_pcm():
    mf = MagicMock()
    mf.xc = "b3lyp"
    td = _mock_td()
    mf.TDDFT.side_effect = AttributeError("no TDDFT")
    mf.TDHF.return_value = td

    result = create_td_molecule_object(mf, nstates=2, force_single_gpu=True)

    mf.TDDFT.assert_called_once()
    mf.TDHF.assert_called_once()
    assert result is td


def test_excited_avoids_pyscf_tdscf_factory_for_solvent():
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "src"
        / "emsuite"
        / "core"
        / "excited.py"
    )
    text = src.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "tdscf.TDDFT(mf)" not in stripped or "mf_cpu" in stripped
