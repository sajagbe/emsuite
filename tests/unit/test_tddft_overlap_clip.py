"""TDDFT one-root overlap clip: tiny negative metric must stay usable."""

import numpy as np

from emsuite.core.excited import _clip_overlap_eigs, _stabilize_pyscf_tddft_overlap


def test_clip_tiny_negative_overlap():
    w = np.array([-1e-16, 1.0])
    out = _clip_overlap_eigs(w, 1e-12)
    assert np.all(out > 0)
    assert out[1] == 1.0


def test_clip_leaves_healthy_weights():
    w = np.array([0.5, 1.2])
    out = _clip_overlap_eigs(w)
    assert np.allclose(out, w)


def test_stabilize_is_idempotent():
    from pyscf.tdscf import _lr_eig

    _stabilize_pyscf_tddft_overlap()
    first = _lr_eig.VW_Gram_Schmidt_fill_holder
    _stabilize_pyscf_tddft_overlap()
    assert _lr_eig.VW_Gram_Schmidt_fill_holder is first
    assert _lr_eig._emsuite_overlap_clip is True
