"""Unit tests for combined-mode single-GPU pinning."""

from __future__ import annotations

from emsuite.tuning.runner import _pin_combined_to_single_gpu


def test_pin_combined_noop_when_single_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert _pin_combined_to_single_gpu() == "2"
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_pin_combined_keeps_first_when_multi_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4,5")
    assert _pin_combined_to_single_gpu() == "3"
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_pin_combined_default_when_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _pin_combined_to_single_gpu() == "0"
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "0"
