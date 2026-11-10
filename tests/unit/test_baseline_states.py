"""tuning/baseline.py: neutral/anion/cation starting SCFs at the same time, one GPU each."""

import json

import pytest

from emsuite.tuning import baseline


def test_one_state_per_gpu_when_enough_gpus():
    plan = baseline.assign_states(["neutral", "anion", "cation"], ["0", "1", "2", "3"])
    assert plan == {"0": ["neutral"], "1": ["anion"], "2": ["cation"]}


def test_states_share_gpus_when_too_few():
    assert baseline.assign_states(["neutral", "anion", "cation"], ["5"]) == {
        "5": ["neutral", "anion", "cation"]
    }
    assert baseline.assign_states(["neutral", "anion", "cation"], ["0", "1"]) == {
        "0": ["neutral", "cation"],
        "1": ["anion"],
    }


def test_visible_gpus_from_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert baseline.visible_gpus() == ["2", "3"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert baseline.visible_gpus() == []


class _FakeProc:
    """Stands in for a worker process: writes the result files the test asks for."""

    results: dict = {}

    def __init__(self, args, env, stdout, stderr, cwd):
        spec = json.loads(args[-1])
        assert env["CUDA_VISIBLE_DEVICES"] in ("0", "1", "2")
        for state in spec["states"]:
            (cwd / f"baseline_{state}.json").write_text(json.dumps(self.results[state]))
        self.returncode = 0

    def wait(self):
        return 0


def _run(monkeypatch, tmp_path, results):
    _FakeProc.results = results
    monkeypatch.setattr(baseline.subprocess, "Popen", _FakeProc)
    return baseline.run_states(
        "x.xyz", "sto-3g", "dft", "b3lyp", 0, [0], list(results), ["0", "1", "2"], tmp_path
    )


def test_collects_results(monkeypatch, tmp_path):
    res = {
        "neutral": {"chkfile": "n.chk", "e_tot": -10.0, "spin": 0},
        "anion": {"chkfile": "a.chk", "e_tot": -10.1, "spin": 1},
        "cation": {"chkfile": "c.chk", "e_tot": -9.6, "spin": 1},
    }
    assert _run(monkeypatch, tmp_path, res) == res


def test_missing_state_raises(monkeypatch, tmp_path):
    res = {"neutral": {"chkfile": "n.chk", "e_tot": -10.0, "spin": 0}, "anion": {"e_tot": None}}
    with pytest.raises(RuntimeError, match="for the anion state"):
        _run(monkeypatch, tmp_path, res)


def test_ion_far_from_neutral_raises(monkeypatch, tmp_path):
    res = {
        "neutral": {"chkfile": "n.chk", "e_tot": -873.95, "spin": 0},
        "cation": {"chkfile": "c.chk", "e_tot": -1240.39, "spin": 1},
    }
    with pytest.raises(RuntimeError, match="for the cation state"):
        _run(monkeypatch, tmp_path, res)


def test_main_process_pinned_to_first_gpu(monkeypatch):
    from emsuite.tuning import runner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6")
    runner._pin_main_process_to_one_gpu(["4", "5", "6"])
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
