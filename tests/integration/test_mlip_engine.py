"""v1.2 MLIP/xTB engine optional integration test."""

from __future__ import annotations

import pytest

from emsuite.engines import get_engine

from .helpers import record_assertions, write_methane_xyz


@pytest.mark.slow
def test_mlip_engine_optional(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    xyz = write_methane_xyz(tmp_path)

    engine = get_engine("mlip")
    available = engine.is_available()
    description = engine.describe()

    energy = None
    if available:
        energy = engine.single_point_energy(str(xyz))

    record_assertions(
        tmp_path,
        engine_name="mlip",
        available=available,
        description=description,
        single_point_energy=energy,
        install_hint="uv sync --extra mlip" if not available else None,
    )

    if not available:
        pytest.skip("mlip extra not installed; install with: uv sync --extra mlip")

    assert energy is not None
