"""GPU integration tests for surface, potential, tuning, and coupled channels.

Run on a GPU allocation, e.g.::

    ./scripts/run_gpu_integration.sh

Or interactively::

    srun --partition=qDEV --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
    python -m pytest tests/integration/test_gpu_channels.py -v -m gpu
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emsuite.core import GPU_AVAILABLE, check_gpu_info
from emsuite.inputs import CoupledInput, PotentialInput, SurfaceInput, TuningInput

from .helpers import METHANE_SURFACE_IN, latest_results_dir, record_assertions, write_methane_xyz

_GPU_PROPS = ("homo", "lumo", "gap")
_GPU_BASIS = "sto-3g"


def _gpu_available() -> bool:
    return bool(GPU_AVAILABLE and (check_gpu_info() or 0) >= 1)


@pytest.fixture(scope="module")
def require_gpu() -> None:
    if not _gpu_available():
        pytest.skip("GPU not available — allocate a GPU node (see scripts/run_gpu_integration.sh)")


def _prepare_methane_surface(tmp_path: Path) -> tuple[Path, Path]:
    """UFF VDW surface + methane.xyz in *tmp_path*."""
    (tmp_path / "surface.in").write_text(METHANE_SURFACE_IN)
    surf = SurfaceInput.from_file(tmp_path / "surface.in").run()
    return Path(surf.path), tmp_path / "methane.xyz"


@pytest.mark.gpu
def test_gpu_preflight(tmp_path: Path, require_gpu: None) -> None:
    """Confirm CuPy/gpu4pyscf see at least one GPU on this node."""
    count = check_gpu_info() or 0
    assert count >= 1
    record_assertions(tmp_path, gpu_count=count, gpu_available=True)


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_surface_uff_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_gpu: None
) -> None:
    """Surface channel smoke on a GPU node (UFF path is CPU; validates env)."""
    monkeypatch.chdir(tmp_path)
    surf_path, xyz_path = _prepare_methane_surface(tmp_path)
    assert surf_path.is_file()
    assert xyz_path.is_file()
    lines = surf_path.read_text().strip().splitlines()
    assert len(lines) >= 11
    record_assertions(
        tmp_path, channel="surface", optimize_method="uff", surface_points=len(lines) - 1
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_surface_pyscf_optimize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_gpu: None
) -> None:
    """Surface channel with PySCF geometry optimization (gpu4pyscf when available)."""
    monkeypatch.chdir(tmp_path)
    write_methane_xyz(tmp_path)
    result = SurfaceInput.from_config(
        input_type="XYZ",
        input_data="methane.xyz",
        output_surf="methane_pyscf.surf",
        optimized_xyz="methane_pyscf.xyz",
        surface_density=0.5,
        surface_scale=1.0,
        surface_type="homogenous",
        surface_charge=0.1,
        optimize=True,
        optimize_method="pyscf",
        basis_set=_GPU_BASIS,
        method="dft",
        functional="b3lyp",
    ).run()
    assert Path(result.path).is_file()
    assert Path("methane_pyscf.xyz").is_file()
    record_assertions(tmp_path, channel="surface", optimize_method="pyscf", surf_path=result.path)


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_potential_apbs_potential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_gpu: None
) -> None:
    """Potential channel: APBS electrostatic potential map (CPU math on GPU node)."""
    monkeypatch.chdir(tmp_path)
    surf_path, _ = _prepare_methane_surface(tmp_path)
    result = PotentialInput.from_config(
        molecule="methane.xyz",
        surface_file=str(surf_path),
        output_surf="methane_potential.surf",
        method="apbs",
        quantity="potential",
    ).run()
    assert Path(result.path).is_file()
    assert np.all(np.isfinite(result.values))
    record_assertions(
        tmp_path,
        channel="potential",
        quantity="potential",
        surface_points=len(result.values),
        values_finite=True,
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_potential_apbs_gauss_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_gpu: None
) -> None:
    """Potential channel: Gauss-law surface charges from APBS φ and dielectric maps."""
    monkeypatch.chdir(tmp_path)
    surf_path, _ = _prepare_methane_surface(tmp_path)
    result = PotentialInput.from_config(
        molecule="methane.xyz",
        surface_file=str(surf_path),
        output_surf="methane_charge.surf",
        method="apbs",
        quantity="charge",
    ).run()
    assert Path(result.path).is_file()
    assert np.all(np.isfinite(result.values))
    record_assertions(
        tmp_path,
        channel="potential",
        quantity="charge",
        surface_points=len(result.values),
        values_finite=True,
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_tuning_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_gpu: None
) -> None:
    """Tuning channel with Ray + gpu4pyscf (parallel=True)."""
    monkeypatch.chdir(tmp_path)
    surf_path, _ = _prepare_methane_surface(tmp_path)
    TuningInput.from_config(
        molecule="methane.xyz",
        surface_file=str(surf_path),
        properties=_GPU_PROPS,
        basis_set=_GPU_BASIS,
        calc_type="separate",
        parallel=True,
        num_procs=1,
    ).run()
    results_dir = latest_results_dir(tmp_path)
    summary = results_dir / "methane_tuning_summary.csv"
    assert summary.is_file()
    for prop in _GPU_PROPS:
        assert (results_dir / f"methane_{prop}.mol2").is_file()
    record_assertions(
        tmp_path,
        channel="tuning",
        parallel=True,
        properties=list(_GPU_PROPS),
        results_dir=str(results_dir),
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_coupled_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_gpu: None
) -> None:
    """Coupled channel: APBS Gauss charges → parallel GPU tuning."""
    monkeypatch.chdir(tmp_path)
    surf_path, _ = _prepare_methane_surface(tmp_path)
    result = CoupledInput.from_config(
        molecule="methane.xyz",
        surface_file=str(surf_path),
        output_surf="coupled_gpu.surf",
        potential_method="apbs",
        potential_quantity="charge",
        properties=_GPU_PROPS,
        basis_set=_GPU_BASIS,
        calc_type="separate",
        parallel=True,
        num_procs=1,
    ).run()
    assert Path(result.potential.path).is_file()
    assert result.tuning.results_dir
    results_dir = Path(result.tuning.results_dir)
    assert results_dir.is_dir()
    for prop in _GPU_PROPS:
        assert (results_dir / f"methane_{prop}.mol2").is_file()
    record_assertions(
        tmp_path,
        channel="coupled",
        parallel=True,
        potential_quantity="charge",
        properties=list(_GPU_PROPS),
        results_dir=str(results_dir),
    )
