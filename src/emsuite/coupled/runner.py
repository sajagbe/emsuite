"""One-run surface → environment charges → tuning pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from emsuite.banner import with_banner
from emsuite.config import parse_assignments, parse_config_file
from emsuite.config.schemas import reject_removed_coupled_keys, validate_coupled_params


def parse_coupled_input(input_file: str) -> dict:
    from emsuite.inputs import CoupledInput, _channel_defaults

    path = Path(input_file)
    if path.exists():
        parsed = parse_assignments(path.read_text())
        reject_removed_coupled_keys(parsed)
        unknown = set(parsed) - set(_channel_defaults(CoupledInput))
        if unknown:
            raise ValueError(f"unknown coupled input key: {sorted(unknown)[0]!r}")
    params = parse_config_file(input_file, defaults=_channel_defaults(CoupledInput))
    return validate_coupled_params(params)


@with_banner
def run_coupled(spec):
    from emsuite.geometry import Geometry
    from emsuite.inputs import SurfaceInput
    from emsuite.results import CoupledResult, PotentialResult, SurfaceResult
    from emsuite.surface.io import load_surf, save_surf, surface_quantity
    from emsuite.tuning.geometry import check_surface_frame

    origin = Path.cwd()
    inputs = {"molecule": Path(spec.molecule).resolve(strict=True)}
    geometry = Geometry.from_xyz(inputs["molecule"])
    if not np.isfinite(geometry.coords).all():
        raise ValueError("molecule coordinates must be finite")
    if spec.potential_surf:
        inputs["charged_surface"] = Path(spec.potential_surf).resolve(strict=True)
        kind = surface_quantity(inputs["charged_surface"])
        if kind not in (None, "charge"):
            raise ValueError(
                f"precomputed surface contains {kind}, but tuning requires charge in electrons"
            )
        coords, _ = load_surf(inputs["charged_surface"])
        check_surface_frame(inputs["molecule"], coords, strict=True)
        if kind is None:
            print(
                "Legacy precomputed surface has no units metadata; treating its values as charges in electrons."
            )
    else:
        inputs["pdb"] = Path(spec.pdb).resolve(strict=True)
        if spec.surface_file:
            inputs["surface"] = Path(spec.surface_file).resolve(strict=True)
            coords, _ = load_surf(inputs["surface"])
            check_surface_frame(inputs["molecule"], coords, strict=True)

    if spec.output_dir:
        work = Path(spec.output_dir).resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        work = Path(tempfile.mkdtemp(prefix=f"coupled_{inputs['molecule'].stem}_", dir=origin))
    saved = {}
    records = {}
    for name, source in inputs.items():
        target = work / "inputs" / name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        saved[name] = target
        records[name] = dict(
            source=str(source),
            snapshot=str(target),
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )
    timings = {}
    manifest = dict(
        status="running",
        stage="surface",
        inputs=records,
        settings=asdict(spec),
        timings_seconds=timings,
        charge_units="e",
        run_dir=str(work),
    )

    def record():
        (work / "run.json").write_text(json.dumps(manifest, indent=2, default=str))

    record()
    print(f"Coupled Surface → Charges → Tuning: {work}")
    started = perf_counter()
    surface = None
    try:
        os.chdir(work)
        start = perf_counter()
        if spec.potential_surf:
            potential = PotentialResult.from_surf(saved["charged_surface"], quantity="charge")
            timings["surface_and_charge_reuse_s"] = perf_counter() - start
        else:
            surface_path = work / "surface" / "geometry.surf"
            surface_path.parent.mkdir()
            if spec.surface_file:
                coords, _ = load_surf(saved["surface"])
                save_surf(coords, 0.0, surface_path, quantity="geometry")
                surface = SurfaceResult.from_surf(surface_path)
            else:
                surface = SurfaceInput(
                    input_type="XYZ",
                    input_data=str(saved["molecule"]),
                    output_surf=str(surface_path),
                    surface_density=spec.surface_density,
                    surface_scale=spec.surface_scale,
                    surface_charge=0.0,
                    optimize=False,
                ).run()
                save_surf(surface.coords, surface.values, surface_path, quantity="geometry")
            check_surface_frame(saved["molecule"], surface.coords, strict=True)
            timings["surface_s"] = perf_counter() - start
            manifest.update(stage="potential", surface=str(surface_path))
            record()
            output = work / "potential" / spec.output_surf
            output.parent.mkdir(parents=True)
            potential_spec = replace(
                spec.to_potential_input(str(surface_path)),
                pdb=str(saved["pdb"]),
                output_surf=str(output),
            )
            start = perf_counter()
            potential = potential_spec.run()
            if potential.quantity != "charge":
                raise ValueError("potential stage must return charges for tuning")
            timings["potential_s"] = perf_counter() - start
        manifest.update(stage="tuning", charged_surface=potential.path)
        record()
        start = perf_counter()
        tuning = replace(
            spec.to_tuning_input(str(potential.path)), molecule=str(saved["molecule"])
        ).run()
        if tuning.results_dir:
            path = Path(tuning.results_dir)
            tuning = replace(tuning, results_dir=str(path if path.is_absolute() else work / path))
        timings["tuning_s"] = perf_counter() - start
        manifest.update(status="completed", stage="completed", tuning_results=tuning.results_dir)
        return CoupledResult(
            potential=potential, tuning=tuning, run_dir=str(work), surface=surface, timings=timings
        )
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        timings["total_s"] = perf_counter() - started
        os.chdir(origin)
        record()
        print(
            "Coupled stage timings (seconds): "
            + ", ".join(f"{key}={value:.3f}" for key, value in timings.items())
        )
