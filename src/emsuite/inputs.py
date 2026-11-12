"""Immutable channel inputs. ``.in`` files and constructors build these; ``.run()`` executes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from emsuite.config.schemas import (
    validate_coupled_params,
    validate_surface_params,
    validate_tuning_params,
)
from emsuite.potential.spec import DEFAULT_FORCEFIELD, PotentialSpec
from emsuite.results import CoupledResult, PotentialResult, SurfaceResult, TuningResult


def _channel_defaults(cls: type) -> dict[str, Any]:
    """Defaults for ``.in`` parsing: dataclass field defaults, else ``None`` for required fields."""
    defaults: dict[str, Any] = {}
    for item in fields(cls):
        if item.default is not MISSING:
            defaults[item.name] = item.default
        elif item.default_factory is not MISSING:  # type: ignore[unreachable]
            defaults[item.name] = item.default_factory()
        else:
            defaults[item.name] = None
    return defaults


def _take(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    kwargs: dict[str, Any] = {}
    for name in names:
        if name not in params:
            continue
        value = params[name]
        if name == "properties" and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return kwargs


def _to_dict(obj: Any) -> dict[str, Any]:
    data = asdict(obj)
    if isinstance(data.get("properties"), tuple):
        data["properties"] = list(data["properties"])
    return data


def _finalize(
    instance: Any,
    *,
    validate: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Validate and write normalized fields onto a frozen input (defaults live on the dataclass)."""
    validated = validate({**_channel_defaults(type(instance)), **_to_dict(instance)})
    for name, value in _take(type(instance), validated).items():
        object.__setattr__(instance, name, value)


@dataclass(frozen=True)
class SurfaceInput:
    input_type: str
    input_data: str
    output_surf: str = "surface.surf"
    optimized_xyz: str | None = None
    surface_density: float = 1.0
    surface_scale: float = 1.0
    surface_type: str = "homogenous"
    surface_charge: float = 0.10
    optimize: bool | None = None
    optimize_method: str = "mmff"
    method: str = "dft"
    basis_set: str = "6-31G*"
    functional: str = "b3lyp"
    solvent: str | None = None
    charge: int = 0
    spin: int = 0

    def __post_init__(self) -> None:
        _finalize(self, validate=validate_surface_params)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        from emsuite.surface.runner import parse_surface_input

        return cls(**_take(cls, parse_surface_input(str(path))))

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def run(self) -> SurfaceResult:
        from emsuite.surface.runner import _run_surface

        path = _run_surface(self)
        return SurfaceResult.from_surf(path)


# The potential channel's input: one file names the PDB, the .surf and every setting.
PotentialInput = PotentialSpec


@dataclass(frozen=True)
class TuningInput:
    molecule: str
    surface_file: str
    properties: tuple[str, ...] = ("all",)
    basis_set: str = "6-31G*"
    method: str = "dft"
    functional: str = "b3lyp"
    charge: int = 0
    spin: int = 0
    solvent: str | None = None
    calc_type: str = "separate"
    parallel: bool = True
    num_procs: int | None = None
    state_of_interest: int = 2
    triplet: bool = False
    resume: bool = True

    def __post_init__(self) -> None:
        _finalize(self, validate=validate_tuning_params)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        from emsuite.tuning.config_io import parse_tuning_input

        return cls(**_take(cls, parse_tuning_input(str(path))))

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def run(self) -> TuningResult:
        from emsuite.tuning.runner import _run_tuning

        results_dir = _run_tuning(self)
        return TuningResult(results_dir=str(results_dir) if results_dir else None)


@dataclass(frozen=True)
class CoupledInput:
    molecule: str
    surface_file: str | None = None
    pdb: str | None = None
    output_surf: str = "coupled.surf"
    potential_method: str = "apbs"
    potential_quantity: str = "charge"
    pdie: float = 2.0
    sdie: float = 78.54
    grid_spacing: float | None = None
    forcefield: tuple[str, ...] = DEFAULT_FORCEFIELD
    add_hydrogens: bool = False
    ph: float = 7.0
    unknown_charge: dict[str, int] = field(default_factory=dict)
    unknown_spin: dict[str, int] = field(default_factory=dict)
    unknown_method: str = "hf"
    unknown_basis: str = "6-31g*"
    resp_copies: str = "each"
    off: tuple[Any, ...] = ()
    removed: tuple[Any, ...] = ()
    properties: tuple[str, ...] = ("homo", "lumo", "gap")
    basis_set: str = "6-31G*"
    method: str = "dft"
    functional: str = "b3lyp"
    charge: int = 0
    spin: int = 0
    solvent: str | None = None
    calc_type: str = "separate"
    parallel: bool = False
    state_of_interest: int = 2
    triplet: bool = False
    num_procs: int | None = None
    ligand: str | None = None
    potential_surf: str | None = None

    def __post_init__(self) -> None:
        params = {**_channel_defaults(type(self)), **self.to_dict()}
        validated = validate_coupled_params(params)
        for name, value in _take(type(self), validated).items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        from emsuite.coupled.runner import parse_coupled_input

        return cls(**_take(cls, parse_coupled_input(str(path))))

    def to_dict(self) -> dict[str, Any]:
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        if isinstance(data.get("properties"), tuple):
            data["properties"] = list(data["properties"])
        return data

    def to_potential_input(self) -> PotentialInput:
        """Map coupled fields onto a PotentialInput (skipped when ``potential_surf`` is set)."""
        from emsuite.config.schemas import potential_spec_kwargs

        return PotentialSpec(**potential_spec_kwargs(self.to_dict()))

    def to_tuning_input(self, surface_file: str) -> TuningInput:
        """Map coupled fields onto a TuningInput for the given surface path."""
        return TuningInput(
            molecule=self.molecule,
            surface_file=surface_file,
            properties=self.properties,
            basis_set=self.basis_set,
            method=self.method,
            functional=self.functional,
            charge=self.charge,
            spin=self.spin,
            solvent=self.solvent,
            calc_type=self.calc_type,
            parallel=self.parallel,
            num_procs=self.num_procs,
            state_of_interest=self.state_of_interest,
            triplet=self.triplet,
        )

    def run(self) -> CoupledResult:
        print("\n" + "=" * 60)
        print("           Coupled Potential → Tuning Pipeline")
        print("=" * 60 + "\n")

        if self.potential_surf:
            potential = PotentialResult.from_surf(
                self.potential_surf, quantity=self.potential_quantity
            )
        else:
            potential = self.to_potential_input().run()
        surface_file = potential.path or potential.to_surf(self.output_surf)
        tuning = self.to_tuning_input(surface_file).run()

        print("\nCoupled calculation complete.")
        print("=" * 60 + "\n")
        return CoupledResult(potential=potential, tuning=tuning)
