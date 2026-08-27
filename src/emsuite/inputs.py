"""Immutable channel inputs. ``.in`` files and kwargs build these; runners consume them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

from emsuite.config import resolve_config
from emsuite.config.schemas import (
    validate_coupled_params,
    validate_potential_params,
    validate_surface_params,
    validate_tuning_params,
)
from emsuite.results import CoupledResult, PotentialResult, SurfaceResult, TuningResult

_TUNING_DEFAULTS = {
    "molecule": None,
    "surface_file": None,
    "properties": ("all",),
    "basis_set": "6-31G*",
    "method": "dft",
    "functional": "b3lyp",
    "charge": 0,
    "spin": 0,
    "solvent": None,
    "calc_type": "separate",
    "parallel": True,
    "num_procs": None,
    "state_of_interest": 2,
    "triplet": False,
}


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

    @classmethod
    def from_mapping(cls, params: dict[str, Any]) -> Self:
        from emsuite.surface.runner import SURFACE_DEFAULTS

        validated = validate_surface_params({**SURFACE_DEFAULTS, **params})
        return cls(**_take(cls, validated))

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        from emsuite.surface.runner import parse_surface_input

        return cls.from_mapping(parse_surface_input(str(path)))

    @classmethod
    def from_config(cls, config: str | Path | dict | None = None, **overrides: Any) -> Self:
        from emsuite.surface.runner import SURFACE_DEFAULTS

        return cls.from_mapping(resolve_config(config, overrides, defaults=SURFACE_DEFAULTS))

    @classmethod
    def from_any(cls, config: SurfaceInput | str | Path | dict) -> Self:
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_mapping(config)
        return cls.from_file(config)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def run(self) -> SurfaceResult:
        from emsuite.surface.runner import run_surface_calculation

        path = run_surface_calculation(self)
        return SurfaceResult.from_surf(path)


@dataclass(frozen=True)
class PotentialInput:
    molecule: str
    surface_file: str | None = None
    output_surf: str = "potential.surf"
    surface_density: float = 0.5
    surface_scale: float = 1.0
    method: str = "apbs"
    quantity: str = "potential"
    pdie: float = 2.0
    sdie: float = 78.54
    charge: int = 0
    spin: int = 0
    ligand: str | None = None
    protein: str | None = None
    ligand_atoms: str = "present"

    @classmethod
    def from_mapping(cls, params: dict[str, Any]) -> Self:
        from emsuite.potential.config_io import POTENTIAL_DEFAULTS

        validated = validate_potential_params({**POTENTIAL_DEFAULTS, **params})
        return cls(**_take(cls, validated))

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        from emsuite.potential.config_io import parse_potential_input

        return cls.from_mapping(parse_potential_input(str(path)))

    @classmethod
    def from_config(cls, config: str | Path | dict | None = None, **overrides: Any) -> Self:
        from emsuite.potential.config_io import POTENTIAL_DEFAULTS

        return cls.from_mapping(resolve_config(config, overrides, defaults=POTENTIAL_DEFAULTS))

    @classmethod
    def from_any(cls, config: PotentialInput | str | Path | dict) -> Self:
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_mapping(config)
        return cls.from_file(config)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def run(self) -> PotentialResult:
        from emsuite.potential.runner import run_potential_calculation

        path = run_potential_calculation(self)
        return PotentialResult.from_surf(path, quantity=self.quantity)


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

    @classmethod
    def from_mapping(cls, params: dict[str, Any]) -> Self:
        validated = validate_tuning_params({**_TUNING_DEFAULTS, **params})
        return cls(**_take(cls, validated))

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        from emsuite.tuning.config_io import get_tuning_parameters

        return cls.from_mapping(get_tuning_parameters(str(path)))

    @classmethod
    def from_config(cls, config: str | Path | dict | None = None, **overrides: Any) -> Self:
        return cls.from_mapping(resolve_config(config, overrides, defaults=_TUNING_DEFAULTS))

    @classmethod
    def from_any(cls, config: TuningInput | str | Path | dict) -> Self:
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_mapping(config)
        return cls.from_file(config)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def run(self) -> TuningResult:
        from emsuite.tuning.runner import main as run_tuning

        results_dir = run_tuning(self)
        return TuningResult(results_dir=str(results_dir) if results_dir else None)


@dataclass(frozen=True)
class CoupledInput:
    molecule: str
    surface_file: str | None = None
    output_surf: str = "coupled.surf"
    surface_density: float = 0.5
    surface_scale: float = 1.0
    potential_method: str = "apbs"
    potential_quantity: str = "charge"
    pdie: float = 2.0
    sdie: float = 78.54
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
    protein: str | None = None
    ligand_atoms: str = "present"

    @classmethod
    def from_mapping(cls, params: dict[str, Any]) -> Self:
        from emsuite.coupled.runner import COUPLED_DEFAULTS

        validated = validate_coupled_params({**COUPLED_DEFAULTS, **params})
        return cls(**_take(cls, validated))

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        from emsuite.coupled.runner import parse_coupled_input

        return cls.from_mapping(parse_coupled_input(str(path)))

    @classmethod
    def from_config(cls, config: str | Path | dict | None = None, **overrides: Any) -> Self:
        from emsuite.coupled.runner import COUPLED_DEFAULTS

        return cls.from_mapping(resolve_config(config, overrides, defaults=COUPLED_DEFAULTS))

    @classmethod
    def from_any(cls, config: CoupledInput | str | Path | dict) -> Self:
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_mapping(config)
        return cls.from_file(config)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def run(self) -> CoupledResult:
        potential = PotentialInput(
            molecule=self.molecule,
            surface_file=self.surface_file,
            output_surf=self.output_surf,
            surface_density=self.surface_density,
            surface_scale=self.surface_scale,
            method=self.potential_method,
            quantity=self.potential_quantity,
            pdie=self.pdie,
            sdie=self.sdie,
            charge=self.charge,
            spin=self.spin,
            ligand=self.ligand or self.molecule,
            protein=self.protein,
            ligand_atoms=self.ligand_atoms,
        ).run()
        surface_file = potential.path or potential.to_surf(self.output_surf)
        tuning = TuningInput(
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
        ).run()
        return CoupledResult(potential=potential, tuning=tuning)
