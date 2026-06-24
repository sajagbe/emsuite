"""Engine backends."""

from __future__ import annotations

from .base import Engine
from .mlip_engine import MLIPEngine
from .pyscf_engine import PySCFEngine


def get_engine(name: str = "pyscf") -> Engine:
    engines: dict[str, Engine] = {
        "pyscf": PySCFEngine(),
        "mlip": MLIPEngine(),
        "tblite": _get_tblite_engine(),
    }
    if name not in engines:
        raise ValueError(f"Unknown engine {name!r}; choose from {list(engines)}")
    return engines[name]


def _get_tblite_engine() -> Engine:
    try:
        from .tblite_engine import TBLiteEngine

        return TBLiteEngine()
    except ImportError:
        return MLIPEngine()


__all__ = ["Engine", "MLIPEngine", "PySCFEngine", "get_engine"]
