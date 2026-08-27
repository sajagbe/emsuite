"""Engine backends."""

from __future__ import annotations

from .base import Engine
from .pyscf_engine import PySCFEngine


def get_engine(name: str = "pyscf") -> Engine:
    if name != "pyscf":
        raise ValueError(f"Unknown engine {name!r}; only 'pyscf' is available")
    return PySCFEngine()


__all__ = ["Engine", "PySCFEngine", "get_engine"]
