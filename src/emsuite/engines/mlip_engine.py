"""MLIP engine — uses TBLite (GFN-xTB) when installed."""

from __future__ import annotations


class MLIPEngine:
    """Fast semi-empirical backend; requires optional ``mlip`` extra."""

    name = "mlip"

    def __init__(self) -> None:
        try:
            from .tblite_engine import TBLiteEngine

            self._backend = TBLiteEngine()
        except ImportError:
            self._backend = None

    def is_available(self) -> bool:
        return self._backend is not None and self._backend.is_available()

    def describe(self) -> str:
        if self.is_available():
            return self._backend.describe()
        return "MLIP/xTB backend unavailable. Install: uv sync --extra mlip"

    def optimize_geometry(self, xyz_path: str, **kwargs) -> str:
        if not self.is_available():
            raise NotImplementedError(self.describe())
        return self._backend.optimize_geometry(xyz_path, **kwargs)

    def single_point_energy(self, xyz_path: str, **kwargs) -> float:
        if not self.is_available():
            raise NotImplementedError(self.describe())
        return self._backend.single_point_energy(xyz_path, **kwargs)


__all__ = ["MLIPEngine"]
