"""MLIP engine placeholder for future fast screening workflows."""


class MLIPEngine:
    """Stub — not yet implemented."""

    def optimize_geometry(self, xyz_path: str, **kwargs) -> str:
        raise NotImplementedError("MLIP engine is not yet implemented")

    def single_point_energy(self, xyz_path: str, **kwargs) -> float:
        raise NotImplementedError("MLIP engine is not yet implemented")
