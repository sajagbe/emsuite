"""EMSuite — Electrostatic Map Suite."""

__version__ = "1.6.0"

from emsuite.inputs import CoupledInput, PotentialInput, SurfaceInput, TuningInput
from emsuite.results import CoupledResult, PotentialResult, SurfaceResult, TuningResult

__all__ = [
    "CoupledInput",
    "CoupledResult",
    "PotentialInput",
    "PotentialResult",
    "SurfaceInput",
    "SurfaceResult",
    "TuningInput",
    "TuningResult",
    "__version__",
]
