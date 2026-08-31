"""OR-Tools base solver config types."""

from dataclasses import dataclass

from dataclasses_json import Undefined
from dataclasses_json import dataclass_json

from umip.solver_config.base import SolverConfig


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(frozen=True)
class OrToolsSolverConfig(SolverConfig):
    """Base class for OR-Tools solver configuration objects."""
