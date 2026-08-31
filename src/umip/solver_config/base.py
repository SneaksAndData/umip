"""Base solver config types."""

from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin
from dataclasses_json import Undefined
from dataclasses_json import dataclass_json


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(frozen=True)
class SolverConfig(DataClassJsonMixin):
    """Base class for solver specific configuration objects."""
