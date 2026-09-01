"""Base solver config types."""

from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(frozen=True)
class SolverConfig(DataClassJsonMixin):
    """Base class for solver specific configuration objects."""
