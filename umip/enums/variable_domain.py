"""Variable domains."""
from enum import Enum


class VariableDomain(Enum):
    """Possible decision variable domains."""

    INTEGER = "integer"
    CONTINUOUS = "continuous"
    BINARY = "binary"
