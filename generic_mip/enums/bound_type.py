"""Bound types."""
from enum import Enum


class BoundType(Enum):
    """Possible bound types."""

    LOWER = "lower"
    UPPER = "upper"
