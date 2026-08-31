"""Constraint types."""
from enum import Enum


class ConstraintType(Enum):
    """Possible constraint types."""

    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    EQUAL = "equal"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
