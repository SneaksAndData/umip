"""Module for argument type enums."""
from enum import Enum


class DataFrameArgumentType(Enum):
    """Possible dataframe types."""

    POLARS = "polars"
    PANDAS = "pandas"


class BoundArgumentType(Enum):
    """Possible types for bound arguments."""

    FLOAT = "float"
    STRING = "string"
    NONE = "none"


class FilterColumnArgumentType(Enum):
    """Possible types for filter column arguments."""

    STRING = "string"
    NONE = "none"


class IndexColumnsArgumentType(Enum):
    """Possible types for index column arguments."""

    LIST_OF_STRINGS = "list_of_strings"
    NONE = "none"
