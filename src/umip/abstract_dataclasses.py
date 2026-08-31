"""Module for abstract dataclasses"""

from abc import ABC
from dataclasses import dataclass


@dataclass
class AbstractInputData(ABC):
    """
    Dataclass for input data.
    """


@dataclass
class AbstractInternalData(ABC):
    """
    Dataclass for internal data.
    """


@dataclass
class AbstractOutputData(ABC):
    """
    Dataclass for output data.
    """
