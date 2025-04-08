from abc import abstractmethod
from typing import TypeVar, Generic
import numpy.typing as npt
from generic_mip import AbstractObjectiveBuilder

DT = TypeVar("DT")  # Data Type
VT = TypeVar("VT")  # Variable type


class AbstractExtendedObjectiveBuilder(AbstractObjectiveBuilder, Generic[VT, DT]):
    """An extended objective builder has the responsibility of building one or more objective terms"""

    @abstractmethod
    def get_variables(self, data: dict[str, DT]) -> npt.NDArray[npt.NDArray[VT]] | npt.NDArray[VT]:
        """
        Gets the variables for the objective
        """

    @abstractmethod
    def get_objective_terms(self, data: dict[str, DT]) -> npt.NDArray[float] | float:
        """
        Gets the objective terms for the objective.
        """
