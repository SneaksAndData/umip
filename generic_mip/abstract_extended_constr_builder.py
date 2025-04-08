from abc import abstractmethod
from typing import TypeVar, Generic
import numpy.typing as npt
from generic_mip import AbstractConstraintBuilder

DT = TypeVar("DT")  # Data Type
VT = TypeVar("VT")  # Variable type


class AbstractExtendedConstraintBuilder(AbstractConstraintBuilder, Generic[VT, DT]):
    """An extended Constraint has the responsibility of building one or more constraints."""

    @abstractmethod
    def get_variables(self, data: dict[str, DT]) -> npt.NDArray[npt.NDArray[VT]] | npt.NDArray[VT]:
        """
        Gets the variables for the constraint
        """

    @abstractmethod
    def get_coefficients(self, data: dict[str, DT]) -> npt.NDArray[npt.NDArray[float]] | npt.NDArray[float]:
        """
        Gets the coefficients for the constraint
        """

    @abstractmethod
    def get_right_hand_side(self, data: dict[str, DT]) -> npt.NDArray[float] | float:
        """
        Gets the right hand side for the constraint
        """
