"""Abstract definition of an optimization model."""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any

T = TypeVar("T")


class AbstractOptimizationModel(ABC, Generic[T]):
    """A generic optimization model interface."""

    @abstractmethod
    def build(self, **input_data: T) -> None:
        """
        Builds the model using the given variable, constraint and objective builders.
        :param input_data: Input data to the variables, constraints and objectives.
        :return:
        """

    @abstractmethod
    def solve(self, time_limit: float | None = None, **kwargs: Any) -> Any:
        """
        Solves the model and returns the result of the optimization.
        :param time_limit: The time limit of the optimization in seconds.
        :param kwargs: Optional arguments to the optimization.
        :return: The result.
        """

    @abstractmethod
    def objective_value(self) -> float:
        """
        Get the objective value of the optimization.
        :return: The objective value.
        """

    @abstractmethod
    def get_gap(self) -> float:
        """
        Get the gap of the model.
        :return: The gap of the model.
        """
