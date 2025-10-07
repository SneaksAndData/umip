"""Abstract definition of an objective builder."""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from adapta.logs import LoggerInterface
from generic_mip.abstract_solver import AbstractOptimizationSolver

T = TypeVar("T")


class AbstractObjectiveBuilder(ABC, Generic[T]):
    """An objective builder has the responsibility of building one or more objective terms."""

    def __init__(self, logger: LoggerInterface):
        """
        Initialize the objective builder.
        :param logger: The logger to use.
        """
        self.objective_name = self.__class__.__name__
        self._logger = logger
        self._analytics_granularity_functions: dict[str, callable] = {}

    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, T]) -> None:
        """
        Builds the objective terms on the given model and the given data.

        :param solver: The solver to use to build the objective terms.
        :param data: The data (e.g. dataframes) providing variables and parameters for the objective terms.
        :return:
        """

    def add_analytics_granularity(self, granularity_name: str, analytics_calculator: callable) -> None:
        """
        Register an analytics calculator for a specific granularity.

        :param granularity_name: The name of the granularity (e.g., "location")
        :param analytics_calculator: The function that calculates analytics for this granularity
        """
        if granularity_name in self._analytics_granularity_functions:
            raise ValueError(f"Analytics granularity '{granularity_name}' is already registered.")

        self._analytics_granularity_functions[granularity_name] = analytics_calculator

    def get_supported_analytics_granularities(self) -> list[str]:
        """Returns the list of supported granularity names for implementation of builder."""
        return list(self._analytics_granularity_functions.keys())

    def get_analytics(self, granularity: str, output_data: any) -> any:
        """
        Retrieve analytics for the specified granularity.

        :param granularity: The granularity level to compute analytics for
        :param output_data: The output data needed for calculation (e.g., model output data)
        :return: The computed analytics
        """
        if granularity not in self._analytics_granularity_functions:
            raise ValueError(
                f"Unsupported analytics granularity: {granularity}. "
                f"Supported: {self.get_supported_analytics_granularities()}"
            )

        return self._analytics_granularity_functions[granularity](output_data)
