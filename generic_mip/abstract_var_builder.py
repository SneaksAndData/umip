"""Abstract definition of a variable builder."""
from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic
from proteus.logs import ProteusLogger
from generic_mip.abstract_solver import AbstractOptimizationSolver

T = TypeVar('T')


class AbstractDecisionVariableBuilder(ABC, Generic[T]):
    """A variable builder has the responsibility of building one or more decision variables."""
    def __init__(self, logger: ProteusLogger):
        """
        Initialize the variable builder.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, data: Dict[str, T]) -> Dict[str, T]:
        """
        Builds the decision variables on the given model and the given data.

        :param solver: The solver to use to build the variables.
        :param data: The data (e.g. dataframes) providing parameters for the variables.
        :return: The dataframes decorated with the created decision variables.
        """

    @abstractmethod
    def unpack(self, solver: AbstractOptimizationSolver, data: Dict[str, T]) -> Dict[str, T]:
        """
        Unpacks the decision variables after optimization and inserts variable values in the dataframes.

        :param solver: The solver to get the variable values from.
        :param data: The data (e.g. dataframes) containing the variables.
        :return: The dataframes decorated with the values of the decision variables.
        """
