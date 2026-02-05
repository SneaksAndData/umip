"""Abstract definition of a constraint builder."""
from abc import ABC, abstractmethod
from adapta.logs import LoggerInterface
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.abstract_dataclasses import AbstractInternalData


class AbstractConstraintBuilder(ABC):
    """A constraint builder has the responsibility of building one or more constraints."""

    def __init__(self, logger: LoggerInterface):
        """
        Initialize the constraint builder.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, data: AbstractInternalData) -> None:
        """
        Builds the constraints on the given model and the given data.

        :param solver: The solver to use to build the constraints.
        :param data: The data (e.g. dataframes) providing variables and parameters for the constraints.
        :return:
        """
