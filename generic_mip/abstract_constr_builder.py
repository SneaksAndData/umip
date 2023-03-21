"""Abstract definition of a constraint builder."""
from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic
from adapta.logs import SemanticLogger
from generic_mip.abstract_solver import AbstractOptimizationSolver

T = TypeVar('T')


class AbstractConstraintBuilder(ABC, Generic[T]):
    """A constraint builder has the responsibility of building one or more constraints."""
    def __init__(self, logger: SemanticLogger):
        """
        Initialize the constraint builder.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, data: Dict[str, T]) -> None:
        """
        Builds the constraints on the given model and the given data.

        :param solver: The solver to use to build the constraints.
        :param data: The data (e.g. dataframes) providing variables and parameters for the constraints.
        :return:
        """
