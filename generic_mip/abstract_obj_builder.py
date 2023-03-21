"""Abstract definition of an objective builder."""
from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic
from adapta.logs import SemanticLogger
from generic_mip.abstract_solver import AbstractOptimizationSolver

T = TypeVar('T')


class AbstractObjectiveBuilder(ABC, Generic[T]):
    """An objective builder has the responsibility of building one or more objective terms."""
    def __init__(self, logger: SemanticLogger):
        """
        Initialize the objective builder.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, data: Dict[str, T]) -> None:
        """
        Builds the objective terms on the given model and the given data.

        :param solver: The solver to use to build the objective terms.
        :param data: The data (e.g. dataframes) providing variables and parameters for the objective terms.
        :return:
        """
