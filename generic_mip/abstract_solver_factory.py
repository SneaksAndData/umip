"""Abstract definition of a solver factory."""
from abc import ABC, abstractmethod
from adapta.logs import SemanticLogger
from generic_mip.abstract_solver import AbstractOptimizationSolver


class AbstractOptimizationSolverFactory(ABC):
    """A generic definition of a solver factory."""

    def __init__(self, logger: SemanticLogger):
        """
        Initialize the solver factory.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def construct(self, **kwargs: any) -> AbstractOptimizationSolver:
        """
        Given the arguments, construct an optimization solver.
        :param kwargs: The arguments to the construction.
        :return: The constructed optimization solver.
        """
