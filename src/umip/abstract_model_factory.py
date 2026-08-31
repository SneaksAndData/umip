"""Abstract definition of an optimization model factory."""

from abc import ABC, abstractmethod
from typing import Any

from adapta.logs import LoggerInterface

from umip.abstract_solver import AbstractOptimizationSolver
from umip.abstract_mip import AbstractMipModel
from umip.enums import SolverType
from umip.solver_config import SolverConfig
from umip.solver_factory import SolverFactory


class AbstractMipModelFactory(ABC):
    """A generic MIP model factory."""

    def __init__(
        self,
        logger: LoggerInterface,
        solver_type: SolverType,
        solver_config: SolverConfig | None = None,
    ):
        """
        Initialize the model factory.
        :param logger: The logger to use.
        """
        self._logger = logger
        self._solver = self._get_solver(
            solver_type=solver_type, solver_config=solver_config
        )

    @abstractmethod
    def construct(
        self,
        **kwargs: Any,
    ) -> AbstractMipModel:
        """
        Given the arguments, construct an MIP model.
        :param kwargs: The arguments to the construction.
        :return: The constructed MIP model.
        """

    def _get_solver(
        self, solver_type: SolverType, solver_config: SolverConfig | None = None
    ) -> AbstractOptimizationSolver:
        """
        Get the solver instance based on the specified solver type and configuration.
        :param solver_type: The type of solver to construct.
        :param solver_config: Optional typed configuration object for the solver type.
        :return: solver.
        """
        return SolverFactory(logger=self._logger).construct(
            solver_type=solver_type, solver_config=solver_config
        )
