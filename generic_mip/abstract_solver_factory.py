"""Abstract definition of a solver factory."""
from abc import ABC
from adapta.logs import LoggerInterface
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.enums.solver_type import SolverType
from generic_mip.solver.cplex import CplexSolver
from generic_mip.solver.gurobi import GurobiSolver
from generic_mip.solver.highs import HighsSolver
from generic_mip.solver.or_tools import OrToolsSolver, OrToolsSolverEngine
from generic_mip.solver.scip import ScipSolver


class AbstractOptimizationSolverFactory(ABC):
    """A generic definition of a solver factory."""

    def __init__(self, logger: LoggerInterface):
        """
        Initialize the solver factory.
        :param logger: The logger to use.
        """
        self._logger = logger

    # pylint: disable=unused-argument
    def construct(self, solver_type: SolverType, **kwargs: any) -> AbstractOptimizationSolver:
        """
        Given the arguments, construct an optimization solver.
        :param solver_type: The type of the solver.
        :param kwargs: The arguments to the construction.
        :return: The constructed optimization solver.
        """
        if solver_type == SolverType.CPLEX:
            return CplexSolver(logger=self._logger)

        if solver_type == SolverType.GUROBI:
            return GurobiSolver(logger=self._logger)

        if solver_type == SolverType.HIGHS:
            return HighsSolver(logger=self._logger)

        if solver_type == SolverType.SCIP:
            return ScipSolver(logger=self._logger)

        return OrToolsSolver(solver_engine=self.__to_ortools_solver_engine(solver_type), logger=self._logger)

    @staticmethod
    def __to_ortools_solver_engine(solver_type: SolverType) -> OrToolsSolverEngine:
        if solver_type == SolverType.ORTOOLS_SCIP:
            return OrToolsSolverEngine.SCIP
        if solver_type == SolverType.ORTOOLS_CPLEX:
            return OrToolsSolverEngine.CPLEX
        if solver_type == SolverType.ORTOOLS_GUROBI:
            return OrToolsSolverEngine.GUROBI
        if solver_type == SolverType.ORTOOLS_XPRESS:
            return OrToolsSolverEngine.XPRESS
        if solver_type == SolverType.ORTOOLS_CBC:
            return OrToolsSolverEngine.CBC
        if solver_type == SolverType.ORTOOLS_GLPK:
            return OrToolsSolverEngine.GLPK

        raise ValueError(f"Unknown solver: {solver_type}")
