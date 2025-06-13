# pylint: disable=import-outside-toplevel
"""Solver factory for obtaining a implementation of a solver."""
from adapta.logs import LoggerInterface
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.enums.solver_type import SolverType

ORTOOLS_SOLVERS = [
    SolverType.ORTOOLS_SCIP,
    SolverType.ORTOOLS_CPLEX,
    SolverType.ORTOOLS_GUROBI,
    SolverType.ORTOOLS_XPRESS,
    SolverType.ORTOOLS_CBC,
    SolverType.ORTOOLS_GLPK,
]


class SolverFactory:
    """
    Solver factory for obtaining a implementation of a solver
    """

    def __init__(self, logger: LoggerInterface):
        self._logger = logger

    def construct(self, solver_type: SolverType, solver_settings: list[str] = None) -> AbstractOptimizationSolver:
        """
        Construct a solver instance based on the specified solver type.

        :param solver_type: The type of solver to construct.
        :param solver_settings: Optional settings for the solver. Only OR-Tools solvers support settings.
        """
        if not isinstance(solver_type, SolverType):
            raise ValueError(
                f"Unknown solver type: {solver_type}. " f"Supported types: {[solver.value for solver in SolverType]}"
            )

        self._logger.debug(
            template="Constructing {solver} solver with settings: \n {settings}",
            solver=solver_type.value,
            settings=str(solver_settings) if solver_settings else "None",
        )

        if solver_type == SolverType.CPLEX:
            return self._get_cplex_solver(solver_settings=solver_settings, logger=self._logger)
        if solver_type == SolverType.GUROBI:
            return self._get_gurobi_solver(solver_settings=solver_settings, logger=self._logger)
        if solver_type == SolverType.HIGHS:
            return self._get_highs_solver(solver_settings=solver_settings, logger=self._logger)
        if solver_type == SolverType.SCIP:
            return self._get_scip_solver(solver_settings=solver_settings, logger=self._logger)
        if solver_type in ORTOOLS_SOLVERS:
            return self._get_ortools_solver(solver_type, solver_settings=solver_settings, logger=self._logger)
        raise RuntimeError(f'Failed to construct solver for type "{solver_type}". ')

    @staticmethod
    def _get_cplex_solver(solver_settings: list[str], logger: LoggerInterface) -> AbstractOptimizationSolver:
        try:
            from generic_mip.solver.cplex import CplexSolver

            if solver_settings:
                logger.warning(
                    "CPLEX solver does not support solver settings. Ignoring provided settings: {settings}",
                    settings=solver_settings,
                )

            return CplexSolver(logger=logger)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("CPLEX solver unavailable. Install the 'docplex' extra in generic-mip.") from exc

    @staticmethod
    def _get_gurobi_solver(solver_settings: list[str], logger: LoggerInterface) -> AbstractOptimizationSolver:
        try:
            from generic_mip.solver.gurobi import GurobiSolver

            if solver_settings:
                logger.warning(
                    "Gurobi solver does not support solver settings. Ignoring provided settings: {settings}",
                    settings=solver_settings,
                )

            return GurobiSolver(logger=logger)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("Gurobi solver unavailable. Install the 'gurobi' extra in generic-mip.") from exc

    @staticmethod
    def _get_highs_solver(solver_settings: list[str], logger: LoggerInterface) -> AbstractOptimizationSolver:
        try:
            from generic_mip.solver.highs import HighsSolver

            if solver_settings:
                logger.warning(
                    "HiGHS solver does not support solver settings. Ignoring provided settings: {settings}",
                    settings=solver_settings,
                )

            return HighsSolver(logger=logger)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("HiGHS solver unavailable. Install the 'highs' extra in generic-mip.") from exc

    @staticmethod
    def _get_scip_solver(solver_settings: list[str], logger: LoggerInterface) -> AbstractOptimizationSolver:
        try:
            from generic_mip.solver.scip import ScipSolver

            if solver_settings:
                logger.warning(
                    "SCIP solver does not support solver settings. Ignoring provided settings: {settings}",
                    settings=solver_settings,
                )

            return ScipSolver(logger=logger)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("SCIP solver unavailable. Install the 'pyscipopt' extra in generic-mip.") from exc

    @staticmethod
    def _get_ortools_solver(
        solver_type: SolverType, solver_settings: list[str], logger: LoggerInterface
    ) -> AbstractOptimizationSolver:
        try:
            from generic_mip.solver.or_tools import OrToolsSolver, OrToolsSolverEngine

            ortools_solver_mapping = {
                SolverType.ORTOOLS_SCIP: OrToolsSolverEngine.SCIP,
                SolverType.ORTOOLS_CPLEX: OrToolsSolverEngine.CPLEX,
                SolverType.ORTOOLS_GUROBI: OrToolsSolverEngine.GUROBI,
                SolverType.ORTOOLS_XPRESS: OrToolsSolverEngine.XPRESS,
                SolverType.ORTOOLS_CBC: OrToolsSolverEngine.CBC,
                SolverType.ORTOOLS_GLPK: OrToolsSolverEngine.GLPK,
            }
            ortools_engine = ortools_solver_mapping[solver_type]
            solver = OrToolsSolver(solver_engine=ortools_engine, logger=logger)
            if solver_settings:
                for setting in solver_settings:
                    solver.set_solver_setting(setting=setting)
            return solver
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("OR-Tools solver unavailable. Install the 'ortools' extra in generic-mip.") from exc
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid OR-Tools solver type: {solver_type}. "
                f"Supported types: {[solver.value for solver in ORTOOLS_SOLVERS]}"
            ) from exc
