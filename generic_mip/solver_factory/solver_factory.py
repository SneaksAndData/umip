# pylint: disable=import-outside-toplevel
"""Solver factory for obtaining a implementation of a solver."""
from adapta.logs import LoggerInterface
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.enums.solver_type import SolverType
from generic_mip.solver_config import CplexSolverConfig
from generic_mip.solver_config import GurobiSolverConfig
from generic_mip.solver_config import HighsSolverConfig
from generic_mip.solver_config import OrToolsCbcSolverConfig
from generic_mip.solver_config import OrToolsCplexSolverConfig
from generic_mip.solver_config import OrToolsGlpkSolverConfig
from generic_mip.solver_config import OrToolsGurobiSolverConfig
from generic_mip.solver_config import OrToolsScipSolverConfig
from generic_mip.solver_config import OrToolsXpressSolverConfig
from generic_mip.solver_config import ScipSolverConfig
from generic_mip.solver_config import SolverConfig

ORTOOLS_SOLVERS = [
    SolverType.ORTOOLS_SCIP,
    SolverType.ORTOOLS_CPLEX,
    SolverType.ORTOOLS_GUROBI,
    SolverType.ORTOOLS_XPRESS,
    SolverType.ORTOOLS_CBC,
    SolverType.ORTOOLS_GLPK,
]

SOLVER_CONFIG_TYPE_BY_SOLVER_TYPE: dict[SolverType, type[SolverConfig]] = {
    SolverType.CPLEX: CplexSolverConfig,
    SolverType.GUROBI: GurobiSolverConfig,
    SolverType.HIGHS: HighsSolverConfig,
    SolverType.SCIP: ScipSolverConfig,
    SolverType.ORTOOLS_SCIP: OrToolsScipSolverConfig,
    SolverType.ORTOOLS_CBC: OrToolsCbcSolverConfig,
    SolverType.ORTOOLS_CPLEX: OrToolsCplexSolverConfig,
    SolverType.ORTOOLS_GLPK: OrToolsGlpkSolverConfig,
    SolverType.ORTOOLS_GUROBI: OrToolsGurobiSolverConfig,
    SolverType.ORTOOLS_XPRESS: OrToolsXpressSolverConfig,
}


class SolverFactory:
    """
    Solver factory for obtaining an implementation of a solver
    """

    def __init__(self, logger: LoggerInterface):
        self._logger = logger

    def construct(
        self, solver_type: SolverType, solver_config: SolverConfig | None = None
    ) -> AbstractOptimizationSolver:
        """
        Construct a solver instance based on the specified solver type.

        :param solver_type: The type of solver to construct.
        :param solver_config: Optional typed configuration object for the solver type.
        :return: Constructed solver implementation.
        :raises ValueError: If the solver type or solver config is invalid.
        """
        if not isinstance(solver_type, SolverType):
            raise ValueError(
                f"Unknown solver type: {solver_type}. " f"Supported types: {[solver.value for solver in SolverType]}"
            )

        self._validate_solver_config(solver_type=solver_type, solver_config=solver_config)

        self._logger.debug(
            template="Constructing {solver} solver with config: {config}",
            solver=solver_type.value,
            config=str(solver_config) if solver_config else "None",
        )

        if solver_type == SolverType.CPLEX:
            return self._get_cplex_solver(logger=self._logger)
        if solver_type == SolverType.GUROBI:
            return self._get_gurobi_solver(logger=self._logger)
        if solver_type == SolverType.HIGHS:
            return self._get_highs_solver(
                logger=self._logger,
                solver_config=solver_config,
            )
        if solver_type == SolverType.SCIP:
            return self._get_scip_solver(logger=self._logger)
        if solver_type in ORTOOLS_SOLVERS:
            return self._get_ortools_solver(
                solver_type=solver_type,
                solver_config=solver_config,
                logger=self._logger,
            )
        raise RuntimeError(f'Failed to construct solver for type "{solver_type}". ')

    @staticmethod
    def _validate_solver_config(solver_type: SolverType, solver_config: SolverConfig | None) -> None:
        """
        Validate that the provided solver config matches the requested solver type.

        :param solver_type: The requested solver type.
        :param solver_config: Optional typed solver configuration object.
        :return: ``None``.
        :raises ValueError: If the provided config type does not match solver type.
        """
        if solver_config is None:
            return

        expected_config_type = SOLVER_CONFIG_TYPE_BY_SOLVER_TYPE[solver_type]
        if not isinstance(solver_config, expected_config_type):
            raise ValueError(
                f"Invalid solver config for solver type {solver_type.value}. "
                f"Expected {expected_config_type.__name__}, got {type(solver_config).__name__}."
            )

    @staticmethod
    def _get_cplex_solver(logger: LoggerInterface) -> AbstractOptimizationSolver:
        try:
            from generic_mip.solver.cplex import CplexSolver

            return CplexSolver(logger=logger)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("CPLEX solver unavailable. Install the 'docplex' extra in generic-mip.") from exc

    @staticmethod
    def _get_gurobi_solver(logger: LoggerInterface) -> AbstractOptimizationSolver:
        try:
            from generic_mip.solver.gurobi import GurobiSolver

            return GurobiSolver(logger=logger)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("Gurobi solver unavailable. Install the 'gurobi' extra in generic-mip.") from exc

    @staticmethod
    def _get_highs_solver(
        logger: LoggerInterface,
        solver_config: SolverConfig | None,
    ) -> AbstractOptimizationSolver:
        try:
            from generic_mip.solver.highs import HighsSolver

            solver = HighsSolver(logger=logger)
            if isinstance(solver_config, HighsSolverConfig):
                solver.set_solver_setting(setting=solver_config)
            return solver
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("HiGHS solver unavailable. Install the 'highs' extra in generic-mip.") from exc

    @staticmethod
    def _get_scip_solver(logger: LoggerInterface) -> AbstractOptimizationSolver:
        try:
            from generic_mip.solver.scip import ScipSolver

            return ScipSolver(logger=logger)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("SCIP solver unavailable. Install the 'pyscipopt' extra in generic-mip.") from exc

    @staticmethod
    def _get_ortools_solver(
        solver_type: SolverType,
        solver_config: SolverConfig | None,
        logger: LoggerInterface,
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
            if solver_config is not None:
                solver.set_solver_setting(setting=solver_config)

            return solver
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("OR-Tools solver unavailable. Install the 'ortools' extra in generic-mip.") from exc
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid OR-Tools solver type: {solver_type}. "
                f"Supported types: {[solver.value for solver in ORTOOLS_SOLVERS]}"
            ) from exc
