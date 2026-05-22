import sys
import pytest
from adapta.logs import SemanticLogger, LoggerInterface
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel

from generic_mip.enums import SolverType
from generic_mip.solver.cplex import CplexSolver
from generic_mip.solver.gurobi import GurobiSolver
from generic_mip.solver.highs import HighsSolver
from generic_mip.solver.scip import ScipSolver
from generic_mip.solver.or_tools import OrToolsSolver, OrToolsSolverEngine
from generic_mip.solver.local_solver import LocalSolver
from generic_mip.solver_factory import SolverFactory


@pytest.fixture(scope="session")
def logger() -> LoggerInterface:
    logger = SemanticLogger().add_log_source(
        log_source_name="auto-replenishment-crystal-orchestrator",
        min_log_level=LogLevel.INFO,
        log_handlers=[SafeStreamHandler(sys.stdout)],
        is_default=True,
    )
    return logger


@pytest.fixture(scope="function")
def solver(logger, request):
    if request.param == "OrTools":
        return SolverFactory(logger=logger).construct(solver_type=SolverType.ORTOOLS_SCIP)
    elif request.param == "Gurobi":
        return SolverFactory(logger=logger).construct(solver_type=SolverType.GUROBI)
    elif request.param == "LocalSolver":
        raise NotImplementedError("LocalSolver is not implemented yet.")
    elif request.param == "Highs":
        return SolverFactory(logger=logger).construct(solver_type=SolverType.HIGHS)
    elif request.param == "Cplex":
        return SolverFactory(logger=logger).construct(solver_type=SolverType.CPLEX)
    elif request.param == "Scip":
        return SolverFactory(logger=logger).construct(solver_type=SolverType.SCIP)
