import sys
import pytest
from logging import StreamHandler
from adapta.logs import SemanticLogger, LoggerInterface
from adapta.logs.models import LogLevel

from generic_mip.solver.cplex import CplexSolver
from generic_mip.solver.gurobi import GurobiSolver
from generic_mip.solver.highs import HighsSolver
from generic_mip.solver.or_tools import OrToolsSolver, OrToolsSolverEngine
from generic_mip.solver.local_solver import LocalSolver


@pytest.fixture(scope="session")
def logger() -> LoggerInterface:
    logger = SemanticLogger().add_log_source(
        log_source_name="auto-replenishment-crystal-orchestrator",
        min_log_level=LogLevel.INFO,
        log_handlers=[StreamHandler(sys.stdout)],
        is_default=True,
    )
    return logger


@pytest.fixture
def solver(logger, request):
    if request.param == "OrTools":
        return OrToolsSolver(solver_engine=OrToolsSolverEngine.SCIP, logger=logger)
    elif request.param == "Gurobi":
        return GurobiSolver(logger=logger)
    elif request.param == "LocalSolver":
        return LocalSolver(logger=logger)
    elif request.param == "Highs":
        return HighsSolver(logger=logger)
    elif request.param == "Cplex":
        return CplexSolver(logger=logger)
