import sys
import pytest
from logging import StreamHandler
from proteus.logs._base import ProteusLogger
from proteus.logs.models import LogLevel
from ortools.linear_solver import pywraplp
from generic_mip.solver import GurobiSolver, OrToolsSolver, OrToolsSolverEngine


@pytest.fixture(scope="session")
def logger():
    logger = ProteusLogger().add_log_source(
        log_source_name='auto-replenishment-crystal-orchestrator',
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
