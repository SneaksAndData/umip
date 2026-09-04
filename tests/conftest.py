#  Copyright (c) 2026. ECCO Data & AI and other project contributors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import sys

import pytest
from adapta.logs import LoggerInterface, SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel

from umip.enums import SolverType
from umip.solver_factory import SolverFactory


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
