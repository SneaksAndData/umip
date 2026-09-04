"""Tests for SolverFactory construction and solver config validation."""

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

import pytest

from umip import AbstractOptimizationSolver
from umip.enums.solver_type import SolverType
from umip.solver.cplex import CplexSolver
from umip.solver.highs import HighsSolver
from umip.solver.or_tools import OrToolsSolver
from umip.solver_config import GurobiSolverConfig
from umip.solver_factory import SolverFactory


@pytest.mark.parametrize(
    "solver_type",
    [
        SolverType.ORTOOLS_SCIP,
        SolverType.ORTOOLS_CBC,
        SolverType.CPLEX,
        SolverType.HIGHS,
    ],
)
def test_construct_solver_is_correct_type(solver_type, logger) -> None:
    # Arrange
    sut = SolverFactory(logger=logger)

    # Act
    solver = sut.construct(solver_type=solver_type)

    # Assert
    assert solver is not None
    assert __isinstance_of_solver(solver=solver, solver_type=solver_type)


def test_construct_unknown_solver_type_throws_exception(logger) -> None:
    # Arrange
    sut = SolverFactory(logger=logger)

    # Act & Assert
    with pytest.raises(TypeError):
        sut.construct(solver_type=100)


def test_construct_raises_for_incompatible_solver_config(logger) -> None:
    # Arrange
    sut = SolverFactory(logger=logger)

    # Act & Assert
    with pytest.raises(TypeError):
        sut.construct(solver_type=SolverType.ORTOOLS_SCIP, solver_config=GurobiSolverConfig())


def __isinstance_of_solver(solver: AbstractOptimizationSolver, solver_type: SolverType) -> bool:
    if solver_type == SolverType.CPLEX:
        return isinstance(solver, CplexSolver)

    if solver_type == SolverType.HIGHS:
        return isinstance(solver, HighsSolver)

    return isinstance(solver, OrToolsSolver)
