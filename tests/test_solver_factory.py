"""Tests for SolverFactory construction and solver config validation."""

import pytest

from generic_mip import AbstractOptimizationSolver
from src.umip.enums.solver_type import SolverType
from src.umip.solver.cplex import CplexSolver
from src.umip.solver.highs import HighsSolver
from src.umip.solver.or_tools import OrToolsSolver, OrToolsSolverEngine
from src.umip.solver_config import GurobiSolverConfig
from src.umip import SolverFactory


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
    with pytest.raises(ValueError):
        sut.construct(solver_type=100)


def test_construct_raises_for_incompatible_solver_config(logger) -> None:
    # Arrange
    sut = SolverFactory(logger=logger)

    # Act & Assert
    with pytest.raises(ValueError):
        sut.construct(solver_type=SolverType.ORTOOLS_SCIP, solver_config=GurobiSolverConfig())


def __isinstance_of_solver(solver: AbstractOptimizationSolver, solver_type: SolverType) -> bool:
    if solver_type == SolverType.CPLEX:
        return isinstance(solver, CplexSolver)

    if solver_type == SolverType.HIGHS:
        return isinstance(solver, HighsSolver)

    return isinstance(solver, OrToolsSolver)
