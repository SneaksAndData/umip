from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from adapta.logs import LoggerInterface

from generic_mip import VariableDomain
from src.umip.solver.cplex import CplexSolver
from src.umip.solver.gurobi import GurobiSolver
from src.umip.solver.highs import HighsSolver
from src.umip.solver.or_tools import OrToolsSolverEngine, OrToolsSolver


def create_solver(solver_type: str, logger: LoggerInterface):
    """Helper to instantiate the appropriate solver."""
    solvers = {
        "ortools": lambda: OrToolsSolver(solver_engine=OrToolsSolverEngine.SCIP, logger=logger),
        "gurobi": lambda: GurobiSolver(logger=logger),
        "highs": lambda: HighsSolver(logger=logger),
        "cplex": lambda: CplexSolver(logger=logger),
    }
    if solver_type not in solvers:
        raise ValueError(f"Invalid solver: {solver_type}")
    return solvers[solver_type]()


@dataclass
class TestInput:
    objective_coefficients: np.ndarray
    constraint_1_coefficients: np.ndarray
    constraint_2_coefficients: np.ndarray
    constraint_1_upper_bound: Any
    constraint_2_upper_bound: Any
    variable_lower_bound: Any = 0
    variable_upper_bound: Any = 100


@dataclass
class TestOutput:
    expected_x: float = 80.0
    expected_y: float = 20.0
    expected_objective: float = 160.0


@pytest.mark.parametrize(
    "solver_type",
    [
        pytest.param("ortools", id="ortools"),
        pytest.param("highs", id="highs"),
        pytest.param("gurobi", id="gurobi", marks=pytest.mark.skip(reason="Not available")),
        pytest.param("cplex", id="cplex", marks=pytest.mark.skip(reason="Not available")),
    ],
)
@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(
            TestInput(
                objective_coefficients=np.array([1, 4], dtype=int),
                constraint_1_coefficients=np.array([1, 1], dtype=int),
                constraint_2_coefficients=np.array([1], dtype=int),
                constraint_1_upper_bound=100,
                constraint_2_upper_bound=20,
            ),
            TestOutput(),
            id="1) Standard Python integers",
        ),
        pytest.param(
            TestInput(
                objective_coefficients=np.array([1.0, 4.0], dtype=float),
                constraint_1_coefficients=np.array([1.0, 1.0], dtype=float),
                constraint_2_coefficients=np.array([1.0], dtype=float),
                constraint_1_upper_bound=100.0,
                constraint_2_upper_bound=20.0,
            ),
            TestOutput(),
            id="2) Standard Python floats",
        ),
        pytest.param(
            TestInput(
                objective_coefficients=np.array([1, 4], dtype=int),
                constraint_1_coefficients=np.array([1.0, 1.0], dtype=float),
                constraint_2_coefficients=np.array([1], dtype=int),
                constraint_1_upper_bound=100,
                constraint_2_upper_bound=20,
            ),
            TestOutput(),
            id="3) Mixed native int objective and float constraints",
        ),
        pytest.param(
            TestInput(
                objective_coefficients=np.array([1, 4], dtype=np.int32),
                constraint_1_coefficients=np.array([1, 1], dtype=np.int64),
                constraint_2_coefficients=np.array([1], dtype=np.int32),
                constraint_1_upper_bound=np.int32(100),
                constraint_2_upper_bound=np.int32(20),
            ),
            TestOutput(),
            id="4) Numpy int32 and int64 mixed",
        ),
        pytest.param(
            TestInput(
                objective_coefficients=np.array([1.0, 4.0], dtype=np.float32),
                constraint_1_coefficients=np.array([1.0, 1.0], dtype=np.float64),
                constraint_2_coefficients=np.array([1.0], dtype=np.float32),
                constraint_1_upper_bound=np.float32(100.0),
                constraint_2_upper_bound=np.float32(20.0),
            ),
            TestOutput(),
            id="5) Numpy float32 and float64 mixed",
        ),
        pytest.param(
            TestInput(
                objective_coefficients=np.array([1, 4], dtype=np.int64),
                constraint_1_coefficients=np.array([1.0, 1.0], dtype=np.float32),
                constraint_2_coefficients=np.array([1.0], dtype=np.float32),
                constraint_1_upper_bound=np.float64(100.0),
                constraint_2_upper_bound=np.float64(20.0),
            ),
            TestOutput(),
            id="6) Heavy mix: int64 objective with float32 coefficients and float64 bounds",
        ),
        pytest.param(
            TestInput(
                objective_coefficients=np.array([1.0, 4.0], dtype=np.float64),
                constraint_1_coefficients=np.array([1, 1], dtype=np.int32),
                constraint_2_coefficients=np.array([1], dtype=np.int32),
                constraint_1_upper_bound=np.int64(100),
                constraint_2_upper_bound=np.int64(20),
            ),
            TestOutput(),
            id="7) Heavy mix: float64 objective with int32 coefficients and int64 bounds",
        ),
        pytest.param(
            TestInput(
                objective_coefficients=np.array([1, 4], dtype=int),
                constraint_1_coefficients=np.array([1, 1], dtype=int),
                constraint_2_coefficients=np.array([1], dtype=int),
                constraint_1_upper_bound=100,
                constraint_2_upper_bound=20,
                variable_lower_bound=False,  # 0
                variable_upper_bound=True,  # 1
            ),
            TestOutput(expected_x=1.0, expected_y=1.0, expected_objective=5.0),
            id="8) Boolean variable bounds - x,y restricted to [0, 1]",
        ),
    ],
)
def test__solver_numeric_type_handling__functional(
    solver_type: str, inputs: TestInput, expected: TestOutput, logger: LoggerInterface
):
    """
    Functional test for numeric type handling and variable bounds.

    General Model:
    Maximize: c1*x + c2*y
    Subject to:
        a1*x + a2*y <= b1
               a3*y <= b2
        x, y in [var_lb, var_ub], Integer

    This test ensures that the solver correctly casts and handles:
    1. Native Python types (int, float, bool)
    2. NumPy types (int32, int64, float32, float64)
    3. Variable bounds provided as Booleans.
    """
    # Arrange
    solver = create_solver(solver_type, logger)

    # Act
    x = solver.add_variable(
        lower_bound=inputs.variable_lower_bound,
        upper_bound=inputs.variable_upper_bound,
        name="x",
        variable_domain=VariableDomain.INTEGER,
    )
    y = solver.add_variable(
        lower_bound=inputs.variable_lower_bound,
        upper_bound=inputs.variable_upper_bound,
        name="y",
        variable_domain=VariableDomain.INTEGER,
    )

    solver.add_multiple_objective_terms(
        coefficients=inputs.objective_coefficients,
        variables=np.array([x, y]),
    )

    solver.add_constraint(
        upper_bound=inputs.constraint_1_upper_bound,
        coefficients=inputs.constraint_1_coefficients,
        variables=np.array([x, y]),
        name="constraint_1",
    )

    solver.add_constraint(
        upper_bound=inputs.constraint_2_upper_bound,
        coefficients=inputs.constraint_2_coefficients,
        variables=np.array([y]),
        name="constraint_2",
    )

    solver.set_optimization_direction(True)
    solver.solve()

    # Assert
    assert solver.get_variable_value(x) == pytest.approx(expected.expected_x, abs=1e-6)
    assert solver.get_variable_value(y) == pytest.approx(expected.expected_y, abs=1e-6)
    assert solver.get_objective_value() == pytest.approx(expected.expected_objective, abs=1e-6)
