from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest
from generic_mip import VariableDomain

from tests.mock_classes_and_data import MockDecisionVariableBuilder
from umip import BoundArgumentType, SolverFactory, SolverType
from umip.enums.bound_type import BoundType

bound_col = "bound"
bound_col_none = "bound_none"


@dataclass
class TestInputs:
    bound: str | float | None
    variable_dtype: VariableDomain
    bound_type: BoundType


@dataclass
class TestExpected:
    expected_bound: np.ndarray


solver_mocked = SolverFactory(logger=MagicMock()).construct(solver_type=SolverType.ORTOOLS_SCIP)


@pytest.mark.parametrize("solver", ["OrTools", "Highs"], indirect=True)
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            pd.DataFrame({bound_col: [10.0, 20.0, 30.0], bound_col_none: [10.0, None, 30.0]}),
            id="Pandas dataframe",
        ),
        pytest.param(
            pl.DataFrame({bound_col: [10.0, 20.0, 30.0], bound_col_none: [10.0, None, 30.0]}),
            id="Polars dataframe",
        ),
    ],
)
@pytest.mark.parametrize(
    ("inputs,expected"),
    [
        pytest.param(
            TestInputs(
                bound=bound_col,
                variable_dtype=MagicMock(),
                bound_type=MagicMock(),
            ),
            TestExpected(
                expected_bound=np.array([10.0, 20.0, 30.0]),
            ),
            id="Case 1: Bound is a string and the column exists",
        ),
        pytest.param(
            TestInputs(
                bound=0.2,
                variable_dtype=MagicMock(),
                bound_type=MagicMock(),
            ),
            TestExpected(
                expected_bound=np.array([0.2, 0.2, 0.2]),
            ),
            id="Case 2: Bound is a float",
        ),
        pytest.param(
            TestInputs(
                bound=None,
                variable_dtype=VariableDomain.BINARY,
                bound_type=BoundType.LOWER,
            ),
            TestExpected(
                expected_bound=np.array([0.0, 0.0, 0.0]),
            ),
            id="Case 3: Bound is none, variable domain is binary and bound type is lower",
        ),
        pytest.param(
            TestInputs(
                bound=None,
                variable_dtype=VariableDomain.BINARY,
                bound_type=BoundType.UPPER,
            ),
            TestExpected(
                expected_bound=np.array([1.0, 1.0, 1.0]),
            ),
            id="Case 4: Bound is none, variable domain is binary and bound type is upper",
        ),
        pytest.param(
            TestInputs(
                bound=None,
                variable_dtype=VariableDomain.CONTINUOUS,
                bound_type=BoundType.LOWER,
            ),
            TestExpected(
                expected_bound=np.array([-np.inf, -np.inf, -np.inf]),
            ),
            id="Case 5: Bound is none, variable domain is continuous and bound type is upper",
        ),
        pytest.param(
            TestInputs(
                bound=None,
                variable_dtype=VariableDomain.CONTINUOUS,
                bound_type=BoundType.UPPER,
            ),
            TestExpected(
                expected_bound=np.array(
                    [
                        solver_mocked.infinity(),
                        solver_mocked.infinity(),
                        solver_mocked.infinity(),
                    ]
                ),
            ),
            id="Case 6: Bound is none, variable domain is continuous and bound type is upper",
        ),
        pytest.param(
            TestInputs(
                bound=bound_col_none,
                variable_dtype=MagicMock(),
                bound_type=MagicMock(),
            ),
            TestExpected(
                expected_bound=np.array([10.0, solver_mocked.infinity(), 30.0]),
            ),
            id="Case 7: Bound is a string and the column exists. The column has None, which should be solver.infinity()",
        ),
    ],
)
def test__get_bounds__general(
    solver,
    data: pl.DataFrame | pd.DataFrame,
    inputs: TestInputs,
    expected: TestExpected,
):
    """
    Tests whether the _get_bounds method works as expected with polars and pandas dataframes.
    Case 1: bound is a string and column exists
    Case 2: bound is a float
    Case 3: bound is None, bound type is lower and variable domain is binary
    Case 4: bound is None, bound type is upper and variable domain is binary
    Case 5: bound is None, bound type is lower and variable domain is continuous
    Case 6: bound is None, bound type is upper and variable domain is continuous
    Case 7: bound is a string and column exists. The column has None, which should be solver.infinity()
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())

    # Act
    result = var_builder._get_bounds(
        solver=solver,
        bound=inputs.bound,
        data=data,
        variable_domain=inputs.variable_dtype,
        bound_type=inputs.bound_type,
    )

    # Assert
    assert np.array_equal(result, expected.expected_bound)


def test__get_bounds__invalid_dataframe_type__raises_value_error():
    """
    Tests whether the _get_bounds method throws an error when the bound argument is a string but the dataframe type is invalid.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value="INVALID_TYPE")
    var_builder._get_row_count = MagicMock()
    var_builder._get_bound_argument_type = MagicMock(return_value=BoundArgumentType.STRING)

    # Act & Assert
    with pytest.raises(ValueError, match="Cannot find a bound column in DataFrame of unsupported type"):
        var_builder._get_bounds(
            solver=MagicMock(),
            bound=MagicMock(),
            data=MagicMock(),
            variable_domain=MagicMock(),
            bound_type=MagicMock(),
        )


def test__get_bounds__invalid_bound_type__raises_value_error():
    """
    Tests whether the _get_bounds method throws an error when the bound argument is None and the bound type is invalid.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock()
    var_builder._get_row_count = MagicMock()
    var_builder._get_bound_argument_type = MagicMock(return_value=BoundArgumentType.NONE)

    # Act & Assert
    with pytest.raises(ValueError, match="Handling of bound_type=some_invalid_type not supported."):
        var_builder._get_bounds(
            solver=MagicMock(),
            bound=None,
            data=MagicMock(),
            variable_domain=MagicMock(),
            bound_type="some_invalid_type",
        )


def test__get_bounds__invalid_bound__raises_value_error():
    """
    Tests whether the _get_bounds method throws an error when the bound argument has an invalid type.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock()
    var_builder._get_row_count = MagicMock()
    var_builder._get_bound_argument_type = MagicMock(return_value="INVALID_TYPE")

    # Act & Assert
    with pytest.raises(ValueError, match="Handling bound of type INVALID_TYPE not supported."):
        var_builder._get_bounds(
            solver=MagicMock(),
            bound=MagicMock(),
            data=MagicMock(),
            variable_domain=MagicMock(),
            bound_type=BoundType.LOWER,
        )
