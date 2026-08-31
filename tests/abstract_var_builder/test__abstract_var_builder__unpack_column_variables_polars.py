from dataclasses import dataclass
from unittest.mock import MagicMock
import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np
from umip import VariableDomain
from tests.mock_classes_and_data import MockDecisionVariableBuilder


@dataclass
class TestInputs:
    data: pl.DataFrame
    indicators: np.ndarray
    default_unpack_value: float
    return_dtype: VariableDomain


@dataclass
class TestExpected:
    expected_data: pl.DataFrame


def mock_get_variable_value(var):
    """
    Variables are entered in the dataframe in the following format: var_{number}__value_{value}.
    The string is first split on '__' and then on '_', and each time the second part is returned.
    Hence, for var='var_1__value_1.1', we return 1.1.
    """
    return float(var.split("__")[1].split("_")[1])


@pytest.mark.parametrize(
    ("inputs,expected"),
    [
        pytest.param(
            TestInputs(
                data=pl.DataFrame({"vars": []}),
                indicators=np.array([]),
                default_unpack_value=0.0,
                return_dtype=VariableDomain.CONTINUOUS,
            ),
            TestExpected(expected_data=pl.DataFrame({"result": pl.Series([], dtype=pl.Float64)})),
            id="Case 1: Empty dataframe",
        ),
        pytest.param(
            TestInputs(
                data=pl.DataFrame({"vars": ["var_1__value_1.8", "var_2__value_20.1", "var_3__value_4.0"]}),
                indicators=np.array([False, False, False]),
                default_unpack_value=0.0,
                return_dtype=VariableDomain.CONTINUOUS,
            ),
            TestExpected(expected_data=pl.DataFrame({"result": [0.0, 0.0, 0.0]})),
            id="Case 2: Continuous, all indicators are False, all default unpack values",
        ),
        pytest.param(
            TestInputs(
                data=pl.DataFrame({"vars": ["var_1__value_1.8", "var_2__value_20.1", "var_3__value_4.0"]}),
                indicators=np.array([True, True, True]),
                default_unpack_value=0.0,
                return_dtype=VariableDomain.CONTINUOUS,
            ),
            TestExpected(expected_data=pl.DataFrame({"result": [1.8, 20.1, 4.0]})),
            id="Case 3: Continuous unpack all",
        ),
        pytest.param(
            TestInputs(
                data=pl.DataFrame({"vars": ["var_1__value_1.8", "var_2__value_20.1", "var_3__value_4.0"]}),
                indicators=np.array([True, False, True]),
                default_unpack_value=0.0,
                return_dtype=VariableDomain.CONTINUOUS,
            ),
            TestExpected(expected_data=pl.DataFrame({"result": [1.8, 0.0, 4.0]})),
            id="Case 4: Partial unpack with default value",
        ),
        pytest.param(
            TestInputs(
                data=pl.DataFrame({"vars": ["var_1__value_0.01", "var_2__value_0.999999", "var_3__value_1.0000002"]}),
                indicators=np.array([True, True, True]),
                default_unpack_value=0.0,
                return_dtype=VariableDomain.BINARY,
            ),
            TestExpected(expected_data=pl.DataFrame({"result": [False, True, True]})),
            id="Case 5: Binary unpack",
        ),
        pytest.param(
            TestInputs(
                data=pl.DataFrame({"vars": ["var_1__value_2.9999", "var_2__value_0.000001", "var_3__value_4.0"]}),
                indicators=np.array([True, True, True]),
                default_unpack_value=0.0,
                return_dtype=VariableDomain.INTEGER,
            ),
            TestExpected(expected_data=pl.DataFrame({"result": [3, 0, 4]})),
            id="Case 6: Integer unpack (rounding and casting)",
        ),
    ],
)
def test__unpack_column_variables_polars__general(inputs, expected):
    """
    Tests whether the _unpack_column_variables_polars method works as expected.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    solver = MagicMock()
    solver.get_variable_value.side_effect = mock_get_variable_value

    decision_var_col = "vars"
    value_col = "result"

    # Act
    result_df = var_builder._unpack_column_variables_polars(
        data=inputs.data,
        decision_variable_column=decision_var_col,
        decision_variable_value_column=value_col,
        solver=solver,
        indicators=inputs.indicators,
        default_unpack_value=inputs.default_unpack_value,
        variable_domain=inputs.return_dtype,
    )

    # Assert
    assert_frame_equal(result_df, expected.expected_data)


def test__unpack_column_variables_polars__has_invalid_columns__raises_value_error():
    """Test that it raises ValueError when column is missing in a Polars DataFrame."""
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._dataframe_has_invalid_columns = MagicMock(return_value=True)
    var_builder.invalid_column_names_unpack = ["invalid_col", "another_invalid_col"]

    # Act & Assert
    with pytest.raises(ValueError, match=f"DataFrame must not contain column names from the following list: "):
        var_builder._unpack_column_variables_polars(
            data=MagicMock(),
            decision_variable_column=MagicMock(),
            decision_variable_value_column=MagicMock(),
            solver=MagicMock(),
            indicators=MagicMock(),
            default_unpack_value=MagicMock(),
            variable_domain=MagicMock(),
        )
