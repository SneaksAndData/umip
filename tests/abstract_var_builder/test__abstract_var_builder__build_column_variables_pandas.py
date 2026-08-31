from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.mock_classes_and_data import MockDecisionVariableBuilder


@dataclass
class TestInputs:
    data: pd.DataFrame
    names: np.ndarray
    indicators: np.ndarray


@dataclass
class TestExpected:
    expected_data: pd.DataFrame


def side_effect_add_variable(lower_bound, upper_bound, name, variable_domain):
    return name + "_var"


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(
            TestInputs(
                data=pd.DataFrame(),
                names=np.array(["x", "y", "z"]),
                indicators=np.array([True, True, True]),
            ),
            TestExpected(
                expected_data=pd.DataFrame({"destination_column": pd.Series(dtype="object")}),
            ),
            id="Case 1: empty dataframe",
        ),
        pytest.param(
            TestInputs(
                data=pd.DataFrame(
                    {
                        "index": [1, 2, 3],
                    }
                ),
                names=np.array(["x", "y", "z"]),
                indicators=np.array([True, True, True]),
            ),
            TestExpected(
                expected_data=pd.DataFrame(
                    {
                        "index": [1, 2, 3],
                        "destination_column": ["x_var", "y_var", "z_var"],
                    }
                )
            ),
            id="Case 2: nonempty dataframe, variables created for all rows",
        ),
        pytest.param(
            TestInputs(
                data=pd.DataFrame(
                    {
                        "index": [1, 2, 3],
                    }
                ),
                names=np.array(["x", "y", "z"]),
                indicators=np.array([True, False, True]),
            ),
            TestExpected(
                expected_data=pd.DataFrame(
                    {
                        "index": [1, 2, 3],
                        "destination_column": ["x_var", None, "z_var"],
                    }
                )
            ),
            id="Case 3: nonempty dataframe, variable not created for the second row",
        ),
    ],
)
def test__build_column_variables_pandas__general(inputs, expected):
    """
    Tests whether the variables are built correctly in a pandas dataframe.
    """
    # Arrange
    variable_builder = MockDecisionVariableBuilder(logger=MagicMock())
    solver = MagicMock()
    solver.add_variable.side_effect = side_effect_add_variable

    # Act
    result = variable_builder._build_column_variables_pandas(
        solver=solver,
        data=inputs.data,
        destination_column="destination_column",
        variable_domain=MagicMock(),
        lower_bound_values=MagicMock(),
        upper_bound_values=MagicMock(),
        names=inputs.names,
        indicators=inputs.indicators,
    )

    # Assert
    assert_frame_equal(result, expected.expected_data)


def test__build_column_variables_pandas__has_invalid_columns__raises_value_error():
    """
    Tests that it raises ValueError when the dataframe contains invalid columns.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder.invalid_column_names_build = ["invalid_col", "another_invalid_col"]

    # Mock the necessary method
    var_builder._dataframe_has_invalid_columns = MagicMock(return_value=True)

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="DataFrame must not contain column names from the following list: ",
    ):
        var_builder._build_column_variables_pandas(
            solver=MagicMock(),
            data=MagicMock(),
            destination_column="destination_column",
            variable_domain=MagicMock(),
            lower_bound_values=MagicMock(),
            upper_bound_values=MagicMock(),
            names=MagicMock(),
            indicators=MagicMock(),
        )
