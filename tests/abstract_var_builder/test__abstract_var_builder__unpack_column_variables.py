from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tests.mock_classes_and_data import MockDecisionVariableBuilder
from umip import VariableDomain

variables_column = "vars"
unpacked_values_column = "unpacked_val"
filter_column = "filter"

var_attr = "Var"
value_attr = "value"


def test__unpack_column_variables__pandas_dataframe__pandas_method_is_called():
    """
    Tests whether the correct method calls are made if a pandas dataframe is passed.
    """
    # Arrange
    data = pd.DataFrame(
        {
            variables_column: [],
        }
    )

    variable_builder = MockDecisionVariableBuilder(logger=MagicMock())
    variable_builder._unpack_column_variables_pandas = MagicMock()
    variable_builder._unpack_column_variables_polars = MagicMock()

    solver = MagicMock()

    # Act
    variable_builder.unpack_column_variables(
        data=data,
        decision_variable_column=variables_column,
        decision_variable_value_column="vals",
        solver=solver,
    )

    # Assert
    variable_builder._unpack_column_variables_pandas.assert_called_once()
    variable_builder._unpack_column_variables_polars.assert_not_called()


def test__unpack_column_variables__polars_dataframe__polars_method_is_called():
    """
    Tests whether the correct method calls are made if a polars dataframe is passed.
    """
    # Arrange
    data = pl.DataFrame({variables_column: []})

    variable_builder = MockDecisionVariableBuilder(logger=MagicMock())
    variable_builder._unpack_column_variables_pandas = MagicMock()
    variable_builder._unpack_column_variables_polars = MagicMock()

    solver = MagicMock()

    # Act
    variable_builder.unpack_column_variables(
        data=data,
        decision_variable_column=variables_column,
        decision_variable_value_column="vals",
        solver=solver,
    )

    # Assert
    variable_builder._unpack_column_variables_polars.assert_called_once()
    variable_builder._unpack_column_variables_pandas.assert_not_called()


def test__unpack_column_variables__invalid_dataframe__raises_value_error():
    """
    Tests that an error is raised if the dataframe type is not supported.
    """
    # Arrange
    variable_builder = MockDecisionVariableBuilder(logger=MagicMock())
    variable_builder._get_dataframe_argument_type = MagicMock(
        return_value="UNSUPPORTED_TYPE"
    )
    variable_builder._get_indicators = MagicMock()

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="No method for unpacking variables in DataFrame of type UNSUPPORTED_TYPE is defined",
    ):
        variable_builder.unpack_column_variables(
            data="not a dataframe",
            decision_variable_column=variables_column,
            decision_variable_value_column="vals",
            solver=MagicMock(),
        )


def mock_solver_get_value(var_obj):
    """Simulates retrieving a value from a solver based on a mock object's value attribute."""
    return getattr(var_obj, value_attr, 0.0)


def test__unpack_column_variables__pandas__functional():
    """
    Tests the end-to-end behavior of unpack_column_variables for a pandas dataframe.
    """
    # Arrange
    variable_builder = MockDecisionVariableBuilder(logger=MagicMock())
    solver = MagicMock()
    solver.get_variable_value.side_effect = mock_solver_get_value
    input_data = pd.DataFrame(
        {
            variables_column: [
                type(var_attr, (), {value_attr: 10.0})(),
                None,
                type(var_attr, (), {value_attr: 20.99999})(),
            ],
            filter_column: [True, False, True],
        }
    )

    expected_data = pd.DataFrame(
        {
            filter_column: [True, False, True],
            unpacked_values_column: [10, 0, 21],
        }
    )

    # Act
    result_data = variable_builder.unpack_column_variables(
        data=input_data,
        decision_variable_column=variables_column,
        decision_variable_value_column=unpacked_values_column,
        solver=solver,
        filter_column=filter_column,
        default_unpack_value=0.0,
        variable_domain=VariableDomain.INTEGER,
    )

    # Assert
    pd.testing.assert_frame_equal(result_data, expected_data)


def test__unpack_column_variables__polars__functional():
    """
    Tests the end-to-end behavior of unpack_column_variables for a polars dataframe.
    """
    # Arrange
    variable_builder = MockDecisionVariableBuilder(logger=MagicMock())
    solver = MagicMock()
    solver.get_variable_value.side_effect = mock_solver_get_value
    input_data = pl.DataFrame(
        {
            variables_column: [
                type(var_attr, (), {value_attr: 10.0})(),
                None,
                type(var_attr, (), {value_attr: 20.99999})(),
            ],
            filter_column: [True, False, True],
        }
    )

    expected_data = pl.DataFrame(
        {
            filter_column: [True, False, True],
            unpacked_values_column: [10, 0, 21],
        }
    )

    # Act
    result_data = variable_builder.unpack_column_variables(
        data=input_data,
        decision_variable_column=variables_column,
        decision_variable_value_column=unpacked_values_column,
        solver=solver,
        filter_column=filter_column,
        default_unpack_value=0.0,
        variable_domain=VariableDomain.INTEGER,
    )

    # Assert
    assert_frame_equal(result_data, expected_data)
