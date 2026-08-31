from unittest.mock import MagicMock, call

import pandas as pd
import polars as pl
import pytest
from tests.mock_classes_and_data import MockDecisionVariableBuilder

from umip import VariableDomain
from umip.enums import DataFrameArgumentType
from umip.enums.bound_type import BoundType


def test__build_column_variables__pandas_dataframe__pandas_method_is_called():
    """
    Tests that _build_column_variables_pandas is called when a pandas dataframe is passed.
    """
    # Arrange
    data = pd.DataFrame()

    variable_builder = MockDecisionVariableBuilder(logger=MagicMock())
    solver = MagicMock()

    variable_builder._get_dataframe_argument_type = MagicMock(return_value=DataFrameArgumentType.PANDAS)
    variable_builder._get_variable_name = MagicMock()
    variable_builder._get_bounds = MagicMock()
    variable_builder._get_indicators = MagicMock()
    variable_builder._get_variable_name_with_indices = MagicMock()
    variable_builder._build_column_variables_pandas = MagicMock()
    variable_builder._build_column_variables_polars = MagicMock()

    # Act
    variable_builder.build_column_variables(
        solver=solver,
        data=data,
        destination_column=MagicMock(),
        variable_domain=VariableDomain.INTEGER,
        lower_bound="lower_bound",
        upper_bound="upper_bound",
    )

    # Assert

    variable_builder._get_dataframe_argument_type.assert_called_once()
    variable_builder._get_variable_name.assert_called_once()
    variable_builder._get_indicators.assert_called_once()
    variable_builder._get_variable_name_with_indices.assert_called_once()

    variable_builder._get_bounds.assert_has_calls(
        [
            call(
                bound="lower_bound",
                solver=solver,
                data=data,
                variable_domain=VariableDomain.INTEGER,
                bound_type=BoundType.LOWER,
            ),
            call(
                bound="upper_bound",
                solver=solver,
                data=data,
                variable_domain=VariableDomain.INTEGER,
                bound_type=BoundType.UPPER,
            ),
        ]
    )
    assert variable_builder._get_bounds.call_count == 2

    variable_builder._build_column_variables_pandas.assert_called_once()
    variable_builder._build_column_variables_polars.assert_not_called()


def test__build_column_variables__polars_dataframe__polars_method_is_called():
    """
    Tests whether the correct method calls are made if a polars dataframe is passed. In particular,
    tests whether _build_column_variables_polars is called, as opposed to _build_column_variables_pandas.
    """
    # Arrange
    data = pl.DataFrame()
    variable_builder = MockDecisionVariableBuilder(logger=MagicMock())
    solver = MagicMock()

    variable_builder._get_dataframe_argument_type = MagicMock(return_value=DataFrameArgumentType.POLARS)
    variable_builder._get_variable_name = MagicMock()
    variable_builder._get_bounds = MagicMock()
    variable_builder._get_indicators = MagicMock()
    variable_builder._get_variable_name_with_indices = MagicMock()
    variable_builder._build_column_variables_pandas = MagicMock()
    variable_builder._build_column_variables_polars = MagicMock()

    # Act
    variable_builder.build_column_variables(
        solver=solver,
        data=data,
        destination_column=MagicMock(),
        variable_domain=VariableDomain.INTEGER,
        lower_bound="lower_bound",
        upper_bound="upper_bound",
    )

    # Assert
    variable_builder._get_dataframe_argument_type.assert_called_once()
    variable_builder._get_variable_name.assert_called_once()
    variable_builder._get_indicators.assert_called_once()
    variable_builder._get_variable_name_with_indices.assert_called_once()

    variable_builder._get_bounds.assert_has_calls(
        [
            call(
                bound="lower_bound",
                solver=solver,
                data=data,
                variable_domain=VariableDomain.INTEGER,
                bound_type=BoundType.LOWER,
            ),
            call(
                bound="upper_bound",
                solver=solver,
                data=data,
                variable_domain=VariableDomain.INTEGER,
                bound_type=BoundType.UPPER,
            ),
        ]
    )
    assert variable_builder._get_bounds.call_count == 2

    variable_builder._build_column_variables_polars.assert_called_once()
    variable_builder._build_column_variables_pandas.assert_not_called()


def test__build_column_variables__invalid_dataframe_type__raises_value_error():
    """
    Tests that it raises ValueError when the dataframe of an unsupported type is passed.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())

    var_builder._get_dataframe_argument_type = MagicMock(return_value="INVALID_TYPE")
    var_builder._get_variable_name = MagicMock()
    var_builder._get_bounds = MagicMock()
    var_builder._get_indicators = MagicMock()
    var_builder._get_variable_name_with_indices = MagicMock()

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="No method for building variables in DataFrame INVALID_TYPE type is defined.",
    ):
        var_builder.build_column_variables(
            solver=MagicMock(),
            data=MagicMock(),
            destination_column=MagicMock(),
            variable_domain=MagicMock(),
        )


index_col = "index"
ub_col = "upper_bound"
lb_col = "lower_bound"
make_var_col = "make_var"


@pytest.mark.parametrize("solver", ["OrTools"], indirect=True)
@pytest.mark.parametrize(
    "input_data",
    [
        pytest.param(
            pl.DataFrame(
                {
                    index_col: [1, 2, 3],
                    ub_col: [10.0, 20.0, 30.0],
                    lb_col: [1.0, 2.0, 3.0],
                    make_var_col: [True, False, True],
                }
            ),
            id="Case 1: polars dataframe",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    index_col: [1, 2, 3],
                    ub_col: [10.0, 20.0, 30.0],
                    lb_col: [1.0, 2.0, 3.0],
                    make_var_col: [True, False, True],
                }
            ),
            id="Case 2: pandas dataframe",
        ),
    ],
)
def test__build_column_variables__functional(solver, input_data):
    """
    Tests the behavior of the build_column_variables method.
    """
    # Arrange
    destination_column = "var"
    variable_dtype = VariableDomain.INTEGER

    variable_builder = MockDecisionVariableBuilder(logger=MagicMock())

    # Act
    data_with_vars = variable_builder.build_column_variables(
        solver=solver,
        data=input_data,
        destination_column=destination_column,
        variable_domain=variable_dtype,
        index_name_columns=[index_col],
        filter_column=make_var_col,
        lower_bound=lb_col,
        upper_bound=ub_col,
    )

    # Assert

    # row 1: variable named var[1] with lower bound 1.0 and upper bound 10.0
    assert data_with_vars[destination_column][0].name() == "var[1]"
    assert data_with_vars[destination_column][0].lb() == 1.0
    assert data_with_vars[destination_column][0].ub() == 10.0

    # row 2: no variable due to make_var = False
    assert data_with_vars[destination_column][1] is None

    # row 3: variable named var[3] with lower bound 3.0 and upper bound 30.0
    assert data_with_vars[destination_column][2].name() == "var[3]"
    assert data_with_vars[destination_column][2].lb() == 3.0
    assert data_with_vars[destination_column][2].ub() == 30.0
