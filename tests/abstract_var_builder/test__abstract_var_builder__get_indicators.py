from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tests.mock_classes_and_data import MockDecisionVariableBuilder
from umip import DataFrameArgumentType, FilterColumnArgumentType


@pytest.mark.parametrize(
    ("data", "dataframe_type"),
    [
        pytest.param(
            pd.DataFrame({"col1": [True, False, True], "other": [1, 2, 3]}),
            DataFrameArgumentType.PANDAS,
            id="Pandas dataframe",
        ),
        pytest.param(
            pl.DataFrame({"col1": [True, False, True], "other": [1, 2, 3]}),
            DataFrameArgumentType.POLARS,
            id="Polars dataframe",
        ),
    ],
)
def test__get_indicators__filter_column_as_string__returns_the_column_as_numpy_array(data, dataframe_type):
    """
    Tests whether _get_indicators correctly returns the values from the given indicator column from a dataframe.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_filter_column_argument_type = MagicMock(return_value=FilterColumnArgumentType.STRING)
    var_builder._get_dataframe_argument_type = MagicMock(return_value=dataframe_type)
    var_builder._get_row_count = MagicMock()

    # Act
    result = var_builder._get_indicators("col1", data)

    # Assert
    np.testing.assert_array_equal(result, np.array([True, False, True]))


def test__get_indicators__filter_column_none__returns_all_true_numpy_array():
    """
    Tests whether _get_indicators correctly returns a numpy array of True values of the same length as the data
    when no indicator column is provided and the dataframe has three rows
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_filter_column_argument_type = MagicMock(return_value=FilterColumnArgumentType.NONE)
    var_builder._get_dataframe_argument_type = MagicMock()
    var_builder._get_row_count = MagicMock(return_value=3)

    # Act
    result = var_builder._get_indicators(filter_column=None, data=MagicMock())

    # Assert
    np.testing.assert_array_equal(result, np.array([True, True, True]))


def test__get_indicators__unsupported_dataframe_type__raises_value_error():
    """
    Tests whether _get_indicators raises a ValueError when the dataframe has an unsupported type.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_filter_column_argument_type = MagicMock(return_value=FilterColumnArgumentType.STRING)
    var_builder._get_dataframe_argument_type = MagicMock(return_value="INVALID_TYPE")
    var_builder._get_row_count = MagicMock(return_value=3)

    data = MagicMock()
    filter_column = MagicMock()

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="Cannot find a filter column in DataFrame of unsupported type INVALID_TYPE.",
    ):
        var_builder._get_indicators(filter_column=filter_column, data=data)


def test__get_indicators__unsupported_filter_column__raises_value_error():
    """
    Tests whether _get_indicators raises a ValueError when the given indicator column is of an unsupported type.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_filter_column_argument_type = MagicMock(return_value="INVALID_TYPE")
    var_builder._get_dataframe_argument_type = MagicMock(return_value=DataFrameArgumentType.POLARS)
    var_builder._get_row_count = MagicMock(return_value=3)

    data = MagicMock()
    filter_column = MagicMock()

    # Act & Assert
    with pytest.raises(ValueError, match="Handling filter_column of type INVALID_TYPE not supported."):
        var_builder._get_indicators(filter_column=filter_column, data=data)
