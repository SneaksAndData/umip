from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest
from tests.mock_classes_and_data import MockDecisionVariableBuilder

from umip.enums import DataFrameArgumentType


def test__dataframe_has_invalid_columns__pandas_with_invalid__returns_true():
    """
    Tests that it returns True if any invalid column exists in pandas DataFrame.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value=DataFrameArgumentType.PANDAS)
    data = pd.DataFrame(
        {
            "col1": [],
            "col2": [],
        }
    )
    invalid_columns = ["col2", "col3"]

    # Act
    result = var_builder._dataframe_has_invalid_columns(data=data, invalid_column_names=invalid_columns)

    # Assert
    assert result is True


def test__dataframe_has_invalid_columns__pandas_without_invalid__returns_false():
    """Test that it returns False if no invalid column exists in pandas DataFrame."""
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value=DataFrameArgumentType.PANDAS)
    data = pd.DataFrame(
        {
            "col1": [],
            "col2": [],
        }
    )
    invalid_columns = ["col3", "col4"]

    # Act
    result = var_builder._dataframe_has_invalid_columns(data=data, invalid_column_names=invalid_columns)

    # Assert
    assert result is False


def test__dataframe_has_invalid_columns__polars_with_invalid__returns_true():
    """Test that it returns True if any invalid column exists in polars DataFrame."""
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value=DataFrameArgumentType.POLARS)

    data = pl.DataFrame(
        {
            "col1": [],
            "col2": [],
        }
    )
    invalid_columns = ["col1"]

    # Act
    result = var_builder._dataframe_has_invalid_columns(data=data, invalid_column_names=invalid_columns)

    # Assert
    assert result is True


def test__dataframe_has_invalid_columns__polars_without_invalid__returns_false():
    """Test that it returns False if no invalid column exists in polars DataFrame."""
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value=DataFrameArgumentType.POLARS)
    data = pl.DataFrame(
        {
            "col1": [],
        }
    )
    invalid_columns = ["col2", "col3"]

    # Act
    result = var_builder._dataframe_has_invalid_columns(data=data, invalid_column_names=invalid_columns)

    # Assert
    assert result is False


def test__dataframe_has_invalid_columns__unsupported_type__raises_value_error():
    """Test that it raises ValueError when dataframe type is unsupported."""
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value="UNSUPPORTED_TYPE")

    data = MagicMock()
    invalid_columns = ["col1"]

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="Cannot check for invalid column existence for unsupported dataframe type",
    ):
        var_builder._dataframe_has_invalid_columns(data=data, invalid_column_names=invalid_columns)
