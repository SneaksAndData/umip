import pytest
import pandas as pd
import polars as pl
from unittest.mock import MagicMock

from umip import DataFrameArgumentType
from tests.mock_classes_and_data import MockDecisionVariableBuilder


@pytest.mark.parametrize(
    ("data", "dataframe_type"),
    [
        pytest.param(
            pd.DataFrame(
                {
                    "existing_col": [],
                }
            ),
            DataFrameArgumentType.PANDAS,
            id="Pandas dataframe",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "existing_col": [],
                }
            ),
            DataFrameArgumentType.POLARS,
            id="Polars dataframe",
        ),
    ],
)
def test__dataframe_has_column__column_exists__returns_true(data, dataframe_type):
    """Test that it returns True when column exists in the given dataframe."""
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value=dataframe_type)
    column_name = "existing_col"

    # Act
    result = var_builder._dataframe_has_column(data=data, column_name=column_name)

    # Assert
    assert result is True


@pytest.mark.parametrize(
    ("data", "dataframe_type"),
    [
        pytest.param(
            pd.DataFrame(
                {
                    "other_col": [],
                }
            ),
            DataFrameArgumentType.PANDAS,
            id="Pandas dataframe",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "other_col": [],
                }
            ),
            DataFrameArgumentType.POLARS,
            id="Polars dataframe",
        ),
    ],
)
def test__dataframe_has_column__column_missing__raises_value_error(data, dataframe_type):
    """Test that it raises ValueError when column is missing from the dataframe."""
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value=dataframe_type)

    column_name = "missing_col"

    # Act & Assert
    with pytest.raises(ValueError, match=f"DataFrame does not contain column {column_name}"):
        var_builder._dataframe_has_column(data=data, column_name=column_name)


def test__dataframe_has_column__unsupported_dataframe_type__raises_value_error():
    """Test that it raises ValueError when the dataframe type is unsupported."""
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value="UNSUPPORTED_TYPE")

    data = MagicMock()

    # Act & Assert
    with pytest.raises(ValueError, match="Cannot check for column existence for unsupported dataframe type"):
        var_builder._dataframe_has_column(data=data, column_name="some_column")
