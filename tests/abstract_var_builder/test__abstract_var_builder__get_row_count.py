from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from tests.mock_classes_and_data import MockDecisionVariableBuilder
from umip.enums.data_types import DataFrameArgumentType


@pytest.mark.parametrize(
    "data, dataframe_type, expected_row_count",
    [
        pytest.param(
            pd.DataFrame({"col": [1, 2, 3]}),
            DataFrameArgumentType.PANDAS,
            3,
            id="Input is a pandas dataframe, returns length",
        ),
        pytest.param(
            pl.DataFrame({"col": [1, 2, 3, 4]}),
            DataFrameArgumentType.POLARS,
            4,
            id="Input is a polars dataframe, returns height",
        ),
    ],
)
def test__get_row_count(data, dataframe_type, expected_row_count):
    """
    Tests that _get_row_count correctly returns the number of rows in a dataframe.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(return_value=dataframe_type)

    # Act
    result = var_builder._get_row_count(data=data)

    # Assert
    assert result == expected_row_count


def test__get_row_count__unsupported_type__raises_value_error():
    """Test that _get_row_count raises ValueError for unsupported types."""
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(
        return_value="UNSUPPORTED_TYPE"
    )
    data = MagicMock()

    # Act & Assert
    with pytest.raises(
        ValueError, match="Cannot get row count for unsupported data type"
    ):
        var_builder._get_row_count(data)
