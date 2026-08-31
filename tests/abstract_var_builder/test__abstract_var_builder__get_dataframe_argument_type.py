from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from tests.mock_classes_and_data import MockDecisionVariableBuilder
from umip.enums.data_types import (
    DataFrameArgumentType,
)


@pytest.mark.parametrize(
    "data, expected_result",
    [
        pytest.param(
            pd.DataFrame(),
            DataFrameArgumentType.PANDAS,
            id="Input argument is a pandas dataframe",
        ),
        pytest.param(
            pl.DataFrame(),
            DataFrameArgumentType.POLARS,
            id="Input argument is a polars dataframe",
        ),
    ],
)
def test__get_dataframe_argument_type(data, expected_result):
    """
    Parameterized test for _get_dataframe_argument_type with different dataframe types.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())

    # Act
    result = var_builder._get_dataframe_argument_type(data=data)

    # Assert
    assert result == expected_result


def test__get_dataframe_argument_type__unsupported_raises_value_error():
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    data = {"col": [1, 2]}  # Dictionary

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported dataframe type"):
        var_builder._get_dataframe_argument_type(data=data)
