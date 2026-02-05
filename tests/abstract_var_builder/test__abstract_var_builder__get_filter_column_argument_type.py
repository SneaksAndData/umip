from unittest.mock import MagicMock
import pytest
from generic_mip.enums.data_types import (
    FilterColumnArgumentType,
)
from tests.mock_classes_and_data import MockDecisionVariableBuilder


@pytest.mark.parametrize(
    "filter_col, expected_result",
    [
        pytest.param(
            "filter_col",
            FilterColumnArgumentType.STRING,
            id="Input argument is a string representing a column",
        ),
        pytest.param(
            None,
            FilterColumnArgumentType.NONE,
            id="Input argument is None",
        ),
    ],
)
def test__get_filter_column_argument_type(filter_col, expected_result):
    """
    Tests that _get_filter_column_argument_type returns the correct type of the filter column argument.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())

    # Act
    result = var_builder._get_filter_column_argument_type(filter_column=filter_col)

    # Assert
    assert result == expected_result


def test__get_filter_column_argument_type__unsupported_raises_value_error():
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    filter_col = 123

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported filter column argument type"):
        var_builder._get_filter_column_argument_type(filter_col)
