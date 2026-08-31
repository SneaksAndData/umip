import pytest
from unittest.mock import MagicMock
from umip.enums.data_types import (
    IndexColumnsArgumentType,
)
from tests.mock_classes_and_data import MockDecisionVariableBuilder


@pytest.mark.parametrize(
    "index_cols, expected_result",
    [
        pytest.param(
            ["col1", "col2"],
            IndexColumnsArgumentType.LIST_OF_STRINGS,
            id="Input argument is a list of strings representing index columns",
        ),
        pytest.param(
            None,
            IndexColumnsArgumentType.NONE,
            id="Input argument is None",
        ),
    ],
)
def test__get_index_columns_argument_type(index_cols, expected_result):
    """
    Tests that _get_index_columns_argument_type returns the correct type of the index columns argument.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())

    # Act
    result = var_builder._get_index_columns_argument_type(index_name_columns=index_cols)

    # Assert
    assert result == expected_result


@pytest.mark.parametrize(
    "index_cols, expected_exception_message",
    [
        pytest.param(
            "not_a_list",
            "Unsupported index columns argument type",
            id="Input argument is not a list",
        ),
        pytest.param(
            ["col1", 123],
            "Unsupported index columns argument type",
            id="Input argument is a list containing a non-string element",
        ),
    ],
)
def test__get_index_columns_argument_type__invalid_inputs_raise_value_error(index_cols, expected_exception_message):
    """
    Tests that a ValueError is raised when the input to _get_index_columns_argument_type is invalid.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())

    # Act & Assert
    with pytest.raises(ValueError, match=expected_exception_message):
        var_builder._get_index_columns_argument_type(index_name_columns=index_cols)
