import pytest
from unittest.mock import MagicMock
from umip.enums.data_types import (
    BoundArgumentType,
)
from tests.mock_classes_and_data import MockDecisionVariableBuilder


@pytest.mark.parametrize(
    "bound, expected_result",
    [
        pytest.param(1.5, BoundArgumentType.FLOAT, id="Input argument of type float"),
        pytest.param(
            "col_name", BoundArgumentType.STRING, id="Input argument of type string"
        ),
        pytest.param(None, BoundArgumentType.NONE, id="Input argument is None"),
    ],
)
def test__get_bound_argument_type(bound, expected_result):
    """
    Tests that _get_bound_argument_type returns the correct type of the bound argument.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())

    # Act
    result = var_builder._get_bound_argument_type(bound=bound)

    # Assert
    assert result == expected_result


def test__get_bound_argument_type__unsupported_raises_value_error():
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    bound = 10  # Integer (unexpected per type hint logic, treated as unsupported)

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported bound argument type"):
        var_builder._get_bound_argument_type(bound)
