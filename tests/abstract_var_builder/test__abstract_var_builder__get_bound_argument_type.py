#  Copyright (c) 2026. ECCO Data & AI and other project contributors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from unittest.mock import MagicMock

import pytest
from tests.mock_classes_and_data import MockDecisionVariableBuilder

from umip.enums.data_types import (
    BoundArgumentType,
)


@pytest.mark.parametrize(
    "bound, expected_result",
    [
        pytest.param(1.5, BoundArgumentType.FLOAT, id="Input argument of type float"),
        pytest.param("col_name", BoundArgumentType.STRING, id="Input argument of type string"),
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
