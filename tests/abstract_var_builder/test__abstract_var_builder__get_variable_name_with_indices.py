from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tests.mock_classes_and_data import MockDecisionVariableBuilder
from umip import DataFrameArgumentType, IndexColumnsArgumentType


@dataclass
class TestInputs:
    index_name_columns: list[str] | None
    var_name_prefix: str


@dataclass
class TestExpected:
    expected_names: np.ndarray


@pytest.mark.parametrize(
    ("inputs,expected"),
    [
        pytest.param(
            TestInputs(
                index_name_columns=None,
                var_name_prefix="x",
            ),
            TestExpected(
                expected_names=np.array(["x[0]", "x[1]", "x[2]"]),
            ),
            id="Case 1: No index columns provided (default naming)",
        ),
        pytest.param(
            TestInputs(
                index_name_columns=["id"],
                var_name_prefix="var",
            ),
            TestExpected(
                expected_names=np.array(["var[A]", "var[B]", "var[C]"]),
            ),
            id="Case 2: Single index column",
        ),
        pytest.param(
            TestInputs(
                index_name_columns=["id", "sub_id"],
                var_name_prefix="y",
            ),
            TestExpected(
                expected_names=np.array(["y[A, 1]", "y[B, 2]", "y[C, 3]"]),
            ),
            id="Case 3: Multiple index columns",
        ),
    ],
)
def test__get_variable_name_with_indices__pandas_dataframe__general(inputs, expected):
    """
    Tests whether the _get_variable_name_with_indices method works as expected with pandas dataframes.
    """
    # Arrange
    data = pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "sub_id": [1, 2, 3],
        }
    )
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())

    # Act
    result = var_builder._get_variable_name_with_indices(
        index_column_names=inputs.index_name_columns,
        data=data,
        variable_name=inputs.var_name_prefix,
    )

    # Assert
    np.testing.assert_array_equal(result, expected.expected_names)


@pytest.mark.parametrize(
    ("inputs,expected"),
    [
        pytest.param(
            TestInputs(
                index_name_columns=None,
                var_name_prefix="x",
            ),
            TestExpected(
                expected_names=np.array(["x[0]", "x[1]", "x[2]"]),
            ),
            id="Case 1: No index columns provided (default naming)",
        ),
        pytest.param(
            TestInputs(
                index_name_columns=["id"],
                var_name_prefix="var",
            ),
            TestExpected(
                expected_names=np.array(["var[A]", "var[B]", "var[C]"]),
            ),
            id="Case 2: Single index column",
        ),
        pytest.param(
            TestInputs(
                index_name_columns=["id", "sub_id"],
                var_name_prefix="y",
            ),
            TestExpected(
                expected_names=np.array(["y[A, 1]", "y[B, 2]", "y[C, 3]"]),
            ),
            id="Case 3: Multiple index columns",
        ),
    ],
)
def test__get_variable_name_with_indices__polars_dataframe__general(inputs, expected):
    """
    Tests whether the _get_variable_name_with_indices method works as expected with polars dataframes.
    """
    # Arrange
    data = pl.DataFrame(
        {
            "id": ["A", "B", "C"],
            "sub_id": [1, 2, 3],
        }
    )
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())

    # Act
    result = var_builder._get_variable_name_with_indices(
        index_column_names=inputs.index_name_columns,
        data=data,
        variable_name=inputs.var_name_prefix,
    )

    # Assert
    np.testing.assert_array_equal(result, expected.expected_names)


def test__get_variable_name_with_indices__unsupported_dataframe_type__raises_value_error():
    """
    Test that ValueError is raised when index_name_columns is LIST_OF_STRINGS
    but the dataframe type is not PANDAS or POLARS.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(
        return_value="UNSUPPORTED_TYPE"
    )
    var_builder._get_index_columns_argument_type = MagicMock(
        return_value=IndexColumnsArgumentType.LIST_OF_STRINGS
    )

    index_column_names = ["col1"]
    data = MagicMock()
    variable_name = "x"

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="Cannot add variable indices from a DataFrame of unsupported type",
    ):
        var_builder._get_variable_name_with_indices(
            index_column_names=index_column_names,
            data=data,
            variable_name=variable_name,
        )


def test__get_variable_name_with_indices__unsupported_index_columns_type__raises_value_error():
    """
    Test that ValueError is raised when index_name_columns type is neither LIST_OF_STRINGS nor NONE.
    """
    # Arrange
    var_builder = MockDecisionVariableBuilder(logger=MagicMock())
    var_builder._get_dataframe_argument_type = MagicMock(
        return_value=DataFrameArgumentType.PANDAS
    )
    var_builder._get_index_columns_argument_type = MagicMock(
        return_value="UNKNOWN_ENUM_TYPE"
    )

    index_column_names = 123  # Invalid type
    data = pd.DataFrame({"col1": [1, 2]})
    variable_name = "x"

    # Act & Assert
    with pytest.raises(ValueError, match="Handling index_name_columns of type"):
        var_builder._get_variable_name_with_indices(
            index_column_names=index_column_names,
            data=data,
            variable_name=variable_name,
        )
