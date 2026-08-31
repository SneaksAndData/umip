from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from umip import AbstractOptimizationSolver


@dataclass
class TestInput:
    value: npt.NDArray[np.floating | np.integer | np.bool_] | float | int | bool


@dataclass
class TestOutput:
    expected: float | npt.NDArray[float]


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(
            TestInput(value=5),
            TestOutput(expected=5.0),
            id="1) Convert integer scalar to float",
        ),
        pytest.param(
            TestInput(value=5.5),
            TestOutput(expected=5.5),
            id="2) Float scalar remains float",
        ),
        pytest.param(
            TestInput(value=True),
            TestOutput(expected=1.0),
            id="3) Convert boolean True to float",
        ),
        pytest.param(
            TestInput(value=False),
            TestOutput(expected=0.0),
            id="4) Convert boolean False to float",
        ),
        pytest.param(
            TestInput(value=np.int32(10)),
            TestOutput(expected=10.0),
            id="5) Convert numpy int32 to float",
        ),
        pytest.param(
            TestInput(value=np.int64(20)),
            TestOutput(expected=20.0),
            id="6) Convert numpy int64 to float",
        ),
        pytest.param(
            TestInput(value=np.float32(3.14)),
            TestOutput(expected=3.14),
            id="7) Numpy float32 becomes python float",
        ),
        pytest.param(
            TestInput(value=np.float64(2.718)),
            TestOutput(expected=2.718),
            id="8) Numpy float64 becomes python float",
        ),
        pytest.param(
            TestInput(value=np.bool_(True)),
            TestOutput(expected=1.0),
            id="9) Convert numpy bool True to float",
        ),
        pytest.param(
            TestInput(value=np.bool_(False)),
            TestOutput(expected=0.0),
            id="10) Convert numpy bool False to float",
        ),
        pytest.param(
            TestInput(value=np.array([1, 2, 3], dtype=int)),
            TestOutput(expected=np.array([1.0, 2.0, 3.0], dtype=float)),
            id="11) Convert integer array to float array",
        ),
        pytest.param(
            TestInput(value=np.array([1.1, 2.2, 3.3], dtype=float)),
            TestOutput(expected=np.array([1.1, 2.2, 3.3], dtype=float)),
            id="12) Float array remains unchanged",
        ),
        pytest.param(
            TestInput(value=np.array([True, False, True], dtype=bool)),
            TestOutput(expected=np.array([1.0, 0.0, 1.0], dtype=float)),
            id="13) Convert boolean array to float array",
        ),
        pytest.param(
            TestInput(value=np.array([1, 2, 3], dtype=np.int32)),
            TestOutput(expected=np.array([1.0, 2.0, 3.0], dtype=float)),
            id="14) Convert int32 array to float array",
        ),
        pytest.param(
            TestInput(value=np.array([1, 2, 3], dtype=np.int64)),
            TestOutput(expected=np.array([1.0, 2.0, 3.0], dtype=float)),
            id="15) Convert int64 array to float array",
        ),
        pytest.param(
            TestInput(value=np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            TestOutput(expected=np.array([1.0, 2.0, 3.0], dtype=float)),
            id="16) Float32 array remains unchanged",
        ),
        pytest.param(
            TestInput(value=np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            TestOutput(expected=np.array([1.0, 2.0, 3.0], dtype=float)),
            id="17) Float64 array remains unchanged",
        ),
        pytest.param(
            TestInput(value=np.array([], dtype=int)),
            TestOutput(expected=np.array([], dtype=float)),
            id="18) Convert empty integer array to float array",
        ),
        pytest.param(
            TestInput(value=np.array([0], dtype=int)),
            TestOutput(expected=np.array([0.0], dtype=float)),
            id="19) Convert single element integer array to float array",
        ),
        pytest.param(
            TestInput(value=np.array([np.array([1, 2]), np.array([3])], dtype=object)),
            TestOutput(
                expected=np.array([np.array([1.0, 2.0]), np.array([3.0])], dtype=object)
            ),
            id="20) Convert nested object array with integer inner arrays",
        ),
        pytest.param(
            TestInput(
                value=np.array([np.array([1.5, 2.5]), np.array([3])], dtype=object)
            ),
            TestOutput(
                expected=np.array([np.array([1.5, 2.5]), np.array([3.0])], dtype=object)
            ),
            id="21) Convert nested object array with mixed numeric inner arrays",
        ),
        pytest.param(
            TestInput(value=np.array([np.array([1, 2]), np.array([3])], dtype=object)),
            TestOutput(
                expected=np.array([np.array([1.0, 2.0]), np.array([3.0])], dtype=object)
            ),
            id="20) Convert nested object array with integer inner arrays",
        ),
    ],
)
def test__to_float__unit_test(inputs: TestInput, expected: TestOutput):
    """
    Test _to_float static method converts various numeric types to float:

    * Scalar conversions: int, float, bool convert to float correctly
    * Numpy scalar conversions: np.int32, np.int64, np.float32, np.float64, np.bool_ convert correctly
    * Array conversions: integer, float, and boolean arrays convert to float arrays
    * Special cases: empty arrays, single element arrays, and multidimensional arrays work correctly
    * Float types: float and float arrays remain unchanged (identity operation)
    """
    result = AbstractOptimizationSolver._to_float(inputs.value)

    if isinstance(expected.expected, np.ndarray):
        if result.size > 0 and isinstance(result[0], np.ndarray):
            for result_inner, expected_inner in zip(result, expected.expected):
                np.testing.assert_allclose(result_inner, expected_inner, atol=1e-7)
        else:
            np.testing.assert_allclose(result, expected.expected, atol=1e-7)
            assert np.issubdtype(result.dtype, float)
    else:
        assert result == pytest.approx(expected.expected)
        assert isinstance(result, float)


@pytest.mark.parametrize(
    "invalid_input",
    [
        pytest.param("string_value", id="1) String value raises TypeError"),
        pytest.param("1.0", id="2) String representation of a number raises TypeError"),
        pytest.param(np.array(["a", "b", "c"]), id="3) String array raises TypeError"),
        pytest.param(None, id="4) None value raises TypeError"),
    ],
)
def test__to_float__invalid_inputs(invalid_input):
    """
    Test _to_float raises TypeError for invalid inputs:

    * String values are not convertible
    * String arrays are not convertible
    * None values are not convertible
    """
    with pytest.raises(TypeError):
        AbstractOptimizationSolver._to_float(invalid_input)
