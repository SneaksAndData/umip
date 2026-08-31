"""Unit tests for HiGHS solver configuration dataclasses."""

from dataclasses import dataclass

import pytest

from umip.solver_config import (
    HighsParallelOption,
    HighsPresolveOption,
    HighsSolverConfig,
    HighsSolverOption,
)


@dataclass
class InputHighsTest:
    config: HighsSolverConfig


@dataclass
class OutputHighsTest:
    expected_options: dict[str, bool | int | float | str]


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(
            InputHighsTest(config=HighsSolverConfig()),
            OutputHighsTest(expected_options={}),
            id="1) Returns empty options for empty config",
        ),
        pytest.param(
            InputHighsTest(config=HighsSolverConfig(threads=4, random_seed=7)),
            OutputHighsTest(
                expected_options={
                    "threads": 4,
                    "random_seed": 7,
                }
            ),
            id="2) Includes numeric threading and seed options",
        ),
        pytest.param(
            InputHighsTest(
                config=HighsSolverConfig(
                    presolve=HighsPresolveOption.OFF,
                    solver=HighsSolverOption.IPM,
                    parallel=HighsParallelOption.ON,
                    random_seed=7,
                )
            ),
            OutputHighsTest(
                expected_options={
                    "random_seed": 7,
                    "presolve": "off",
                    "solver": "ipm",
                    "parallel": "on",
                }
            ),
            id="3) Includes enum mode and seed options",
        ),
        pytest.param(
            InputHighsTest(
                config=HighsSolverConfig(
                    presolve=HighsPresolveOption.ON,
                    solver=HighsSolverOption.SIMPLEX,
                    parallel=HighsParallelOption.CHOOSE,
                )
            ),
            OutputHighsTest(
                expected_options={
                    "presolve": "on",
                    "solver": "simplex",
                    "parallel": "choose",
                }
            ),
            id="4) Converts enum mode options to HiGHS strings",
        ),
    ],
)
def test__HighsSolverConfig__to_highs_options__unit_test(
    inputs: InputHighsTest, expected: OutputHighsTest
) -> None:
    """
    Test HighsSolverConfig.to_highs_options logic:

    * 1) Empty config yields no option entries.
    * 2) Numeric HiGHS knobs are mapped with expected keys.
    * 3) Enum mode knobs and seed are mapped with expected keys.
    * 4) Enum mode knobs are converted to expected HiGHS strings.
    """
    # Act
    result = inputs.config.to_highs_options()

    # Assert
    assert result == expected.expected_options
