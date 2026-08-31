"""Unit tests for OR-Tools solver configuration dataclasses."""

from dataclasses import dataclass

import pytest

from umip.solver_config import OrToolsScipSolverConfig


@dataclass
class InputTest:
    config: OrToolsScipSolverConfig


@dataclass
class OutputTest:
    expected_parameter_string: str


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(
            InputTest(config=OrToolsScipSolverConfig()),
            OutputTest(expected_parameter_string=""),
            id="1) Returns empty string when no fields are set",
        ),
        pytest.param(
            InputTest(config=OrToolsScipSolverConfig(branching_prefer_binary=True)),
            OutputTest(expected_parameter_string="branching/preferbinary = TRUE"),
            id="2) Emits boolean field as uppercase TRUE/FALSE",
        ),
        pytest.param(
            InputTest(config=OrToolsScipSolverConfig(limits_gap=0.05)),
            OutputTest(expected_parameter_string="limits/gap = 0.05"),
            id="3) Emits limits gap as a SCIP limits parameter",
        ),
        pytest.param(
            InputTest(
                config=OrToolsScipSolverConfig(
                    numerics_epsilon=1e-10,
                    numerics_feastol=1e-9,
                    presolving_max_rounds=0,
                    limits_gap=0.01,
                    branching_prefer_binary=False,
                )
            ),
            OutputTest(
                expected_parameter_string=(
                    "numerics/epsilon = 1e-10\n"
                    "numerics/feastol = 1e-09\n"
                    "presolving/maxrounds = 0\n"
                    "limits/gap = 0.01\n"
                    "branching/preferbinary = FALSE"
                )
            ),
            id="4) Combines float, int and bool fields in declaration order",
        ),
    ],
)
def test__OrToolsScipSolverConfig__to_scip_parameters_string__unit_test(
    inputs: InputTest, expected: OutputTest
) -> None:
    """
    Test OrToolsScipSolverConfig.to_scip_parameters_string logic:

    * 1) Empty config produces an empty string (no settings passed to SCIP).
    * 2) Boolean field is emitted as uppercase TRUE or FALSE.
    * 3) Limits gap is emitted with the SCIP limits/gap key.
    * 4) Mixed float, int, and bool fields are combined in declaration order.
    """
    # Act
    result = inputs.config.to_scip_parameters_string()

    # Assert
    assert result == expected.expected_parameter_string
