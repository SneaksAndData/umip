from dataclasses import dataclass, field
from typing import Generic, TypeVar

VT = TypeVar("VT")


@dataclass
class VariableWithObjectiveCoefficient(Generic[VT]):
    """
    Defines a model for a variable and an objective coefficient
    """

    variable: VT = field(
        metadata={
            "display_name": "Variable",
            "description": "The decision variable",
        },
    )
    objective_coefficient: float = field(
        metadata={
            "display_name": "Objective Coefficient",
            "description": "The objective coefficient",
        },
    )
