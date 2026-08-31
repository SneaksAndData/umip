"""DataFrame-backed example of the generic-mip framework.

Model:

    maximize x + 4y

    subject to:
        x + y <= 100
        y <= 20
        0 <= x, y <= 100, integers

This example is almost identical to simple_example.py. The only difference is in the
variable builder, which uses build_column_variables and unpack_column_variables instead
of manually creating and unpacking solver variable objects.
"""
from dataclasses import dataclass
import sys
from typing import Any

import numpy as np
import pandas as pd
from adapta.logs import LoggerInterface, SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel

from generic_mip import AbstractConstraintBuilder
from generic_mip import AbstractDataPreparator
from generic_mip import AbstractDecisionVariableBuilder
from generic_mip import AbstractMipModel
from generic_mip import AbstractObjectiveBuilder
from generic_mip import AbstractOptimizationSolver
from generic_mip import VariableDomain
from umip import AbstractInputData
from umip import AbstractInternalData
from umip import AbstractOutputData
from umip import SolverType
from umip import SolverFactory

VARIABLE_NAME = "variable_name"
UPPER_BOUND = "upper_bound"
VAR = "var"
VALUE = "value"


@dataclass
class ExampleInputData(AbstractInputData):
    my_data: pd.DataFrame


@dataclass
class ExampleInternalData(AbstractInternalData):
    my_data: pd.DataFrame


@dataclass
class ExampleOutputData(AbstractOutputData):
    my_data: pd.DataFrame


class ExampleDataPreparator(AbstractDataPreparator):
    """
    Data preparator for example, that does nothing.
    """

    def prepare(self, input_data: ExampleInputData) -> ExampleInternalData:
        return ExampleInternalData(my_data=input_data.my_data)


class ExampleVariableBuilder(AbstractDecisionVariableBuilder):
    """
    Creates variables x and y using build_column_variables, stored as a column in my_data.
    Per-row upper bounds are read from the UPPER_BOUND column in my_data.
    After solving, unpack_column_variables replaces the solver variable objects with solved values.
    """

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> ExampleInternalData:
        data.my_data = self.build_column_variables(
            solver=solver,
            data=data.my_data,
            destination_column=VAR,
            variable_domain=VariableDomain.INTEGER,
            index_name_columns=[VARIABLE_NAME],
            lower_bound=0.0,
            upper_bound=UPPER_BOUND,
        )
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> ExampleInternalData:
        data.my_data = self.unpack_column_variables(
            data=data.my_data,
            decision_variable_column=VAR,
            decision_variable_value_column=VALUE,
            solver=solver,
            variable_domain=VariableDomain.INTEGER,
        )
        return data


class ExampleCapacityConstraintBuilder(AbstractConstraintBuilder):
    """Adds the joint capacity constraint: x + y <= 100."""

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> None:
        solver.add_constraint(
            coefficients=np.ones(len(data.my_data)),
            variables=data.my_data[VAR].to_numpy(),
            lower_bound=None,
            upper_bound=100,
            name="capacity",
        )


class ExampleYCapConstraintBuilder(AbstractConstraintBuilder):
    """Adds the y upper bound constraint: y <= 20."""

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> None:
        y_row = data.my_data[data.my_data[VARIABLE_NAME] == "y"]
        solver.add_constraint(
            coefficients=np.ones(len(y_row)),
            variables=y_row[VAR].to_numpy(),
            lower_bound=None,
            upper_bound=20,
            name="y_cap",
        )


class ExampleObjectiveBuilder(AbstractObjectiveBuilder):
    """
    Adds objective term: maximize x + 4y and adds granularity analytics.
    """

    def __init__(self, logger: LoggerInterface) -> None:
        super().__init__(logger=logger)
        self.objective_name = "maximize_x_plus_4y"
        self.add_analytics_granularity(
            granularity_name="variable",
            analytics_calculator=self._variable_analytics,
        )
        self.add_analytics_granularity(
            granularity_name="total",
            analytics_calculator=self._total_analytics,
        )

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> None:
        solver.add_multiple_objective_terms(
            coefficients=np.array([1.0, 4.0]),
            variables=data.my_data[VAR].to_numpy(),
        )

    def _variable_analytics(self, analytics_data: ExampleOutputData) -> pd.DataFrame:
        return analytics_data.my_data[[VARIABLE_NAME, VALUE]]

    def _total_analytics(self, analytics_data: ExampleOutputData) -> float:
        variable_analytics = self.get_analytics(granularity="variable", analytics_data=analytics_data)
        return float(variable_analytics[VALUE].sum())


class ExampleMipModel(AbstractMipModel):
    """Concrete model for mip DataFrame example."""

    def build(self, input_data: ExampleInputData, redirect_solver_log: bool = True, **kwargs: Any) -> None:
        super().build(input_data=input_data, redirect_solver_log=redirect_solver_log, **kwargs)
        self._solver.set_optimization_direction(maximization=True)

    def _convert_internal_to_output_data(
        self, internal_unpacked_data: ExampleInternalData, **kwargs: Any
    ) -> ExampleOutputData:
        return ExampleOutputData(my_data=internal_unpacked_data.my_data)


logger = SemanticLogger().add_log_source(
    log_source_name="DataFrameExample",
    min_log_level=LogLevel.INFO,
    log_handlers=[SafeStreamHandler(sys.stdout)],
    is_default=True,
)

model = ExampleMipModel(
    solver=SolverFactory(logger=logger).construct(solver_type=SolverType.ORTOOLS_SCIP),
    data_preparator=ExampleDataPreparator(logger=logger),
    variable_builders=[ExampleVariableBuilder(logger=logger)],
    constraint_builders=[
        ExampleCapacityConstraintBuilder(logger=logger),
        ExampleYCapConstraintBuilder(logger=logger),
    ],
    objective_builders=[ExampleObjectiveBuilder(logger=logger)],
    logger=logger,
)

model.build(
    input_data=ExampleInputData(
        my_data=pd.DataFrame(
            {
                VARIABLE_NAME: ["x", "y"],
                UPPER_BOUND: [100.0, 100.0],
            }
        )
    )
)
model.solve()

output = model.get_output_data()
print(output.my_data[[VARIABLE_NAME, VALUE]])
print(model.get_analytics(granularity="variable"))
print(model.get_analytics(granularity="total"))
