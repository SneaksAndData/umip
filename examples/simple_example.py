"""Simple end-to-end example of the generic-mip framework.

Model:

    maximize x + 4y

    subject to:
        x + y <= 100
        y <= 20
        0 <= x, y <= 100, integers

For a DataFrame-backed version using build_column_variables and unpack_column_variables,
see dataframe_example.py.
"""

from dataclasses import dataclass
import sys
from typing import Any

import numpy as np
from adapta.logs import LoggerInterface, SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel

from umip import AbstractConstraintBuilder
from umip import AbstractDataPreparator
from umip import AbstractDecisionVariableBuilder
from umip import AbstractMipModel
from umip import AbstractObjectiveBuilder
from umip import AbstractOptimizationSolver
from umip import VariableDomain
from umip.abstract_dataclasses import AbstractInputData
from umip.abstract_dataclasses import AbstractInternalData
from umip.abstract_dataclasses import AbstractOutputData
from umip.enums import SolverType
from umip.solver_factory import SolverFactory


@dataclass
class ExampleInputData(AbstractInputData):
    """Input payload. No input data required for this minimal example."""


@dataclass
class ExampleInternalData(AbstractInternalData):
    """Internal model state holding solver variable objects and their solved values.

    :attr x_var: Solver variable object for x.
    :attr y_var: Solver variable object for y.
    :attr x_value: Solved value of x, populated after solve.
    :attr y_value: Solved value of y, populated after solve.
    """

    x_var: Any | None = None
    y_var: Any | None = None
    x_value: float | None = None
    y_value: float | None = None


@dataclass
class ExampleOutputData(AbstractOutputData):
    """Output payload with solved variable values.

    :attr x_value: Solved value of x.
    :attr y_value: Solved value of y.
    """

    x_value: float
    y_value: float


class ExampleDataPreparator(AbstractDataPreparator):
    """
    Data preparator for example, that does nothing.
    """

    def prepare(self, input_data: ExampleInputData) -> ExampleInternalData:
        _ = input_data
        return ExampleInternalData()


class ExampleVariableBuilder(AbstractDecisionVariableBuilder):
    """Creates variables x and y and stores them as fields on ExampleInternalData."""

    def build(
        self, solver: AbstractOptimizationSolver, data: ExampleInternalData
    ) -> ExampleInternalData:
        data.x_var = solver.add_variable(
            name="x",
            variable_domain=VariableDomain.INTEGER,
            lower_bound=0,
            upper_bound=100,
        )
        data.y_var = solver.add_variable(
            name="y",
            variable_domain=VariableDomain.INTEGER,
            lower_bound=0,
            upper_bound=100,
        )
        return data

    def unpack(
        self, solver: AbstractOptimizationSolver, data: ExampleInternalData
    ) -> ExampleInternalData:
        data.x_value = solver.get_variable_value(var=data.x_var)
        data.y_value = solver.get_variable_value(var=data.y_var)
        return data


class ExampleCapacityConstraintBuilder(AbstractConstraintBuilder):
    """Adds the joint capacity constraint: x + y <= 100."""

    def build(
        self, solver: AbstractOptimizationSolver, data: ExampleInternalData
    ) -> None:
        solver.add_constraint(
            coefficients=np.array([1.0, 1.0]),
            variables=np.array([data.x_var, data.y_var]),
            lower_bound=None,
            upper_bound=100,
            name="capacity",
        )


class ExampleYCapConstraintBuilder(AbstractConstraintBuilder):
    """Adds the y upper bound constraint: y <= 20."""

    def build(
        self, solver: AbstractOptimizationSolver, data: ExampleInternalData
    ) -> None:
        solver.add_constraint(
            coefficients=np.array([1.0]),
            variables=np.array([data.y_var]),
            lower_bound=None,
            upper_bound=20,
            name="y_cap",
        )


class ExampleObjectiveBuilder(AbstractObjectiveBuilder):
    """Adds objective terms: maximize x + 4y."""

    def __init__(self, logger: LoggerInterface) -> None:
        super().__init__(logger=logger)
        self.objective_name = "maximize_x_plus_4y"
        self.add_analytics_granularity(
            granularity_name="total",
            analytics_calculator=self._total_analytics,
        )

    def build(
        self, solver: AbstractOptimizationSolver, data: ExampleInternalData
    ) -> None:
        solver.add_multiple_objective_terms(
            coefficients=np.array([1.0, 4.0]),
            variables=np.array([data.x_var, data.y_var]),
        )

    def _total_analytics(self, analytics_data: ExampleOutputData) -> float:
        return analytics_data.x_value * 1.0 + analytics_data.y_value * 4.0


class ExampleMipModel(AbstractMipModel):
    """Concrete model for the simple example."""

    def build(
        self,
        input_data: ExampleInputData,
        redirect_solver_log: bool = True,
        **kwargs: Any,
    ) -> None:
        super().build(
            input_data=input_data, redirect_solver_log=redirect_solver_log, **kwargs
        )
        self._solver.set_optimization_direction(maximization=True)

    def _convert_internal_to_output_data(
        self, internal_unpacked_data: ExampleInternalData, **kwargs: Any
    ) -> ExampleOutputData:
        _ = kwargs
        return ExampleOutputData(
            x_value=internal_unpacked_data.x_value,
            y_value=internal_unpacked_data.y_value,
        )


logger = SemanticLogger().add_log_source(
    log_source_name="SimpleExample",
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

model.build(input_data=ExampleInputData())
model.solve()

output = model.get_output_data()
print(f"x = {output.x_value}, y = {output.y_value}")
print(model.get_analytics(granularity="total"))
