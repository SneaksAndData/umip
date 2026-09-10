"""KNAPSACK Problem using the umip framework.

Model:

    maximize sum(Profit*x)

    subject to:
        sum(Volume*x) <= CAPACITY
        x, binary

"""

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

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from adapta.logs import LoggerInterface, SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel

from umip import (
    AbstractConstraintBuilder,
    AbstractDataPreparator,
    AbstractDecisionVariableBuilder,
    AbstractMipModel,
    AbstractObjectiveBuilder,
    AbstractOptimizationSolver,
    VariableDomain,
)
from umip.abstract_dataclasses import (
    AbstractInputData,
    AbstractInternalData,
    AbstractOutputData,
)
from umip.enums import SolverType
from umip.solver_factory import SolverFactory

VARIABLE_NAME = "variable_name"
VAR = "var"
VALUE = "value"


@dataclass
class KnapsackInputData(AbstractInputData):
    knapsack_data: pl.DataFrame


@dataclass
class KnapsackInternalData(AbstractInternalData):
    knapsack_data: pl.DataFrame


@dataclass
class KnapsackOutputData(AbstractOutputData):
    knapsack_data: pl.DataFrame


class KnapsackDataPreparator(AbstractDataPreparator):
    """
    Data preparator for example, that does nothing.
    """

    def prepare(self, input_data: KnapsackInputData) -> KnapsackInternalData:
        return KnapsackInternalData(knapsack_data=input_data.knapsack_data)


class KnapsackVariableBuilder(AbstractDecisionVariableBuilder):
    """
    Creates variable x using build_column_variables, stored as a column in knapsack_data.
    After solving, unpack_column_variables replaces the solver variable objects with solved values.
    """

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> KnapsackInternalData:
        data.knapsack_data = self.build_column_variables(
            solver=solver,
            data=data.knapsack_data,
            destination_column=VAR,
            variable_domain=VariableDomain.BINARY,
            index_name_columns=[VARIABLE_NAME],
        )
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> KnapsackInternalData:
        data.knapsack_data = self.unpack_column_variables(
            data=data.knapsack_data,
            decision_variable_column=VAR,
            decision_variable_value_column=VALUE,
            solver=solver,
            variable_domain=VariableDomain.BINARY,
        )
        return data


class KnapsackCapacityConstraintBuilder(AbstractConstraintBuilder):
    """Adds the joint capacity constraint: sum(x) <= CAPACITY"""

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        solver.add_constraint(
            coefficients=data.knapsack_data["Volume"].to_numpy(),
            variables=data.knapsack_data[VAR].to_numpy(),
            lower_bound=None,
            upper_bound=20,
            name="capacity",
        )


class KnapsackLargeSmallGapConstraintBuilder(AbstractConstraintBuilder):
    """Adds the large-small difference gap constraint."""

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        y_row = data.knapsack_data.filter(pl.col(VARIABLE_NAME) == "y")
        solver.add_constraint(
            coefficients=np.ones(len(y_row)),
            variables=y_row[VAR].to_numpy(),
            lower_bound=None,
            upper_bound=None,
            name="gap_cap",
        )

class KnapsackSameVolumeTwiceConstraintBuilder(AbstractConstraintBuilder):
    """Adds the at least two items with same volume constraint."""

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        y_row = data.knapsack_data.filter(pl.col(VARIABLE_NAME) == "y")
        solver.add_constraint(
            coefficients=np.ones(len(y_row)),
            variables=y_row[VAR].to_numpy(),
            lower_bound=2,
            upper_bound=None,
            name="duplicate_vol",
        )

class KnapsackObjectiveBuilder(AbstractObjectiveBuilder):
    """
    Adds objective term: maximize sum(profit*x) and adds granularity analytics.
    """

    def __init__(self, logger: LoggerInterface) -> None:
        super().__init__(logger=logger)
        self.objective_name = "maximize_profit"
        self.add_analytics_granularity(
            granularity_name="variable",
            analytics_calculator=self._variable_analytics,
        )
        self.add_analytics_granularity(
            granularity_name="total",
            analytics_calculator=self._total_analytics,
        )

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        solver.add_multiple_objective_terms(
            coefficients=data.knapsack_data["Profit"].to_numpy(),
            variables=data.knapsack_data[VAR].to_numpy(),
        )

    def _variable_analytics(self, analytics_data: KnapsackOutputData) -> pl.DataFrame:
        return analytics_data.knapsack_data[[VARIABLE_NAME, VALUE]]

    def _total_analytics(self, analytics_data: KnapsackOutputData) -> float:
        variable_analytics = self.get_analytics(granularity="variable", analytics_data=analytics_data)
        return float(variable_analytics[VALUE].sum())


class KnapsackMipModel(AbstractMipModel):
    """Concrete model for mip DataFrame example."""

    def build(
        self,
        input_data: KnapsackInputData,
        redirect_solver_log: bool = True,
        **kwargs: Any,
    ) -> None:
        super().build(input_data=input_data, redirect_solver_log=redirect_solver_log, **kwargs)
        self._solver.set_optimization_direction(maximization=True)

    def _convert_internal_to_output_data(
        self, internal_unpacked_data: KnapsackInternalData, **kwargs: Any
    ) -> KnapsackOutputData:
        return KnapsackOutputData(knapsack_data=internal_unpacked_data.knapsack_data)


logger = SemanticLogger().add_log_source(
    log_source_name="DataFrameKnapsack",
    min_log_level=LogLevel.INFO,
    log_handlers=[SafeStreamHandler(sys.stdout)],
    is_default=True,
)

model = KnapsackMipModel(
    solver=SolverFactory(logger=logger).construct(solver_type=SolverType.ORTOOLS_SCIP),
    data_preparator=KnapsackDataPreparator(logger=logger),
    variable_builders=[KnapsackVariableBuilder(logger=logger)],
    constraint_builders=[
        KnapsackCapacityConstraintBuilder(logger=logger),
        #KnapsackLargeSmallGapConstraintBuilder(logger=logger),
        #KnapsackSameVolumeTwiceConstraintBuilder(logger=logger),
    ],
    objective_builders=[KnapsackObjectiveBuilder(logger=logger)],
    logger=logger,
)

model.build(
    input_data=KnapsackInputData(
        knapsack_data=(
            pl.read_parquet("data/knapsack.parquet")
            .with_row_index("item_id")
            .with_columns(
                pl.col("item_id").cast(pl.String).alias(VARIABLE_NAME),
            )
        )
    )
)
model.solve()

output = model.get_output_data()
print(output.knapsack_data[[VARIABLE_NAME, VALUE]])
print(model.get_analytics(granularity="variable"))
print(model.get_analytics(granularity="total"))

selected = output.knapsack_data.filter(pl.col(VALUE) > 0.5)

print(selected)

print(
    selected["Profit"].sum(),
    selected["Volume"].sum(),
)
