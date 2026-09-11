"""
KNAPSACK Problem using the umip framework.

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
import plotly.express as px
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
SAME_VOLUME_PAIR = "same_volume_pair"
VAR = "var"
Y_VAR = "y_var"
VALUE = "value"
Y_VALUE = "y_value"


@dataclass
class KnapsackInputData(AbstractInputData):
    knapsack_data: pl.DataFrame


@dataclass
class KnapsackInternalData(AbstractInternalData):
    knapsack_data: pl.DataFrame
    same_volume_pair: pl.DataFrame


@dataclass
class KnapsackOutputData(AbstractOutputData):
    knapsack_data: pl.DataFrame


class KnapsackDataPreparator(AbstractDataPreparator):
    """
    Data preparator for the pair dataframe.
    """

    def prepare(self, input_data: KnapsackInputData) -> KnapsackInternalData:
        knapsack_data = input_data.knapsack_data

        same_volume_pair = (
            knapsack_data.with_row_index("i")
            .join(
                knapsack_data.with_row_index("j"),
                how="cross",
                suffix="_j",
            )
            .filter(pl.col("i") != pl.col("j"))
            .with_columns(
                pl.concat_str(
                    [
                        pl.lit("y_"),
                        pl.col("i").cast(pl.String),
                        pl.lit("_"),
                        pl.col("j").cast(pl.String),
                    ]
                ).alias(SAME_VOLUME_PAIR)
            )
        )

        return KnapsackInternalData(
            knapsack_data=knapsack_data,
            same_volume_pair=same_volume_pair,
        )


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


class KnapsackSameVolumePairsVariableBuilder(AbstractDecisionVariableBuilder):
    """
    Creates y_i,j variables for all distinct item pairs.
    After solving, unpack_column_variables replaces the solver variable objects with solved values.
    """

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> KnapsackInternalData:
        if data.same_volume_pair.is_empty():
            raise ValueError("No same-volume pairs found in the data.")

        data.same_volume_pair = self.build_column_variables(
            solver=solver,
            data=data.same_volume_pair,
            destination_column=Y_VAR,
            variable_domain=VariableDomain.BINARY,
            index_name_columns=[SAME_VOLUME_PAIR],
        )
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> KnapsackInternalData:
        data.same_volume_pair = self.unpack_column_variables(
            data=data.same_volume_pair,
            decision_variable_column=Y_VAR,
            decision_variable_value_column=Y_VALUE,
            solver=solver,
            variable_domain=VariableDomain.BINARY,
        )
        return data


class KnapsackCapacityConstraintBuilder(AbstractConstraintBuilder):
    """
    Adds the joint capacity constraint: sum(x) <= CAPACITY
    """

    def __init__(self, logger: LoggerInterface, capacity: int) -> None:
        super().__init__(logger=logger)
        self.capacity = capacity

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        solver.add_constraint(
            coefficients=data.knapsack_data["Volume"].to_numpy(),
            variables=data.knapsack_data[VAR].to_numpy(),
            lower_bound=None,
            upper_bound=self.capacity,
            name="capacity",
        )


class KnapsackLargeSmallGapConstraintBuilder(AbstractConstraintBuilder):
    """
    Adds the large-small difference gap constraint.
    """

    def __init__(self, logger: LoggerInterface, max_gap: int) -> None:
        super().__init__(logger=logger)
        self.max_gap = max_gap

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        L = self.max_gap
        M = data.knapsack_data["Volume"].max()
        pairs = (
            data.knapsack_data.with_row_index("i")
            .join(
                data.knapsack_data.with_row_index("j"),
                how="cross",
                suffix="_j",
            )
            .filter(pl.col("i") != pl.col("j"))
        )
        solver.add_multiple_constraints(
            coefficients=np.column_stack(
                (
                    pairs["Volume"].to_numpy() + M,
                    M - pairs["Volume_j"].to_numpy(),
                )
            ),
            variables=np.column_stack(
                (
                    pairs[VAR].to_numpy(),
                    pairs[f"{VAR}_j"].to_numpy(),
                )
            ).astype(object),
            lower_bounds=None,
            upper_bounds=np.full(
                len(pairs),
                L + 2 * M,
            ),
            names=np.array(
                [f"gap_{i}" for i in range(len(pairs))],
            ),
        )


class KnapsackSameVolumePairsConstraintBuilder(AbstractConstraintBuilder):
    """
    Requires at least one selected pair with the same volume.
    """

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        pairs = data.same_volume_pair
        pair_count = len(pairs)

        if pair_count == 0:
            raise ValueError("No same-volume pairs found in the data.")

        M = data.knapsack_data["Volume"].max()
        item_variables = data.knapsack_data[VAR].to_numpy()

        x_i = item_variables[pairs["i"].to_numpy()]
        x_j = item_variables[pairs["j"].to_numpy()]
        y = pairs[Y_VAR].to_numpy()

        variables = np.column_stack((x_i, x_j, y)).astype(object)

        # x_i + x_j >= 2*y_i,j
        solver.add_multiple_constraints(
            coefficients=np.tile(
                np.array([-1, -1, 2]),
                (pair_count, 1),
            ),
            variables=variables,
            lower_bounds=None,
            upper_bounds=np.zeros(pair_count),
            names=np.array(
                [f"same_volume_link_{i}" for i in range(pair_count)],
            ),
        )

        # V_i*x_i - V_j*x_j <= M*(1-y_i,j)
        solver.add_multiple_constraints(
            coefficients=np.column_stack(
                (
                    pairs["Volume"].to_numpy(),
                    -pairs["Volume_j"].to_numpy(),
                    np.full(pair_count, M),
                )
            ),
            variables=variables,
            lower_bounds=None,
            upper_bounds=np.full(pair_count, M),
            names=np.array(
                [f"same_volume_forward_{i}" for i in range(pair_count)],
            ),
        )

        # - V_i*x_i + V_j*x_j <= M*(1-y_i,j)
        solver.add_multiple_constraints(
            coefficients=np.column_stack(
                (
                    -pairs["Volume"].to_numpy(),
                    pairs["Volume_j"].to_numpy(),
                    np.full(pair_count, M),
                )
            ),
            variables=variables,
            lower_bounds=None,
            upper_bounds=np.full(pair_count, M),
            names=np.array(
                [f"same_volume_reverse_{i}" for i in range(pair_count)],
            ),
        )

        # sum(y_i,j) >= 1
        solver.add_constraint(
            coefficients=np.ones(pair_count),
            variables=y,
            lower_bound=1,
            upper_bound=None,
            name="at_least_one_same_volume_pair",
        )


class KnapsackObjectiveBuilder(AbstractObjectiveBuilder):
    """
    Adds objective term: maximize profit and adds granularity analytics.
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
    """
    Concrete model for mip knapsack problem.
    """

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


def build_model(capacity: int, max_gap: int) -> KnapsackMipModel:
    solver = SolverFactory(logger=logger).construct(solver_type=SolverType.ORTOOLS_SCIP)

    return KnapsackMipModel(
        solver=solver,
        data_preparator=KnapsackDataPreparator(logger=logger),
        variable_builders=[
            KnapsackVariableBuilder(logger=logger),
            KnapsackSameVolumePairsVariableBuilder(logger=logger),
        ],
        constraint_builders=[
            KnapsackCapacityConstraintBuilder(logger=logger, capacity=capacity),
            KnapsackLargeSmallGapConstraintBuilder(logger=logger, max_gap=max_gap),
            KnapsackSameVolumePairsConstraintBuilder(logger=logger),
        ],
        objective_builders=[KnapsackObjectiveBuilder(logger=logger)],
        logger=logger,
    )


input_data = KnapsackInputData(
    knapsack_data=(
        pl.read_parquet("data/knapsack.parquet")
        .with_row_index("item_id")
        .with_columns(
            pl.col("item_id").cast(pl.String).alias(VARIABLE_NAME),
        )
    )
)

"""Scenarios to test maximum capacity"""
# scenarios = [
#     {"capacity": 10, "max_gap": 2},
#     {"capacity": 15, "max_gap": 2},
#     {"capacity": 20, "max_gap": 2},
#     {"capacity": 25, "max_gap": 2},
#     {"capacity": 30, "max_gap": 2},
#     {"capacity": 35, "max_gap": 2},
#     {"capacity": 40, "max_gap": 2},
#     {"capacity": 45, "max_gap": 2},
#     {"capacity": 50, "max_gap": 2},
#     {"capacity": 55, "max_gap": 2},
#     {"capacity": 60, "max_gap": 2},
#     {"capacity": 65, "max_gap": 2},
#     {"capacity": 70, "max_gap": 2},
# ]

"""Scenarios to test volume difference L"""
scenarios = [
    {"capacity": 100, "max_gap": 1},
    {"capacity": 100, "max_gap": 2},
    {"capacity": 100, "max_gap": 3},
    {"capacity": 100, "max_gap": 4},
    {"capacity": 100, "max_gap": 5},
    {"capacity": 100, "max_gap": 6},
]

results = []

for scenario in scenarios:
    capacity = scenario["capacity"]
    max_gap = scenario["max_gap"]

    model = build_model(
        capacity=capacity,
        max_gap=max_gap,
    )

    model.build(
        input_data=KnapsackInputData(
            knapsack_data=input_data.knapsack_data.clone(),
        ),
        redirect_solver_log=False,
    )
    model.solve()

    output = model.get_output_data()
    selected = output.knapsack_data.filter(pl.col(VALUE))

    result = {
        "capacity": capacity,
        "max_gap": max_gap,
        "profit": selected["Profit"].sum(),
        "volume": selected["Volume"].sum(),
        "selected_items": selected.height,
    }
    results.append(result)

    print(f"\nScenario: capacity={capacity}, max_gap={max_gap}")
    print(selected)
    print(f"Profit: {result['profit']}")
    print(f"Volume: {result['volume']}")
    print(f"Selected items: {result['selected_items']}")

print("\nSummary:")
print(pl.DataFrame(results))

results_plots = pl.DataFrame(results)

"""Plot Profit vs Capacity"""
profit_by_capacity = results_plots.group_by("capacity").agg(pl.col("profit").max()).sort("capacity")

fig = px.line(
    x=profit_by_capacity["capacity"].to_list(),
    y=profit_by_capacity["profit"].to_list(),
    markers=True,
    labels={"x": "Capacity (liters)", "y": "Profit (€)"},
    title="Profit vs. Capacity",
)
fig.show()

"""Plot Profit vs L"""
profit_by_gap = results_plots.group_by("max_gap").agg(pl.col("profit").max()).sort("max_gap")

fig = px.line(
    x=profit_by_gap["max_gap"].to_list(),
    y=profit_by_gap["profit"].to_list(),
    markers=True,
    labels={"x": "Maximum volume gap (L)", "y": "Profit (€)"},
    title="Profit vs. L",
)
fig.show()

"""Heatmap Profit for Capacity vs L"""
fig = px.density_heatmap(
    x=results_plots["capacity"].to_list(),
    y=results_plots["max_gap"].to_list(),
    z=results_plots["profit"].to_list(),
    histfunc="max",
    text_auto=True,
    labels={
        "x": "Capacity (liters)",
        "y": "Maximum volume gap (L)",
        "color": "Profit (€)",
    },
    title="Profit by Capacity and L",
)
fig.show()
