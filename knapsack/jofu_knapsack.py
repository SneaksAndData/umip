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
import polars as pl
from adapta.logs import LoggerInterface, SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel

from umip import (
    AbstractConstraintBuilder,
    AbstractDataPreparator,
    AbstractDecisionVariableBuilder,
    AbstractMipModel,
    AbstractMipModelFactory,
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

ITEM_ID = "item_id"
ITEM_IS_SELECTED_VAR = "item_is_selected_var"
SAME_VOLUME_VAR = "same_volume_var"
ITEM_IS_SELECTED_VALUE = "item_is_selected_value"
SAME_VOLUME_VALUE = "same_volume_value"
VOLUME_COLUMN = "Volume"
PROFIT_COLUMN = "Profit"
VOLUME_I = "volume_i"
VOLUME_J = "volume_j"
INDEX_I = "index_i"
INDEX_J = "index_j"
VOLUMES_ARE_EQUAL = "volumes_are_equal"
OBJECTIVE_PROFIT_CONTRIBUTION = "objective_profit_contribution"


@dataclass(frozen=True)
class KnapsackSettings:
    enable_max_volume_difference: bool = False
    enable_at_least_one_same_volume_pair: bool = False


@dataclass
class KnapsackInputData(AbstractInputData):
    item: pl.DataFrame
    knapsack_capacity: int
    knapsack_volume_max_gap: int


@dataclass
class KnapsackInternalData(KnapsackInputData, AbstractInternalData):
    item_pair: pl.DataFrame | None = None


@dataclass
class KnapsackOutputData(KnapsackInternalData, AbstractOutputData):
    pass


class KnapsackDataPreparator(AbstractDataPreparator):
    """
    Data preparator for the pair dataframe.
    """

    def __init__(
        self,
        logger: LoggerInterface,
        settings: KnapsackSettings,
    ) -> None:
        super().__init__(logger=logger)
        self._settings = settings

    def prepare(self, input_data: KnapsackInputData) -> KnapsackInternalData:
        item = input_data.item

        if self._settings.enable_at_least_one_same_volume_pair or self._settings.enable_max_volume_difference:
            item_pair = (
                item.select(pl.col(VOLUME_COLUMN).alias(VOLUME_I))
                .with_row_index(INDEX_I)
                .join(
                    item.select(pl.col(VOLUME_COLUMN).alias(VOLUME_J)).with_row_index(INDEX_J),
                    how="cross",
                )
                .filter(pl.col(INDEX_I) != pl.col(INDEX_J))
                .with_columns(
                    (pl.col(VOLUME_I) == pl.col(VOLUME_J)).alias(VOLUMES_ARE_EQUAL),
                )
            )
        else:
            item_pair = None

        return KnapsackInternalData(
            item=item,
            item_pair=item_pair,
            knapsack_capacity=input_data.knapsack_capacity,
            knapsack_volume_max_gap=input_data.knapsack_volume_max_gap,
        )


class KnapsackIsSelectedVariableBuilder(AbstractDecisionVariableBuilder):
    """
    Creates variable x using build_column_variables, stored as a column in 'item'.
    After solving, unpack_column_variables replaces the solver variable objects with solved values.
    """

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> KnapsackInternalData:
        data.item = self.build_column_variables(
            solver=solver,
            data=data.item,
            destination_column=ITEM_IS_SELECTED_VAR,
            variable_domain=VariableDomain.BINARY,
            index_name_columns=[ITEM_ID],
            variable_name="x",
        )
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> KnapsackInternalData:
        data.item = self.unpack_column_variables(
            data=data.item,
            decision_variable_column=ITEM_IS_SELECTED_VAR,
            decision_variable_value_column=ITEM_IS_SELECTED_VALUE,
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
        if data.item_pair.is_empty():
            raise ValueError("No same-volume pairs found in the data.")

        data.item_pair = self.build_column_variables(
            solver=solver,
            data=data.item_pair,
            destination_column=SAME_VOLUME_VAR,
            variable_domain=VariableDomain.BINARY,
            index_name_columns=[INDEX_I, INDEX_J],
            variable_name="y",
        )
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> KnapsackInternalData:
        data.item_pair = self.unpack_column_variables(
            data=data.item_pair,
            decision_variable_column=SAME_VOLUME_VAR,
            decision_variable_value_column=SAME_VOLUME_VALUE,
            solver=solver,
            variable_domain=VariableDomain.BINARY,
        )
        return data


class KnapsackCapacityConstraintBuilder(AbstractConstraintBuilder):
    """
    Adds the joint capacity constraint: sum(x_i) <= C

    C: data.knapsack_capacity
    VOLUME_COLUMN: "Volume" column in the data, referred to V_i parameter in the formulation
    ITEM_IS_SELECTED_VAR: x_i variable
    """

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        solver.add_constraint(
            coefficients=data.item.get_column(VOLUME_COLUMN).to_numpy(),
            variables=data.item.get_column(ITEM_IS_SELECTED_VAR).to_numpy(),
            lower_bound=None,
            upper_bound=data.knapsack_capacity,
            name="capacity",
        )


class KnapsackLargeSmallGapConstraintBuilder(AbstractConstraintBuilder):
    """
    Class for adding the large-small difference gap constraint to the knapsack problem.
    """

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        """
        Adds the large-small difference gap constraint: V_i * x_i - V_j * x_j <= L + M * (2 - x_i - x_j)

        VOLUME_GAP_MAX: L
        BIG_M_MAX_VOLUME: M
        ITEM_IS_SELECTED_VAR: x_i and x_j
        VOLUME_COLUMN: V_i and V_j
        """
        if data.knapsack_volume_max_gap is None:
            raise ValueError("knapsack_volume_max_gap must be provided when the large-small gap constraint is enabled.")

        VOLUME_GAP_MAX = data.knapsack_volume_max_gap
        BIG_M_MAX_VOLUME = data.item.get_column(VOLUME_COLUMN).max()

        item_i_is_selected_var = "item_i_is_selected"
        item_j_is_selected_var = "item_j_is_selected"

        item_pairs_with_selected_variable = self._join_item_column_to_pairs(
            item_pair=data.item_pair,
            item=data.item,
            item_column=ITEM_IS_SELECTED_VAR,
            index_i_column=INDEX_I,
            index_j_column=INDEX_J,
            item_i_is_selected_var=item_i_is_selected_var,
            item_j_is_selected_var=item_j_is_selected_var,
        )

        variables = (
            item_pairs_with_selected_variable.select(item_i_is_selected_var, item_j_is_selected_var)
            .to_numpy()
            .astype(object)
        )
        solver.add_multiple_constraints(
            coefficients=np.column_stack(
                (
                    data.item_pair.get_column(VOLUME_I).to_numpy() + BIG_M_MAX_VOLUME,
                    BIG_M_MAX_VOLUME - data.item_pair.get_column(VOLUME_J).to_numpy(),
                )
            ),
            variables=variables,
            lower_bounds=None,
            upper_bounds=np.full(
                len(data.item_pair),
                VOLUME_GAP_MAX + 2 * BIG_M_MAX_VOLUME,
            ),
            names=np.array([f"gap_{i}" for i in range(len(data.item_pair))]),
        )


class KnapsackSameVolumePairsConstraintBuilder(AbstractConstraintBuilder):
    """
    Requires at least one selected pair with the same volume.
    Adds constraints to ensure that if two items have the same volume and are both selected, then the corresponding y_i,j variable is set to 1.
    """

    def build(self, solver: AbstractOptimizationSolver, data: KnapsackInternalData) -> None:
        """
        Data Preparation for the constraints to ensure that if two items have the same volume and are both selected, then the corresponding y_i,j variable is set to 1.
        """

        if data.item_pair.is_empty():
            raise ValueError("No same-volume pairs found in the data.")

        BIG_M_MAX_VOLUME = data.item.get_column(VOLUME_COLUMN).max()
        item_i_is_selected_var = "item_i_is_selected"
        item_j_is_selected_var = "item_j_is_selected"

        item_pairs_with_selected_variable = self._join_item_column_to_pairs(
            item_pair=data.item_pair,
            item=data.item,
            item_column=ITEM_IS_SELECTED_VAR,
            index_i_column=INDEX_I,
            index_j_column=INDEX_J,
            item_i_is_selected_var=item_i_is_selected_var,
            item_j_is_selected_var=item_j_is_selected_var,
        )

        variables = (
            item_pairs_with_selected_variable.select(item_i_is_selected_var, item_j_is_selected_var, SAME_VOLUME_VAR)
            .to_numpy()
            .astype(object)
        )

        self._build_variables_link_constraints(solver, variables, len(data.item_pair))
        self._build_forward_same_volume_constraints(
            solver, data.item_pair, variables, len(data.item_pair), BIG_M_MAX_VOLUME
        )
        self._build_reverse_same_volume_constraints(
            solver, data.item_pair, variables, len(data.item_pair), BIG_M_MAX_VOLUME
        )
        self._build_at_least_one_pair_constraint(
            solver, data.item_pair.get_column(SAME_VOLUME_VAR).to_numpy(), len(data.item_pair)
        )

    @staticmethod
    def _build_variables_link_constraints(
        solver: AbstractOptimizationSolver,
        variables: np.ndarray,
        pair_count: int,
    ) -> None:
        """
        Adds constraint that ensures y_i,j takes the value of 1 only when x_i = x_j = 1
        Constraint: x_i + x_j >= 2*y_i,j
        """
        solver.add_multiple_constraints(
            coefficients=np.tile(np.array([-1, -1, 2]), (pair_count, 1)),
            variables=variables,
            lower_bounds=None,
            upper_bounds=np.zeros(pair_count),
            names=np.array(
                [f"same_volume_link_{i}" for i in range(pair_count)],
            ),
        )

    @staticmethod
    def _build_forward_same_volume_constraints(
        solver: AbstractOptimizationSolver,
        item_pair: pl.DataFrame,
        variables: np.ndarray,
        pair_count: int,
        BIG_M_MAX_VOLUME: int,
    ) -> None:
        """
        Adds constraint that ensures volume between two items is zero when both are chosen and y_i,j =1
        Constraint: V_i*x_i - V_j*x_j <= M*(1-y_i,j)
        BIG_M_MAX_VOLUME: M, is the maximum volume of any item in the knapsack, used as a big M constant.
        """
        solver.add_multiple_constraints(
            coefficients=np.column_stack(
                (
                    item_pair.get_column(VOLUME_I).to_numpy(),
                    -item_pair.get_column(VOLUME_J).to_numpy(),
                    np.full(pair_count, BIG_M_MAX_VOLUME),
                )
            ),
            variables=variables,
            lower_bounds=None,
            upper_bounds=np.full(pair_count, BIG_M_MAX_VOLUME),
            names=np.array(
                [f"same_volume_forward_{i}" for i in range(pair_count)],
            ),
        )

    @staticmethod
    def _build_reverse_same_volume_constraints(
        solver: AbstractOptimizationSolver,
        item_pair: pl.DataFrame,
        variables: np.ndarray,
        pair_count: int,
        BIG_M_MAX_VOLUME: int,
    ) -> None:
        """
        Adds constraint that ensures volume between two items is zero when both are chosen and y_i,j =1
        Constraint: - V_i*x_i + V_j*x_j <= M*(1-y_i,j)
        BIG_M_MAX_VOLUME: M, is the maximum volume of any item in the knapsack, used as a big M constant.
        """
        solver.add_multiple_constraints(
            coefficients=np.column_stack(
                (
                    -item_pair.get_column(VOLUME_I).to_numpy(),
                    item_pair.get_column(VOLUME_I).to_numpy(),
                    np.full(pair_count, BIG_M_MAX_VOLUME),
                )
            ),
            variables=variables,
            lower_bounds=None,
            upper_bounds=np.full(pair_count, BIG_M_MAX_VOLUME),
            names=np.array(
                [f"same_volume_reverse_{i}" for i in range(pair_count)],
            ),
        )

    @staticmethod
    def _build_at_least_one_pair_constraint(
        solver: AbstractOptimizationSolver,
        y: np.ndarray,
        pair_count: int,
    ) -> None:
        """
        Adds constraint that ensures at least 1 pair with the same volume is present in the knapsack
        Constraint: sum(y_i,j) >= 1
        """
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
            coefficients=data.item.get_column(PROFIT_COLUMN).to_numpy(),
            variables=data.item.get_column(ITEM_IS_SELECTED_VAR).to_numpy(),
        )

    def _variable_analytics(self, analytics_data: KnapsackOutputData) -> pl.DataFrame:
        return analytics_data.item.select(
            ITEM_ID,
            ITEM_IS_SELECTED_VALUE,
            (pl.col(PROFIT_COLUMN) * pl.col(ITEM_IS_SELECTED_VALUE)).alias(OBJECTIVE_PROFIT_CONTRIBUTION),
        )

    def _total_analytics(self, analytics_data: KnapsackOutputData) -> float:
        variable_analytics = self.get_analytics(granularity="variable", analytics_data=analytics_data)
        return float(variable_analytics.get_column(OBJECTIVE_PROFIT_CONTRIBUTION).sum())


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
        return KnapsackOutputData(
            item=internal_unpacked_data.item,
            item_pair=internal_unpacked_data.item_pair,
            knapsack_capacity=internal_unpacked_data.knapsack_capacity,
            knapsack_volume_max_gap=internal_unpacked_data.knapsack_volume_max_gap,
        )


class KnapsackModelFactory(AbstractMipModelFactory):
    """
    Factory for creating a KnapsackMipModel with the appropriate variable and constraint builders based on the provided settings.
    """

    def construct(self, settings: KnapsackSettings) -> KnapsackMipModel:
        variable_builders = {KnapsackIsSelectedVariableBuilder}
        constraint_builders = {KnapsackCapacityConstraintBuilder}

        if settings.enable_max_volume_difference:
            constraint_builders.add(KnapsackLargeSmallGapConstraintBuilder)

        if settings.enable_at_least_one_same_volume_pair:
            variable_builders.add(KnapsackSameVolumePairsVariableBuilder)
            constraint_builders.add(KnapsackSameVolumePairsConstraintBuilder)

        return KnapsackMipModel(
            solver=self._solver,
            data_preparator=KnapsackDataPreparator(logger=self._logger, settings=settings),
            variable_builders=[variable_builders(logger=self._logger) for variable_builders in variable_builders],
            constraint_builders=[
                constraint_builders(logger=self._logger) for constraint_builders in constraint_builders
            ],
            objective_builders=[KnapsackObjectiveBuilder(logger=self._logger)],
            logger=self._logger,
        )


logger = SemanticLogger().add_log_source(
    log_source_name="DataFrameKnapsack",
    min_log_level=LogLevel.INFO,
    log_handlers=[SafeStreamHandler(sys.stdout)],
    is_default=True,
)


def main() -> None:
    settings = KnapsackSettings(
        enable_max_volume_difference=True,
        enable_at_least_one_same_volume_pair=True,
    )

    model = KnapsackModelFactory(
        logger=logger,
        solver_type=SolverType.ORTOOLS_SCIP,
    ).construct(settings=settings)

    model.build(
        input_data=KnapsackInputData(
            item=(
                pl.read_parquet("data/knapsack.parquet")
                .with_row_index(ITEM_ID)
                .with_columns(
                    pl.col(ITEM_ID).cast(pl.String),
                )
            ),
            knapsack_capacity=15,
            knapsack_volume_max_gap=2,
        )
    )
    model.solve()

    output = model.get_output_data()
    print(output.item[[ITEM_ID, ITEM_IS_SELECTED_VALUE]])
    print(model.get_analytics(granularity="variable"))
    print(model.get_analytics(granularity="total"))

    selected = output.item.filter(pl.col(ITEM_IS_SELECTED_VALUE))

    print(
        selected.get_column(PROFIT_COLUMN).sum(),
        selected.get_column(VOLUME_COLUMN).sum(),
    )


if __name__ == "__main__":
    main()
