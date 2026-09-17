"""Abstract definition of a Knapsack Problem builder."""

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

import polars as pl

from abc import ABC

from umip.abstract_constr_builder import AbstractConstraintBuilder
from umip.abstract_var_builder import AbstractDecisionVariableBuilder
from umip.abstract_obj_builder import AbstractObjectiveBuilder


class KnapsackDecisionVariableBuilder(AbstractDecisionVariableBuilder, ABC):
    """
    Abstract base class for knapsack decision variable builders.
    """


class KnapsackConstraintBuilder(AbstractConstraintBuilder, ABC):
    """
    Abstract base class for knapsack constraint builders.
    """

    @staticmethod
    def _join_item_column_to_pairs(
        item_pair: pl.DataFrame,
        item: pl.DataFrame,
        item_id_column: str,
        item_column: str,
        index_i_column: str,
        index_j_column: str,
        item_i_is_selected_var_column: str,
        item_j_is_selected_var_column: str,
    ) -> pl.DataFrame:
        """
        Joins the item column to the item pair dataframe on the index columns.
        """
        return item_pair.join(
            item.select(
                item_id_column,
                pl.col(item_column).alias(item_i_is_selected_var_column),
            ),
            left_on=index_i_column,
            right_on=item_id_column,
            validate="m:1",
        ).join(
            item.select(
                item_id_column,
                pl.col(item_column).alias(item_j_is_selected_var_column),
            ),
            left_on=index_j_column,
            right_on=item_id_column,
            validate="m:1",
        )


class KnapsackObjectiveFunctionBuilder(AbstractObjectiveBuilder, ABC):
    """
    Abstract base class for knapsack objective function builders.
    """