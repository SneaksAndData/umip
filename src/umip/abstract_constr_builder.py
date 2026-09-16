"""Abstract definition of a constraint builder."""

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

from abc import ABC, abstractmethod

import polars as pl
from adapta.logs import LoggerInterface

from umip.abstract_dataclasses import AbstractInternalData
from umip.abstract_solver import AbstractOptimizationSolver


class AbstractConstraintBuilder(ABC):
    """A constraint builder has the responsibility of building one or more constraints."""

    def __init__(self, logger: LoggerInterface):
        """
        Initialize the constraint builder.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, data: AbstractInternalData) -> None:
        """
        Builds the constraints on the given model and the given data.

        :param solver: The solver to use to build the constraints.
        :param data: The data (e.g. dataframes) providing variables and parameters for the constraints.
        :return:
        """

    @staticmethod
    def _join_item_column_to_pairs(
        item_pair: pl.DataFrame,
        item: pl.DataFrame,
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
            item.with_row_index(index_i_column).select(
                index_i_column,
                pl.col(item_column).alias(item_i_is_selected_var_column),
            ),
            on=index_i_column,
            how="left",
            validate="m:1",
        ).join(
            item.with_row_index(index_j_column).select(
                index_j_column,
                pl.col(item_column).alias(item_j_is_selected_var_column),
            ),
            on=index_j_column,
            how="left",
            validate="m:1",
        )
