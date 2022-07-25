"""Abstract definition of a variable builder."""
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
from generic_mip.abstract_solver import AbstractOptimizationSolver


class AbstractDecisionVariableBuilder(ABC):
    """A variable builder has the responsibility of building one or more decision variables."""
    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, input_dfs: Dict[any, pd.DataFrame]) -> Dict[any, pd.DataFrame]:
        """
        Builds the decision variables on the given model and the given data.

        :param solver: The solver to use to build the variables.
        :param input_dfs: The dataframes providing parameters for the variables.
        :return: The dataframes decorated with the created decision variables.
        """

    @abstractmethod
    def unpack(self, solver: AbstractOptimizationSolver, input_dfs: Dict[any, pd.DataFrame]) -> Dict[any, pd.DataFrame]:
        """
        Unpacks the decision variables after optimization and inserts variable values in the dataframes.

        :param solver: The solver to get the variable values from.
        :param input_dfs: The dataframes containing the variables.
        :return: The dataframes decorated with the values of the decision variables.
        """
