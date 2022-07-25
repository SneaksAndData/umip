"""Abstract definition of a constraint builder."""
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
from generic_mip.abstract_solver import AbstractOptimizationSolver


class AbstractConstraintBuilder(ABC):
    """A constraint builder has the responsibility of building one or more constraints."""
    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, input_dfs: Dict[any, pd.DataFrame]) -> None:
        """
        Builds the constraints on the given model and the given data.

        :param solver: The solver to use to build the constraints.
        :param input_dfs: The dataframes providing variables and parameters for the constraints.
        :return:
        """
