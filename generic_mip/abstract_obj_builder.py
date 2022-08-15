"""Abstract definition of an objective builder."""
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
from proteus.logs import ProteusLogger
from generic_mip.abstract_solver import AbstractOptimizationSolver


class AbstractObjectiveBuilder(ABC):
    """An objective builder has the responsibility of building one or more objective terms."""
    def __init__(self, logger: ProteusLogger):
        """
        Initialize the objective builder.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, input_dfs: Dict[any, pd.DataFrame], **kwargs: any) -> None:
        """
        Builds the objective terms on the given model and the given data.

        :param solver: The solver to use to build the objective terms.
        :param input_dfs: The dataframes providing variables and parameters for the objective terms.
        :return:
        """
