"""Abstract definition of an optimization model."""
from abc import ABC, abstractmethod
import pandas as pd


class AbstractOptimizationModel(ABC):
    """A generic optimization model interface."""
    @abstractmethod
    def build(self, **input_dfs: pd.DataFrame) -> None:
        """
        Builds the model using the given variable, constraint and objective builders.
        :param input_dfs: Input data to the variables, constraints and objectives.
        :return:
        """

    @abstractmethod
    def solve(self, **kwargs: any) -> any:
        """
        Solves the model and returns the result of the optimization.
        :param kwargs: Optional arguments to the optimization.
        :return: The result.
        """

    @abstractmethod
    def objective_value(self) -> float:
        """
        Get the objective value of the optimization.
        :return: The objective value.
        """

    @abstractmethod
    def set_verbose_mode(self, verbose: bool) -> None:
        """
        Sets verbose mode of the building and solving processes.
        :param verbose: Whether to enable verbose mode.
        :return:
        """
