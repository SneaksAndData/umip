"""Abstract definition of a data preparator."""
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
from proteus.logs import ProteusLogger


class AbstractDataPreparator(ABC):
    """The responsibility of the data preparator is to prepare data for the model."""
    def __init__(self, logger: ProteusLogger):
        """
        Initialize the data preparator.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def prepare(self, input_dfs: Dict[any, pd.DataFrame]) -> Dict[any, pd.DataFrame]:
        """
        Prepares the data for building variables, constraints and objectives.
        :param input_dfs: The data to prepare.
        :return: The prepared data.
        """
