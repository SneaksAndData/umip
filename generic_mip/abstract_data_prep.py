"""Abstract definition of a data preparator."""
from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic
from adapta.logs import SemanticLogger

T = TypeVar('T')
U = TypeVar('U')


class AbstractDataPreparator(ABC, Generic[T, U]):
    """The responsibility of the data preparator is to prepare data for the model."""
    def __init__(self, logger: SemanticLogger):
        """
        Initialize the data preparator.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def prepare(self, input_data: Dict[str, T]) -> Dict[str, U]:
        """
        Prepares the data for building variables, constraints and objectives.
        :param input_data: The data to prepare for the model.
        :return: The prepared data.
        """
