"""Abstract definition of a data preparator."""

from abc import ABC, abstractmethod
from adapta.logs import LoggerInterface
from umip.abstract_dataclasses import AbstractInputData, AbstractInternalData


class AbstractDataPreparator(ABC):
    """The responsibility of the data preparator is to prepare data for the model."""

    def __init__(self, logger: LoggerInterface):
        """
        Initialize the data preparator.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def prepare(self, input_data: AbstractInputData) -> AbstractInternalData:
        """
        Prepares the data for building variables, constraints and objectives.
        :param input_data: The data to prepare for the model.
        :return: The prepared data.
        """
