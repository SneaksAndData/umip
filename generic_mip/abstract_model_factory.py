"""Abstract definition of an optimization model factory."""
from abc import ABC, abstractmethod
from adapta.logs import SemanticLogger
from generic_mip.abstract_mip import AbstractOptimizationModel


class AbstractOptimizationModelFactory(ABC):
    """A generic optimization model factory."""
    def __init__(self, logger: SemanticLogger):
        """
        Initialize the model factory.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def construct(
        self,
        **kwargs: any,
    ) -> AbstractOptimizationModel:
        """
        Given the arguments, construct an optimization model.
        :param kwargs: The arguments to the construction.
        :return: The constructed optimization model.
        """
