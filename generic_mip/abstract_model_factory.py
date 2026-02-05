"""Abstract definition of an optimization model factory."""
from abc import ABC, abstractmethod
from typing import Any

from adapta.logs import LoggerInterface
from generic_mip.abstract_mip import AbstractMipModel


class AbstractMipModelFactory(ABC):
    """A generic MIP model factory."""

    def __init__(self, logger: LoggerInterface):
        """
        Initialize the model factory.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def construct(
        self,
        **kwargs: Any,
    ) -> AbstractMipModel:
        """
        Given the arguments, construct an MIP model.
        :param kwargs: The arguments to the construction.
        :return: The constructed MIP model.
        """
