"""Abstract definition of a data preparator."""

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
