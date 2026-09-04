"""Abstract definition of an optimization model factory."""

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
from typing import Any

from adapta.logs import LoggerInterface

from umip.abstract_mip import AbstractMipModel
from umip.abstract_solver import AbstractOptimizationSolver
from umip.enums import SolverType
from umip.solver_config import SolverConfig
from umip.solver_factory import SolverFactory


class AbstractMipModelFactory(ABC):
    """A generic MIP model factory."""

    def __init__(
        self,
        logger: LoggerInterface,
        solver_type: SolverType,
        solver_config: SolverConfig | None = None,
    ):
        """
        Initialize the model factory.
        :param logger: The logger to use.
        """
        self._logger = logger
        self._solver = self._get_solver(solver_type=solver_type, solver_config=solver_config)

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

    def _get_solver(
        self, solver_type: SolverType, solver_config: SolverConfig | None = None
    ) -> AbstractOptimizationSolver:
        """
        Get the solver instance based on the specified solver type and configuration.
        :param solver_type: The type of solver to construct.
        :param solver_config: Optional typed configuration object for the solver type.
        :return: solver.
        """
        return SolverFactory(logger=self._logger).construct(solver_type=solver_type, solver_config=solver_config)
