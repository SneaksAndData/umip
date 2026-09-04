"""
Mock classes for use in tests
"""

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

from dataclasses import dataclass
from typing import Any

from umip import (
    AbstractDataPreparator,
    AbstractMipModel,
    AbstractObjectiveBuilder,
    AbstractOptimizationSolver,
)
from umip.abstract_constr_builder import AbstractConstraintBuilder
from umip.abstract_dataclasses import (
    AbstractInputData,
    AbstractInternalData,
    AbstractOutputData,
)
from umip.abstract_var_builder import AbstractDecisionVariableBuilder


@dataclass
class MockInputData(AbstractInputData):
    data: Any


@dataclass
class MockInternalData(MockInputData, AbstractInternalData):
    pass


@dataclass
class MockOutputData(MockInternalData, AbstractOutputData):
    pass


class MockMipModel(AbstractMipModel):
    @staticmethod
    def _convert_internal_to_output_data(
        internal_unpacked_data: MockInternalData,
    ) -> MockOutputData:
        pass


class MockConstraintBuilder(AbstractConstraintBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: MockInternalData) -> None:
        pass


class MockDecisionVariableBuilder(AbstractDecisionVariableBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: MockInternalData) -> MockInternalData:
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: MockInternalData) -> MockInternalData:
        return data


class MockObjectiveBuilder(AbstractObjectiveBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: MockInternalData) -> None:
        pass


class MockDataPreparator(AbstractDataPreparator):
    def prepare(self, input_data: MockInputData) -> MockInternalData:
        return MockInternalData(data=input_data.data)
