"""
Mock classes for use in tests
"""

from dataclasses import dataclass
from typing import Any

from src.umip.abstract_constr_builder import AbstractConstraintBuilder
from src.umip import SolverType
from src.umip.abstract_var_builder import AbstractDecisionVariableBuilder
from src.umip import AbstractObjectiveBuilder
from src.umip import AbstractDataPreparator
from src.umip import AbstractOptimizationSolver
from src.umip import AbstractMipModel
from src.umip import AbstractInternalData, AbstractInputData, AbstractOutputData


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
    def _convert_internal_to_output_data(internal_unpacked_data: MockInternalData) -> MockOutputData:
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
