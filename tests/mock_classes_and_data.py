"""
Mock classes for use in tests
"""

from dataclasses import dataclass
from typing import Any

from generic_mip.abstract_constr_builder import AbstractConstraintBuilder
from generic_mip.enums import SolverType
from generic_mip.abstract_var_builder import AbstractDecisionVariableBuilder
from generic_mip.abstract_obj_builder import AbstractObjectiveBuilder
from generic_mip.abstract_data_prep import AbstractDataPreparator
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.abstract_mip import AbstractMipModel
from generic_mip.abstract_dataclasses import AbstractInternalData, AbstractInputData, AbstractOutputData


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
