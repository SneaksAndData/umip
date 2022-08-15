"""
Tests of the AbstractMipModel class. Tests do not cover the solver classes and simple wrappers.
"""
from typing import Dict
from unittest import mock
import pandas as pd
import pytest

from generic_mip.abstract_constr_builder import AbstractConstraintBuilder
from generic_mip.abstract_var_builder import AbstractDecisionVariableBuilder
from generic_mip.abstract_obj_builder import AbstractObjectiveBuilder
from generic_mip.abstract_data_prep import AbstractDataPreparator
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.abstract_mip import AbstractMipModel


class MockConstraintBuilder(AbstractConstraintBuilder):
    def build(self, solver: AbstractOptimizationSolver, input_dfs: Dict[any, pd.DataFrame]) -> None:
        pass


class MockDecisionVariableBuilder(AbstractDecisionVariableBuilder):
    def build(self, solver: AbstractOptimizationSolver, input_dfs: Dict[any, pd.DataFrame]) -> Dict[any, pd.DataFrame]:
        return input_dfs

    def unpack(self, solver: AbstractOptimizationSolver, input_dfs: Dict[any, pd.DataFrame]) -> Dict[any, pd.DataFrame]:
        return input_dfs


class MockObjectiveBuilder(AbstractObjectiveBuilder):
    def build(self, solver: AbstractOptimizationSolver, input_dfs: Dict[any, pd.DataFrame], **kwargs: any) -> None:
        pass


class MockDataPreparator(AbstractDataPreparator):
    def prepare(self, input_dfs: Dict[any, pd.DataFrame]) -> Dict[any, pd.DataFrame]:
        return input_dfs


@mock.patch.object(MockDecisionVariableBuilder, "unpack")
@mock.patch.object(MockObjectiveBuilder, "build")
@mock.patch.object(MockConstraintBuilder, "build")
@mock.patch.object(MockDecisionVariableBuilder, "build")
@mock.patch.object(MockDataPreparator, "prepare")
def test_build_abstract_mip(prepare, var_build, constr_build, obj_build, unpack, logger):
    """
    Testing that all provided builders are called.
    """
    model = AbstractMipModel(
        solver=mock.Mock(),
        constraint_builders=[MockConstraintBuilder(logger)],
        variable_builders=[MockDecisionVariableBuilder(logger)],
        objective_builders=[MockObjectiveBuilder(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger
    )
    model.build(df=pd.DataFrame({"a": [1, 2, 3]}))
    prepare.assert_called_once()
    var_build.assert_called_once()
    obj_build.assert_called_once()
    constr_build.assert_called_once()
    unpack.assert_not_called()
    model.solve()
    unpack.assert_called_once()


def test_early_solve(logger):
    """
    Testing that the solve method raises an error if the model is not built.
    """
    model = AbstractMipModel(
        solver=mock.Mock(),
        constraint_builders=[MockConstraintBuilder(logger)],
        variable_builders=[MockDecisionVariableBuilder(logger)],
        objective_builders=[MockObjectiveBuilder(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger
    )
    with pytest.raises(ValueError):
        model.solve()
