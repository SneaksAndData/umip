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
def test_build_abstract_mip(prepare, var_build, constr_build, obj_build, unpack):
    """
    Testing that all provided builders are called.
    """
    model = AbstractMipModel(
        solver=mock.Mock(),
        constraint_builders=[MockConstraintBuilder()],
        variable_builders=[MockDecisionVariableBuilder()],
        objective_builders=[MockObjectiveBuilder()],
        data_preparator=MockDataPreparator()
    )
    model.build(df=pd.DataFrame({"a": [1, 2, 3]}))
    prepare.assert_called_once()
    var_build.assert_called_once()
    obj_build.assert_called_once()
    constr_build.assert_called_once()
    unpack.assert_not_called()
    model.solve()
    unpack.assert_called_once()


def test_early_solve():
    """
    Testing that the solve method raises an error if the model is not built.
    """
    model = AbstractMipModel(
        solver=mock.Mock(),
        constraint_builders=[MockConstraintBuilder()],
        variable_builders=[MockDecisionVariableBuilder()],
        objective_builders=[MockObjectiveBuilder()],
        data_preparator=MockDataPreparator()
    )
    with pytest.raises(ValueError):
        model.solve()


def test_early_set_verbose():
    """
    Testing that the set_verbose method raises an error if the model is built.
    """
    model = AbstractMipModel(
        solver=mock.Mock(),
        constraint_builders=[MockConstraintBuilder()],
        variable_builders=[MockDecisionVariableBuilder()],
        objective_builders=[MockObjectiveBuilder()],
        data_preparator=MockDataPreparator()
    )
    model.build(df=pd.DataFrame())
    with pytest.raises(ValueError):
        model.set_verbose_mode(verbose=True)


@pytest.mark.parametrize("verbose", [True, False])
def test_set_verbose(capfd, verbose):
    model = AbstractMipModel(
        solver=mock.Mock(),
        constraint_builders=[MockConstraintBuilder()],
        variable_builders=[MockDecisionVariableBuilder()],
        objective_builders=[MockObjectiveBuilder()],
        data_preparator=MockDataPreparator()
    )
    model.set_verbose_mode(verbose=verbose)
    model.build(df=pd.DataFrame())

    out, err = capfd.readouterr()

    if verbose:
        assert len(out) > 0
    else:
        assert len(out) == 0
