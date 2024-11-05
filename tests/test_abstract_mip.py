"""
Tests of the AbstractMipModel class. Tests do not cover the solver classes and simple wrappers.
"""
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from generic_mip import VariableDataType
from generic_mip.abstract_constr_builder import AbstractConstraintBuilder
from generic_mip.abstract_var_builder import AbstractDecisionVariableBuilder
from generic_mip.abstract_obj_builder import AbstractObjectiveBuilder
from generic_mip.abstract_data_prep import AbstractDataPreparator
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.abstract_mip import AbstractMipModel


class MockConstraintBuilder(AbstractConstraintBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> None:
        pass


class MockDecisionVariableBuilder(AbstractDecisionVariableBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return data


class MockObjectiveBuilder(AbstractObjectiveBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> None:
        pass


class MockDataPreparator(AbstractDataPreparator):
    def prepare(self, input_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return input_data


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
        logger=logger,
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
        logger=logger,
    )
    with pytest.raises(ValueError):
        model.solve()


class TestBuildVarFunction(AbstractDecisionVariableBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        data["df"] = self.build_column_variables(
            solver=solver,
            data=data["df"],
            destination_column="var_test_int",
            variable_dtype=VariableDataType.INT,
            lower_bound=0.0,
            upper_bound="a",
        )

        data["df"] = self.build_column_variables(
            solver=solver,
            data=data["df"],
            destination_column="var_test_bool",
            variable_dtype=VariableDataType.BOOL,
        )

        data["df"] = self.build_column_variables(
            solver=solver,
            data=data["df"],
            destination_column="var_test_bool_dtype",
            variable_dtype=VariableDataType.BOOL,
            filter_column="indicator_test",
        )

        data["df"] = self.build_column_variables(
            solver=solver,
            data=data["df"],
            destination_column="var_test_float",
            variable_dtype=VariableDataType.FLOAT,
            lower_bound=-4.0,
            upper_bound=4.0,
            index_name_columns=["a", "indicator_test"],
        )

        data["df"] = self.build_column_variables(
            solver=solver,
            data=data["df"],
            destination_column="var_test_float_indicator",
            variable_dtype=VariableDataType.FLOAT,
            lower_bound=1.0,
            upper_bound=4.0,
            filter_column="indicator_test",
        )

        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        data["df"] = self.unpack_column_variables(
            solver=solver,
            data=data["df"],
            decision_variable_column="var_test_int",
            decision_variable_value_column="var_test_int_value",
            return_dtype=VariableDataType.INT,
        )

        data["df"] = self.unpack_column_variables(
            solver=solver,
            data=data["df"],
            decision_variable_column="var_test_bool",
            decision_variable_value_column="var_test_bool_value",
        )

        data["df"] = self.unpack_column_variables(
            solver=solver,
            data=data["df"],
            decision_variable_column="var_test_bool_dtype",
            decision_variable_value_column="var_test_bool_dtype_value",
            filter_column="indicator_test",
            return_dtype=VariableDataType.BOOL,
        )

        data["df"] = self.unpack_column_variables(
            solver=solver,
            data=data["df"],
            decision_variable_column="var_test_float",
            decision_variable_value_column="var_test_float_value",
        )

        data["df"] = self.unpack_column_variables(
            solver=solver,
            data=data["df"],
            decision_variable_column="var_test_float_indicator",
            decision_variable_value_column="var_test_float_indicator_value",
            filter_column="indicator_test",
            default_unpack_value=np.NAN,
        )

        return data


@pytest.mark.parametrize("solver", ["OrTools"], indirect=True)
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1, 2, 3], "indicator_test": [True, False, True]}),
        pl.DataFrame({"a": [1, 2, 3], "indicator_test": [True, False, True]}),
        pd.DataFrame({"a": [], "indicator_test": []}),
        pl.DataFrame({"a": [], "indicator_test": []}),
    ],
)
def test_var_builder(logger, solver: AbstractOptimizationSolver, df: pd.DataFrame | pl.DataFrame):
    """
    Testing that build var function works as intended.

    We test for a simple pandas and polars DataFrame with 3 rows, while also testing a pandas and polars DataFrame
    that is empty. We create the following variables:

         1) var_test_int: integer with lower bound of 0, and upper bound in column "a".
         2) var_test_bool: bool
         3) var_test_float: float with lower bound of -4, and upper bound of 4. The variable index names should be
         according to "a" and "indicator_test".
         4) var_test_float_indicator: float with lower bound of 1, and upper bound of 4. The column should only contain
         variables according to column "indicator_test" - when False, it should contain None.

    We check that the column names, bounds and number of variables is correct.
    """
    model = AbstractMipModel(
        solver=solver,
        constraint_builders=[MockConstraintBuilder(logger)],
        variable_builders=[TestBuildVarFunction(logger)],
        objective_builders=[MockObjectiveBuilder(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger,
    )

    model.build(df=df)

    model_data = model._data["df"]

    if isinstance(model_data, pl.DataFrame):
        model_data = model_data.to_pandas()

    assert all(
        x in model_data.columns
        for x in ["var_test_int", "var_test_bool", "var_test_bool_dtype", "var_test_float", "var_test_float_indicator"]
    )

    if not model_data.empty:
        assert sum(model_data["var_test_int"].apply(lambda x: x.lb()) == 0.0) == 3
        assert sum(model_data["var_test_int"].apply(lambda x: x.ub()) == model_data["a"]) == 3
        assert sum(model_data["var_test_bool"].apply(lambda x: x.lb()) == 0.0) == 3
        assert sum(model_data["var_test_bool"].apply(lambda x: x.ub()) == 1.0) == 3
        assert sum(model_data["var_test_float"].apply(lambda x: x.lb()) == -4.0) == 3
        assert sum(model_data["var_test_float"].apply(lambda x: x.ub()) == 4.0) == 3
        assert (
            sum(
                model_data.loc[lambda x: x["indicator_test"]]["var_test_float_indicator"].apply(lambda x: x.lb()) == 1.0
            )
            == 2
        )
        assert (
            sum(
                model_data.loc[lambda x: x["indicator_test"]]["var_test_float_indicator"].apply(lambda x: x.ub()) == 4.0
            )
            == 2
        )
        assert sum(model_data["var_test_float_indicator"].apply(lambda x: x is None)) == 1

    model.solve()

    model_data = model._data["df"]

    if isinstance(model_data, pl.DataFrame):
        model_data = model_data.to_pandas()

    all(
        x in model_data.columns
        for x in [
            "var_test_int_value",
            "var_test_bool_value",
            "var_test_bool_dtype_value",
            "var_test_float_value",
            "var_test_float_indicator_value",
        ]
    )

    if not model_data.empty:
        assert sum(model_data["var_test_int_value"] >= 0) == 3
        assert sum(model_data["var_test_int_value"] <= model_data["a"]) == 3
        assert isinstance(model_data["var_test_int_value"].dtype, np.dtypes.Int64DType)
        assert sum(model_data["var_test_bool_value"] >= 0.0) == 3
        assert sum(model_data["var_test_bool_value"] <= 1.0) == 3
        assert isinstance(model_data["var_test_bool_dtype_value"].dtype, np.dtypes.BoolDType)
        assert sum(model_data["var_test_float_value"] >= -4.0) == 3
        assert sum(model_data["var_test_float_value"] <= 4.0) == 3
        assert np.isnan(model_data["var_test_float_indicator_value"].values[1])
