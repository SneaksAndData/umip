"""
Tests of the AbstractMipModel class. Tests do not cover the solver classes and simple wrappers.
"""
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl
import pytest
from adapta.logs import LoggerInterface
from unittest.mock import call, MagicMock

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

        data["df"] = self.build_column_variables(
            solver=solver,
            data=data["df"],
            destination_column="var_test_float_all_removed_indicator",
            variable_dtype=VariableDataType.FLOAT,
            lower_bound=1.0,
            upper_bound=4.0,
            filter_column="all_removed_indicator_test",
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

        data["df"] = self.unpack_column_variables(
            solver=solver,
            data=data["df"],
            decision_variable_column="var_test_float_all_removed_indicator",
            decision_variable_value_column="var_test_float_all_removed_indicator_value",
            filter_column="all_removed_indicator_test",
            default_unpack_value=np.NAN,
        )

        return data


@pytest.mark.parametrize("solver", ["OrTools"], indirect=True)
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {"a": [1, 2, 3], "indicator_test": [True, False, True], "all_removed_indicator_test": [False, False, False]}
        ),
        pl.DataFrame(
            {"a": [1, 2, 3], "indicator_test": [True, False, True], "all_removed_indicator_test": [False, False, False]}
        ),
        pd.DataFrame({"a": [], "indicator_test": [], "all_removed_indicator_test": []}),
        pl.DataFrame({"a": [], "indicator_test": [], "all_removed_indicator_test": []}),
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


@pytest.mark.parametrize("solver", ["OrTools"], indirect=True)
def test__abstract_mip__get_analytics__functional(solver: AbstractOptimizationSolver, logger):
    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.objective_name = "builder_1"
            self.add_analytics_granularity("sku", self._sku_analytics)
            self.add_analytics_granularity("sku_location", self._sku_location_analytics)
            self.add_analytics_granularity("aggregated", self._aggregated_analytics)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

        def _sku_analytics(self, analytics_data: dict[str, Any]) -> Any:
            return analytics_data["sku"]

        def _sku_location_analytics(self, analytics_data: dict[str, Any]) -> Any:
            return "something_else"

        def _aggregated_analytics(self, analytics_data: dict[str, Any]) -> Any:
            return sum(analytics_data["sku"])

    class ObjectiveBuilder2(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.add_analytics_granularity("aggregated", self.aggregated_analytics)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

        def aggregated_analytics(self, analytics_data: dict[str, Any]) -> Any:
            return sum(analytics_data["location"])

    model = AbstractMipModel(
        solver=solver,
        constraint_builders=[],
        variable_builders=[],
        objective_builders=[ObjectiveBuilder1(logger), ObjectiveBuilder2(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger,
    )
    # we don't want to test optimization here, so we are simply overriding the _solved attribute
    model._solved = True
    # we need to set the _data attribute manually, as we are not calling the build method
    model._data = {
        "location": [10, 5, 20],
        "sku": [1000, 500, 100],
    }

    assert model.get_analytics(granularity="sku") == {"builder_1": [1000, 500, 100]}
    assert model.get_analytics(granularity="sku_location") == {"builder_1": "something_else"}
    assert model.get_analytics(granularity="aggregated") == {"builder_1": 1600, "ObjectiveBuilder2": 35}
    assert model.get_analytics(granularity="unknown") == {}


@pytest.mark.parametrize("solver", ["OrTools"], indirect=True)
def test__abstract_mip__get_analytics__analytics_data_provided(solver: AbstractOptimizationSolver, logger):
    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.objective_name = "builder_1"
            self.add_analytics_granularity("sku", self._sku_analytics)
            self.add_analytics_granularity("sku_location", self._sku_location_analytics)
            self.add_analytics_granularity("aggregated", self._aggregated_analytics)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

        def _sku_analytics(self, analytics_data: dict[str, Any]) -> Any:
            return analytics_data["sku"]

        def _sku_location_analytics(self, analytics_data: dict[str, Any]) -> Any:
            return "something_else"

        def _aggregated_analytics(self, analytics_data: dict[str, Any]) -> Any:
            return sum(analytics_data["sku"])

    class ObjectiveBuilder2(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.add_analytics_granularity("aggregated", self.aggregated_analytics)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

        def aggregated_analytics(self, analytics_data: dict[str, Any]) -> Any:
            return sum(analytics_data["location"])

    model = AbstractMipModel(
        solver=solver,
        constraint_builders=[],
        variable_builders=[],
        objective_builders=[ObjectiveBuilder1(logger), ObjectiveBuilder2(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger,
    )
    # we need to set the _data attribute manually, as we are not calling the build method
    analytics_data = {
        "location": [10, 5, 20],
        "sku": [1000, 500, 100],
    }

    assert model.get_analytics(granularity="sku", analytics_data=analytics_data) == {"builder_1": [1000, 500, 100]}
    assert model.get_analytics(granularity="sku_location", analytics_data=analytics_data) == {
        "builder_1": "something_else"
    }
    assert model.get_analytics(granularity="aggregated", analytics_data=analytics_data) == {
        "builder_1": 1600,
        "ObjectiveBuilder2": 35,
    }
    assert model.get_analytics(granularity="unknown", analytics_data=analytics_data) == {}


@pytest.mark.parametrize("solver", ["OrTools"], indirect=True)
def test__abstract_mip__get_analytics__logs_warning(solver: AbstractOptimizationSolver):
    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

    class ObjectiveBuilder2(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

    logger = MagicMock()
    model = AbstractMipModel(
        solver=solver,
        constraint_builders=[],
        variable_builders=[],
        objective_builders=[ObjectiveBuilder1(logger), ObjectiveBuilder2(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger,
    )
    # we don't want to test optimization here, so we are simply overriding the _solved attribute
    model._solved = True

    model.get_analytics(granularity="aggregated")

    expected_warning_call = call(
        "No analytics found for granularity 'aggregated' in objective builders: ['ObjectiveBuilder1', 'ObjectiveBuilder2']"
    )

    assert model._logger.warning.call_args_list[0] == expected_warning_call


@pytest.mark.parametrize("solver", ["OrTools"], indirect=True)
def test__abstract_mip__duplicate_objective_builder_names(solver: AbstractOptimizationSolver, logger):
    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.objective_name = "same_as_other"

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

    class ObjectiveBuilder2(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.objective_name = "same_as_other"

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

    model = AbstractMipModel(
        solver=solver,
        constraint_builders=[],
        variable_builders=[],
        objective_builders=[ObjectiveBuilder1(logger), ObjectiveBuilder2(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger,
    )

    with pytest.raises(ValueError, match="Duplicate objective builder name found: same_as_other"):
        model.build()
