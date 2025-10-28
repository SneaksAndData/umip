from typing import Any, TypeVar

import pandas as pd
import numpy as np
from adapta.logs import LoggerInterface

from generic_mip import (
    AbstractDataPreparator,
    AbstractDecisionVariableBuilder,
    AbstractOptimizationSolver,
    VariableDataType,
    AbstractObjectiveBuilder,
    AbstractConstraintBuilder,
    AbstractMipModel,
)

T = TypeVar("T")


class MyDataPreparator(AbstractDataPreparator[pd.DataFrame, pd.DataFrame]):
    def prepare(self, input_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return input_data


class MyVariableBuilder(AbstractDecisionVariableBuilder[pd.DataFrame]):
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        my_df = data["my_df"]
        x = solver.add_variable(lb=0, ub=100, name="x", dtype=VariableDataType.INT)
        y = solver.add_variable(lb=0, ub=100, name="y", dtype=VariableDataType.INT)
        my_df["vars"] = [x, y]
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        my_df = data["my_df"]
        my_df["value"] = my_df["vars"].apply(lambda x: solver.get_variable_value(x))
        return data


class MyObjectiveBuilder(AbstractObjectiveBuilder[pd.DataFrame]):
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame], **kwargs: Any) -> None:
        my_df = data["my_df"]

        solver.add_objective_term(
            coeff=1,
            var=my_df.iloc[0]["vars"],
        )
        solver.add_objective_term(
            coeff=4,
            var=my_df.iloc[1]["vars"],
        )


class MyConstraintBuilder(AbstractConstraintBuilder[pd.DataFrame]):
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> None:
        my_df = data["my_df"]
        solver.add_constraint(
            lb=None,
            ub=100,
            coeffs=np.array([1.0, 1.0]),
            vars_=np.array([my_df.iloc[0]["vars"], my_df.iloc[1]["vars"]]),
            name="my_constraint",
        )


class MyOtherConstraintBuilder(AbstractConstraintBuilder[pd.DataFrame]):
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, pd.DataFrame]) -> None:
        my_df = data["my_df"]
        solver.add_constraint(
            lb=None, ub=20, coeffs=np.array([1.0]), vars_=np.array([my_df.iloc[1]["vars"]]), name="my_constraint_2"
        )


class MyMipModel(AbstractMipModel[pd.DataFrame]):
    def __init__(
        self,
        solver: AbstractOptimizationSolver,
        constraint_builders: list[AbstractConstraintBuilder[pd.DataFrame]],
        variable_builders: list[AbstractDecisionVariableBuilder[pd.DataFrame]],
        objective_builders: list[AbstractObjectiveBuilder[pd.DataFrame]],
        data_preparator: MyDataPreparator([pd.DataFrame, pd.DataFrame]),
        logger: LoggerInterface,
    ):
        super().__init__(
            solver=solver,
            constraint_builders=constraint_builders,
            variable_builders=variable_builders,
            objective_builders=objective_builders,
            data_preparator=data_preparator,
            logger=logger,
        )
        solver.set_verbose(True)

    async def build_async(self, **input_data: T) -> None:
        await super().build_async(**input_data)
        self._solver.set_optimization_direction(True)

    async def solve_async(self, **kwargs: any) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
        await super().solve_async(**kwargs)
        return self._data["my_df"]

    def build(self, **input_data: pd.DataFrame) -> None:
        super().build(**input_data)
        self._solver.set_optimization_direction(True)

    def solve(self, **kwargs: Any) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
        super().solve(**kwargs)
        return self._data["my_df"]
