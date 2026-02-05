from typing import Any, TypeVar

from dataclasses import replace
import pandas as pd
import numpy as np
from adapta.logs import LoggerInterface

from generic_mip import (
    AbstractDataPreparator,
    AbstractDecisionVariableBuilder,
    AbstractOptimizationSolver,
    VariableDomain,
    AbstractObjectiveBuilder,
    AbstractConstraintBuilder,
    AbstractMipModel,
)
from generic_mip.abstract_dataclasses import *

T = TypeVar("T")


@dataclass
class ExampleInputData(AbstractInputData):
    my_data: pd.DataFrame


@dataclass
class ExampleInternalData(ExampleInputData, AbstractInternalData):
    another_data: pd.DataFrame  # to see if inheritance works


@dataclass
class ExampleOutputData(ExampleInternalData, AbstractOutputData):
    pass


class MyDataPreparator(AbstractDataPreparator):
    def prepare(self, input_data: ExampleInputData) -> ExampleInternalData:
        return ExampleInternalData(my_data=input_data.my_data, another_data=pd.DataFrame())


class MyVariableBuilder(AbstractDecisionVariableBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> ExampleInternalData:
        my_df = data.my_data
        x = solver.add_variable(lower_bound=0, upper_bound=100, name="x", variable_domain=VariableDomain.INTEGER)
        y = solver.add_variable(lower_bound=0, upper_bound=100, name="y", variable_domain=VariableDomain.INTEGER)
        my_df["vars"] = [x, y]
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> ExampleInternalData:
        post_optimization_data = replace(data)
        my_df = post_optimization_data.my_data
        my_df["value"] = my_df["vars"].apply(lambda x: solver.get_variable_value(x))
        return post_optimization_data


class MyObjectiveBuilder(AbstractObjectiveBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData, **kwargs: Any) -> None:
        my_df = data.my_data

        solver.add_objective_term(
            coefficient=1,
            variable=my_df.iloc[0]["vars"],
        )
        solver.add_objective_term(
            coefficient=4,
            variable=my_df.iloc[1]["vars"],
        )


class MyConstraintBuilder(AbstractConstraintBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> None:
        my_df = data.my_data
        solver.add_constraint(
            lower_bound=None,
            upper_bound=100,
            coefficients=np.array([1.0, 1.0]),
            variables=np.array([my_df.iloc[0]["vars"], my_df.iloc[1]["vars"]]),
            name="my_constraint",
        )


class MyOtherConstraintBuilder(AbstractConstraintBuilder):
    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> None:
        my_df = data.my_data
        solver.add_constraint(
            lower_bound=None,
            upper_bound=20,
            coefficients=np.array([1.0]),
            variables=np.array([my_df.iloc[1]["vars"]]),
            name="my_constraint_2",
        )


class MyMipModel(AbstractMipModel):
    def __init__(
        self,
        solver: AbstractOptimizationSolver,
        constraint_builders: list[AbstractConstraintBuilder],
        variable_builders: list[AbstractDecisionVariableBuilder],
        objective_builders: list[AbstractObjectiveBuilder],
        data_preparator: MyDataPreparator,
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

    async def build_async(self, input_data: ExampleInputData) -> None:
        await super().build_async(input_data)
        self._solver.set_optimization_direction(True)

    async def solve_async(self, **kwargs: any) -> None:
        await super().solve_async(**kwargs)

    def build(self, input_data: ExampleInputData) -> None:
        super().build(input_data)
        self._solver.set_optimization_direction(True)

    def solve(self, **kwargs: Any) -> None:
        super().solve(**kwargs)

    @staticmethod
    def _convert_internal_to_output_data(internal_data: ExampleInternalData) -> ExampleOutputData:
        return ExampleOutputData(
            my_data=internal_data.my_data,
            another_data=internal_data.another_data,
        )
