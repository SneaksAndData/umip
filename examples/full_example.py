from typing import Dict, Union, Tuple, List
import pandas as pd
import numpy as np
import sys
from logging import StreamHandler
from adapta.logs import SemanticLogger
from adapta.logs.models import LogLevel
from generic_mip import AbstractDataPreparator, AbstractDecisionVariableBuilder, AbstractOptimizationSolver, VariableDataType, AbstractObjectiveBuilder, AbstractConstraintBuilder, AbstractMipModel, OrToolsSolver, GurobiSolver
from generic_mip.solver.ortools_solver import OrToolsSolverEngine

"""
Model to implement:

maximize x + 4y

subject to:
x + y <= 100
y <= 20
100 >= x, y >= 0
x, y are integers

This model is equivalent to the example in solver_example.py.

You can access the solver API directly (as in solver_example.py) for local experiments, prototyping, debugging and project initialization.
For production grade models, use the generic_mip framework.
"""

SOLVER = 'ortools'

logger = SemanticLogger().add_log_source(
    log_source_name='MyModel',
    min_log_level=LogLevel.DEBUG,
    log_handlers=[StreamHandler(sys.stdout)],
    is_default=True
)

if SOLVER == 'ortools':
    solver = OrToolsSolver(solver_engine=OrToolsSolverEngine.SCIP, logger=logger)
elif SOLVER == 'gurobi':
    solver = GurobiSolver(logger=logger)
else:
    raise ValueError(f'Invalid solver: {SOLVER}')


class MyDataPreparator(AbstractDataPreparator[pd.DataFrame, pd.DataFrame]):
    def prepare(self, input_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return input_data


class MyVariableBuilder(AbstractDecisionVariableBuilder[pd.DataFrame]):
    def build(self, solver: AbstractOptimizationSolver, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        my_df = data['my_df']
        x = solver.add_variable(lb=0, ub=100, name='x', dtype=VariableDataType.INT)
        y = solver.add_variable(lb=0, ub=100, name='y', dtype=VariableDataType.INT)
        my_df['vars'] = [x, y]
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        my_df = data['my_df']
        my_df['value'] = my_df['vars'].apply(lambda x: solver.get_variable_value(x))
        return data


class MyObjectiveBuilder(AbstractObjectiveBuilder[pd.DataFrame]):
    def build(self, solver: AbstractOptimizationSolver, data: Dict[str, pd.DataFrame], **kwargs: any) -> None:
        my_df = data['my_df']

        solver.add_objective_term(
            coeff=1,
            var=my_df.iloc[0]['vars'],
        )
        solver.add_objective_term(
            coeff=4,
            var=my_df.iloc[1]['vars'],
        )


class MyConstraintBuilder(AbstractConstraintBuilder[pd.DataFrame]):
    def build(self, solver: AbstractOptimizationSolver, data: Dict[str, pd.DataFrame]) -> None:
        my_df = data['my_df']
        solver.add_constraint(
            lb=None,
            ub=100,
            coeffs=np.array([1.0, 1.0]),
            vars_=np.array([my_df.iloc[0]['vars'], my_df.iloc[1]['vars']]),
            name='my_constraint'
        )


class MyOtherConstraintBuilder(AbstractConstraintBuilder[pd.DataFrame]):
    def build(self, solver: AbstractOptimizationSolver, data: Dict[str, pd.DataFrame]) -> None:
        my_df = data['my_df']
        solver.add_constraint(
            lb=None,
            ub=20,
            coeffs=np.array([1.0]),
            vars_=np.array([my_df.iloc[1]['vars']]),
            name='my_constraint_2'
        )


class MyMipModel(AbstractMipModel[pd.DataFrame]):
    def __init__(
        self,
        solver: AbstractOptimizationSolver,
        constraint_builders: List[AbstractConstraintBuilder[pd.DataFrame]],
        variable_builders: List[AbstractDecisionVariableBuilder[pd.DataFrame]],
        objective_builders: List[AbstractObjectiveBuilder[pd.DataFrame]],
        data_preparator: MyDataPreparator[[pd.DataFrame, pd.DataFrame]],
        logger: SemanticLogger
    ):
        super().__init__(
            solver=solver,
            constraint_builders=constraint_builders,
            variable_builders=variable_builders,
            objective_builders=objective_builders,
            data_preparator=data_preparator,
            logger=logger
        )
        solver.set_verbose(True)

    def build(self, **input_data: pd.DataFrame) -> None:
        super().build(**input_data)
        self._solver.set_optimization_direction(True)

    def solve(self, **kwargs: any) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
        super().solve(**kwargs)
        return self._data['my_df']


my_model = MyMipModel(
    solver=solver,
    constraint_builders=[MyConstraintBuilder(logger=logger), MyOtherConstraintBuilder(logger=logger)],
    variable_builders=[MyVariableBuilder(logger=logger)],
    objective_builders=[MyObjectiveBuilder(logger=logger)],
    data_preparator=MyDataPreparator(logger=logger),
    logger=logger
)

my_model.build(my_df=pd.DataFrame(data={
    'row_number': [1, 2],
}))

result = my_model.solve()

print(result.iloc[0]['value'])
print(result.iloc[1]['value'])
