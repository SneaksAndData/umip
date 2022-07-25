"""Abstract definition of a MIP model."""
import time
from typing import List
import pandas as pd
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.abstract_constr_builder import AbstractConstraintBuilder
from generic_mip.abstract_var_builder import AbstractDecisionVariableBuilder
from generic_mip.abstract_obj_builder import AbstractObjectiveBuilder
from generic_mip.abstract_data_prep import AbstractDataPreparator
from generic_mip.abstract_model import AbstractOptimizationModel
from generic_mip.exception import OptimizationException, AbnormalException, \
    InfeasibleException


class AbstractMipModel(AbstractOptimizationModel):
    """
    The MIP model contains builders for variables, constraints and objectives as well as the data preparator.
    The model orchestrates the building and solving processes.
    """
    def __init__(self, solver: AbstractOptimizationSolver, constraint_builders: List[AbstractConstraintBuilder],
                 variable_builders: List[AbstractDecisionVariableBuilder],
                 objective_builders: List[AbstractObjectiveBuilder], data_preparator: AbstractDataPreparator):
        self._objective_builders = objective_builders
        self._variable_builders = variable_builders
        self._constraint_builders = constraint_builders
        self._data_preparator = data_preparator
        self._solver = solver
        self._dfs = None
        self._built = False
        self._solved = False
        self._verbose = False

    def build(self, **input_dfs: pd.DataFrame) -> None:
        start_time = time.time()

        self._dfs = self._data_preparator.prepare(input_dfs)

        if self._verbose:
            print(f"Spent {time.time() - start_time}s preparing data")
        start_time = time.time()

        for variable_builder in self._variable_builders:
            self._dfs = variable_builder.build(
                solver=self._solver,
                input_dfs=self._dfs
            )

        if self._verbose:
            print(f"Spent {time.time() - start_time}s building variables")
        start_time = time.time()

        for constraint_builder in self._constraint_builders:
            constraint_builder.build(
                solver=self._solver,
                input_dfs=self._dfs
            )

        if self._verbose:
            print(f"Spent {time.time() - start_time}s building constraints")
        start_time = time.time()

        for objective_builder in self._objective_builders:
            objective_builder.build(
                solver=self._solver,
                input_dfs=self._dfs
            )

        if self._verbose:
            print(f"Spent {time.time() - start_time}s building objective")
        self._built = True

    def solve(self, **kwargs: any) -> any:
        if not self._built:
            raise ValueError("Model must be built before calling .solve()")

        start_time = time.time()
        status = self._solver.solve()
        exec_time = time.time() - start_time
        if self._verbose:
            print(f"Spent {exec_time}s optimising.")
        self._solved = True

        for variable_builder in self._variable_builders:
            self._dfs = variable_builder.unpack(
                solver=self._solver,
                input_dfs=self._dfs
            )

        if self._solver.is_optimal():
            return None
        if self._solver.is_infeasible():
            raise InfeasibleException("The model is infeasible.")
        if self._solver.is_abnormal():
            raise AbnormalException("The optimization failed with status ABNORMAL.")
        raise OptimizationException(f"Ended with status code {status}.")

    def objective_value(self) -> float:
        if not self._solved:
            raise ValueError("Model must be solved before calling .objective_value()")

        return self._solver.get_objective_value()

    def set_verbose_mode(self, verbose: bool) -> None:
        if self._built:
            raise ValueError("Verbose mode cannot be changed after the model is built")

        self._verbose = verbose
        self._solver.set_verbose(verbose)

    def export_model_to_file(self, path: str) -> None:
        """
        Export model to file.
        :param path: Where to write the model on disk.
        :return:
        """
        self._solver.export_to_file(path)
