"""Abstract definition of a MIP model."""
import time
import sys
from typing import List
from logging import StreamHandler
from proteus.logs import ProteusLogger
from proteus.logs.models import LogLevel
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
    def __init__(
        self,
        solver: AbstractOptimizationSolver,
        constraint_builders: List[AbstractConstraintBuilder],
        variable_builders: List[AbstractDecisionVariableBuilder],
        objective_builders: List[AbstractObjectiveBuilder],
        data_preparator: AbstractDataPreparator,
        logger: ProteusLogger = ProteusLogger().add_log_source(
            log_source_name='AbstractMipModel',
            min_log_level=LogLevel.INFO,
            log_handlers=[StreamHandler(sys.stdout)],
            is_default=True
        )
    ):
        """
        Initialize the MIP model.
        :param solver: The solver implementation to use.
        :param constraint_builders: The builders of the model constraints.
        :param variable_builders: The builders of the model decision variables.
        :param objective_builders: The builders of the model objective terms.
        :param data_preparator: The data preparator.
        :param logger: The logger to use. Logging to stdout by default.
        """
        self._objective_builders = objective_builders
        self._variable_builders = variable_builders
        self._constraint_builders = constraint_builders
        self._data_preparator = data_preparator
        self._solver = solver
        self._dfs = None
        self._built = False
        self._solved = False
        self._logger = logger

    def build(self, **input_dfs: pd.DataFrame) -> None:
        start_time = time.time()

        self._dfs = self._data_preparator.prepare(input_dfs)

        self._logger.info(template="Spent {time}s preparing data", time=time.time() - start_time)
        start_time = time.time()

        with self._logger.redirect(log_level=LogLevel.INFO):
            for variable_builder in self._variable_builders:
                self._dfs = variable_builder.build(
                    solver=self._solver,
                    input_dfs=self._dfs
                )

        self._logger.info(template="Spent {time}s building variables", time=time.time() - start_time)
        start_time = time.time()

        with self._logger.redirect(log_level=LogLevel.INFO):
            for constraint_builder in self._constraint_builders:
                constraint_builder.build(
                    solver=self._solver,
                    input_dfs=self._dfs
                )

        self._logger.info(template="Spent {time}s building constraints", time=time.time() - start_time)
        start_time = time.time()

        with self._logger.redirect(log_level=LogLevel.INFO):
            for objective_builder in self._objective_builders:
                objective_builder.build(
                    solver=self._solver,
                    input_dfs=self._dfs
                )

        self._logger.info(template="Spent {time}s building objective", time=time.time() - start_time)
        self._built = True

    def solve(self, **kwargs: any) -> any:
        if not self._built:
            raise ValueError("Model must be built before calling .solve()")

        start_time = time.time()
        with self._logger.redirect(log_level=LogLevel.INFO):
            status = self._solver.solve()
        exec_time = time.time() - start_time
        self._logger.info(template="Spent {time}s optimising.", time=exec_time)
        self._solved = True

        with self._logger.redirect(log_level=LogLevel.INFO):
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

    def export_model_to_file(self, path: str) -> None:
        """
        Export model to file.
        :param path: Where to write the model on disk.
        :return:
        """
        self._solver.export_to_file(path)
