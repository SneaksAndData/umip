"""Abstract definition of a MIP model."""
import time
import sys
from abc import abstractmethod, ABC
from typing import Any
from adapta.logs import LoggerInterface, SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel
from adapta.metrics import MetricsProvider
from adapta.metrics.providers.void_provider import VoidMetricsProvider
from adapta.utils.decorators import run_time_metrics_async

from generic_mip.enums.variable_domain import VariableDomain
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.abstract_constr_builder import AbstractConstraintBuilder
from generic_mip.abstract_var_builder import AbstractDecisionVariableBuilder
from generic_mip.abstract_obj_builder import AbstractObjectiveBuilder
from generic_mip.abstract_data_prep import AbstractDataPreparator
from generic_mip.exception import OptimizationException, AbnormalException, InfeasibleException, UnboundedException
from generic_mip.abstract_dataclasses import AbstractInputData, AbstractOutputData, AbstractInternalData


class AbstractMipModel(ABC):
    """
    The MIP model contains builders for variables, constraints and objectives as well as the data preparator.
    The model orchestrates the building and solving processes.
    """

    def __init__(
        self,
        solver: AbstractOptimizationSolver,
        data_preparator: AbstractDataPreparator,
        constraint_builders: list[AbstractConstraintBuilder],
        variable_builders: list[AbstractDecisionVariableBuilder],
        objective_builders: list[AbstractObjectiveBuilder],
        logger: LoggerInterface = SemanticLogger().add_log_source(
            log_source_name="AbstractMipModel",
            min_log_level=LogLevel.INFO,
            log_handlers=[SafeStreamHandler(sys.stdout)],
            is_default=True,
        ),
        metrics_provider: MetricsProvider = VoidMetricsProvider(),
    ):
        """
        Initialize the MIP model.
        :param solver: The solver implementation to use.
        :param constraint_builders: The builders of the model constraints.
        :param variable_builders: The builders of the model decision variables.
        :param objective_builders: The builders of the model objective terms.
        :param data_preparator: The data preparator.
        :param logger: The logger to use. Logging to stdout by default.
        :param metrics_provider: The MetricsProvider to use. Metrics are ignored by default.
        """
        self._objective_builders = objective_builders
        self._variable_builders = variable_builders
        self._constraint_builders = constraint_builders
        self._data_preparator = data_preparator
        self._solver = solver
        self._internal_data: AbstractInternalData | None = None
        self._output_data: AbstractOutputData | None = None
        self._built = False
        self._solved = False
        self._logger = logger
        self._metrics_provider = metrics_provider
        self._objective_builder_names = []

    def build(self, input_data: AbstractInputData) -> None:
        """
        Builds the model using the given variable, constraint and objective builders.
        :param input_data: Input data to the variables, constraints and objectives.
        """
        start_time = time.time()

        self._internal_data = self._data_preparator.prepare(input_data=input_data)

        self._logger.info(template="Spent {time}s preparing data", time=time.time() - start_time)
        start_time = time.time()

        with self._logger.redirect(log_level=LogLevel.INFO):
            self._build_variables()

        self._log_variables()

        self._logger.info(template="Spent {time}s building variables", time=time.time() - start_time)
        start_time = time.time()

        with self._logger.redirect(log_level=LogLevel.INFO):
            self._build_constraints()

        self._logger.info(
            template="Number of constraints: {number_of_constraints}",
            number_of_constraints=self._solver.get_constraint_count(),
        )

        self._logger.info(template="Spent {time}s building constraints", time=time.time() - start_time)
        start_time = time.time()

        with self._logger.redirect(log_level=LogLevel.INFO):
            self._build_objectives()

        self._logger.info(template="Spent {time}s building objective", time=time.time() - start_time)
        self._built = True

    def _build_variables(self):
        for variable_builder in self._variable_builders:
            self._internal_data = variable_builder.build(solver=self._solver, data=self._internal_data)

    def _log_variables(self):
        self._logger.info(
            template="Number of variables: {number_of_variables}", number_of_variables=self._solver.get_variable_count()
        )
        self._logger.info(
            template="Number of continuous variables: {number_of_continuous_variables}",
            number_of_continuous_variables=self._solver.get_variable_count_of_type(VariableDomain.CONTINUOUS),
        )
        self._logger.info(
            template="Number of binary variables: {number_of_binary_variables}",
            number_of_binary_variables=self._solver.get_variable_count_of_type(VariableDomain.BINARY),
        )
        self._logger.info(
            template="Number of integer variables: {number_of_integer_variables}",
            number_of_integer_variables=self._solver.get_variable_count_of_type(VariableDomain.INTEGER),
        )

    def _build_constraints(self):
        for constraint_builder in self._constraint_builders:
            constraint_builder.build(solver=self._solver, data=self._internal_data)

    def _build_objectives(self):
        for objective_builder in self._objective_builders:
            objective_builder.build(solver=self._solver, data=self._internal_data)
            if objective_builder.objective_name in self._objective_builder_names:
                raise ValueError(f"Duplicate objective builder name found: {objective_builder.objective_name}")
            self._objective_builder_names.append(objective_builder.objective_name)

    async def build_async(self, input_data: AbstractInputData) -> None:
        """
        Async version of self.build.
        :param input_data: Input data to the variables, constraints and objectives.
        """

        @run_time_metrics_async(
            metric_name="mip_data_preparation",
            on_finish_message_template="Finished preparing data for {model} in {elapsed:.4f}s seconds",
            template_args={
                "model": self.__class__.__name__,
            },
        )
        async def _prepare_async(**_):
            self._internal_data = self._data_preparator.prepare(input_data=input_data)

        @run_time_metrics_async(
            metric_name="mip_build_variables",
            on_finish_message_template="Finished building variables for {model} in {elapsed:.4f}s seconds",
            template_args={
                "model": self.__class__.__name__,
            },
        )
        async def _build_variables_async(**_):
            self._build_variables()
            self._log_variables()

        @run_time_metrics_async(
            metric_name="mip_build_constraints",
            on_finish_message_template="Finished building constraints for {model} in {elapsed:.4f}s seconds",
            template_args={
                "model": self.__class__.__name__,
            },
        )
        async def _build_constraints_async(**_):
            self._build_constraints()
            self._logger.info(
                template="Number of constraints: {number_of_constraints}",
                number_of_constraints=self._solver.get_constraint_count(),
            )

        @run_time_metrics_async(
            metric_name="mip_build_objectives",
            on_finish_message_template="Finished building objectives for {model} in {elapsed:.4f}s seconds",
            template_args={
                "model": self.__class__.__name__,
            },
        )
        async def _build_objectives_async(**_):
            self._build_objectives()

        async with self._logger.redirect_async(log_level=LogLevel.INFO):
            for _step in [
                _prepare_async(logger=self._logger, metrics_provider=self._metrics_provider),
                _build_variables_async(logger=self._logger, metrics_provider=self._metrics_provider),
                _build_constraints_async(logger=self._logger, metrics_provider=self._metrics_provider),
                _build_objectives_async(logger=self._logger, metrics_provider=self._metrics_provider),
            ]:
                await _step

        self._built = True

    def solve(self, time_limit: float | None = None, redirect_solver_log: bool = True) -> None:
        """
        Solves the model and returns the result of the optimization.
        :param time_limit: The time limit of the optimization in seconds.
        :param redirect_solver_log: Whether to redirect stdout to a temporary file
        """
        if not self._built:
            raise ValueError("Model must be built before calling .solve()")

        start_time = time.time()
        if redirect_solver_log:
            with self._logger.redirect(log_level=LogLevel.INFO):
                status = self._solver.solve(time_limit=time_limit)
        else:
            status = self._solver.solve(time_limit=time_limit)
        exec_time = time.time() - start_time
        self._logger.info(template="Spent {time}s optimising.", time=exec_time)
        self._solved = True

        if self._solver.is_optimal() or self._solver.is_feasible():
            with self._logger.redirect(log_level=LogLevel.INFO):
                for variable_builder in self._variable_builders:
                    self._internal_data = variable_builder.unpack(solver=self._solver, data=self._internal_data)
            return None

        if self._solver.is_infeasible():
            raise InfeasibleException("The model is infeasible.")
        if self._solver.is_unbounded():
            raise UnboundedException("The model is unbounded.")
        if self._solver.is_abnormal():
            raise AbnormalException("The optimization failed with status ABNORMAL.")
        raise OptimizationException(f"Ended with status code {status}.")

    async def solve_async(self, time_limit: float | None = None, redirect_solver_log: bool = True) -> None:
        """
        Async version of self.solve.
        :param time_limit: The time limit of the optimization in seconds.
        :param redirect_solver_log: Whether to redirect stdout to a temporary file
        :return: (optimization status code, boolean indicating whether the model has been solved)
        """

        @run_time_metrics_async(
            metric_name="mip_solve",
            on_finish_message_template="Finished solving {model} in {elapsed:.4f}s seconds",
            template_args={
                "model": self.__class__.__name__,
            },
        )
        async def _solve(**_) -> tuple[int, bool]:
            if redirect_solver_log:
                async with self._logger.redirect_async(log_level=LogLevel.INFO):
                    status = self._solver.solve(time_limit=time_limit)
            else:
                status = self._solver.solve(time_limit=time_limit)

            self._solved = True

            if self._solver.is_optimal() or self._solver.is_feasible():
                async with self._logger.redirect_async(log_level=LogLevel.INFO):
                    for variable_builder in self._variable_builders:
                        self._internal_data = variable_builder.unpack(solver=self._solver, data=self._internal_data)

                return status, True

            return status, False

        if not self._built:
            raise ValueError("Model must be built before calling .solve()")

        result_status_code, is_solved = await _solve(logger=self._logger, metrics_provider=self._metrics_provider)

        if not is_solved:
            if self._solver.is_infeasible():
                raise InfeasibleException("The model is infeasible.")
            if self._solver.is_unbounded():
                raise UnboundedException("The model is unbounded.")
            if self._solver.is_abnormal():
                raise AbnormalException("The optimization failed with status ABNORMAL.")
            raise OptimizationException(f"Ended with status code {result_status_code}.")

    def objective_value(self) -> float:
        """
        Get the objective value of the optimization.
        :return: The objective value.
        """
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

    def get_gap(self) -> float:
        """
        Get the gap of the model.
        :return: The gap of the model.
        """
        if not self._solved:
            raise ValueError("Model must be solved before calling .get_gap()")
        return self._solver.get_gap()

    def get_analytics(self, granularity: str, analytics_data: Any | None = None) -> dict[str, Any]:
        """
        Get analytics for the specified granularity from the objective builders. If analytics_data is not provided,
        the model must have been solved already and must contain the necessary data in self._output_data.

        Returns a dictionary where each key is an objective builder name and the value is its analytics.
        """
        if not self._solved and not analytics_data:
            raise ValueError("Model must be solved or analytics data must be provided before calling .get_analytics()")

        analytics_results = {}

        analytics_data = analytics_data if analytics_data is not None else self._output_data

        for objective_builder in self._objective_builders:
            if granularity in objective_builder.get_supported_analytics_granularities():
                objective_analytics = objective_builder.get_analytics(
                    granularity=granularity, analytics_data=analytics_data
                )
                analytics_results[objective_builder.objective_name] = objective_analytics

        if not analytics_results:
            self._logger.warning(
                f"No analytics found for granularity '{granularity}' in objective builders: "
                f"{[builder.objective_name for builder in self._objective_builders]}"
            )

        return analytics_results

    def get_objective_analytics_granularities(self) -> dict[str, list[str]]:
        """
        Get supported analytics granularities for each objective builder.

        Returns a dictionary where each key is an objective builder name and the value is a list of its supported
        granularities.
        """
        granularities = {}

        for objective_builder in self._objective_builders:
            granularities[objective_builder.objective_name] = objective_builder.get_supported_analytics_granularities()

        return granularities

    def get_objective_builder_names(self) -> list[str]:
        """
        Get the names of all objective builders in the model.
        :return: List of objective builder names.
        """
        return self._objective_builder_names

    def is_solved(self) -> bool:
        """
        Check if the model has been solved.
        :return: True if the model has been solved, False otherwise.
        """
        return self._solved

    def get_output_data(self) -> AbstractOutputData:
        """
        Get the data after solving the model.
        """
        if not self._solved:
            raise ValueError("Model must be solved before calling .get_output_data()")

        if not self._output_data:
            self._output_data = self._convert_internal_to_output_data(internal_data=self._internal_data)

        return self._output_data

    @abstractmethod
    def _convert_internal_to_output_data(self, internal_data: AbstractInternalData) -> AbstractOutputData:
        """
        Converts the internal data post optimization to output data.
        """
