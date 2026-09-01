"""Example showing settings-driven model composition via a model factory."""

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from adapta.logs import LoggerInterface, SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel

from umip import (
    AbstractConstraintBuilder,
    AbstractDataPreparator,
    AbstractDecisionVariableBuilder,
    AbstractMipModel,
    AbstractMipModelFactory,
    AbstractObjectiveBuilder,
    AbstractOptimizationSolver,
    VariableDomain,
)
from umip.abstract_dataclasses import (
    AbstractInputData,
    AbstractInternalData,
    AbstractOutputData,
)
from umip.enums import SolverType


@dataclass
class Settings:
    """
    Feature switches used by the model factory.

    :attr add_bonus_variable: Enables an additional decision variable and builders that depend on it.
    :attr add_tighter_constraint: Enables an extra constraint builder for a stricter feasible region.
    :attr add_bonus_objective: Enables an extra objective function builder.
    """

    add_bonus_variable: bool = False
    add_tighter_constraint: bool = False
    add_bonus_objective: bool = False


@dataclass
class ExampleInputData(AbstractInputData):
    """
    Input data. Empty as no input data required for this example.
    """


@dataclass
class ExampleInternalData(AbstractInternalData):
    """
    Internal model state carrying variables and solved values.

    :attr x_var: Base variable x.
    :attr y_var: Base variable y.
    :attr z_var: Optional variable z.
    :attr x_value: Solved value of x.
    :attr y_value: Solved value of y.
    :attr z_value: Solved value of z.
    """

    x_var: Any | None = None
    y_var: Any | None = None
    z_var: Any | None = None
    x_value: float | None = None
    y_value: float | None = None
    z_value: float | None = None


@dataclass
class ExampleOutputData(AbstractOutputData):
    """Output payload for the example.

    :attr x_value: Solved value of x.
    :attr y_value: Solved value of y.
    :attr z_value: Solved value of z if enabled.
    :attr objective_value: Final objective value.
    """

    x_value: float
    y_value: float
    z_value: float | None
    objective_value: float


class ExampleDataPreparator(AbstractDataPreparator):
    """
    Data preparator for example, that does nothing.
    """

    def prepare(self, input_data: ExampleInputData) -> ExampleInternalData:
        _ = input_data
        return ExampleInternalData()


class BaseVariableBuilder(AbstractDecisionVariableBuilder):
    """Creates the base model variables x and y."""

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> ExampleInternalData:
        data.x_var = solver.add_variable(
            name="x",
            variable_domain=VariableDomain.INTEGER,
            lower_bound=0,
            upper_bound=100,
        )
        data.y_var = solver.add_variable(
            name="y",
            variable_domain=VariableDomain.INTEGER,
            lower_bound=0,
            upper_bound=100,
        )
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> ExampleInternalData:
        data.x_value = solver.get_variable_value(var=data.x_var)
        data.y_value = solver.get_variable_value(var=data.y_var)
        return data


class BonusVariableBuilder(AbstractDecisionVariableBuilder):
    """Creates optional variable z when enabled by settings."""

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> ExampleInternalData:
        data.z_var = solver.add_variable(
            name="z",
            variable_domain=VariableDomain.INTEGER,
            lower_bound=0,
            upper_bound=30,
        )
        return data

    def unpack(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> ExampleInternalData:
        data.z_value = solver.get_variable_value(var=data.z_var)
        return data


class BaseConstraintBuilder(AbstractConstraintBuilder):
    """Adds the always-on constraints."""

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> None:
        solver.add_constraint(
            coefficients=np.array([1.0, 1.0]),
            variables=np.array([data.x_var, data.y_var]),
            lower_bound=None,
            upper_bound=100,
            name="base_capacity",
        )
        solver.add_constraint(
            coefficients=np.array([1.0]),
            variables=np.array([data.y_var]),
            lower_bound=None,
            upper_bound=20,
            name="base_y_cap",
        )


class TighterConstraintBuilder(AbstractConstraintBuilder):
    """Optional extra constraint that can tighten the feasible region."""

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> None:
        solver.add_constraint(
            coefficients=np.array([1.0, -1.0]),
            variables=np.array([data.x_var, data.z_var]),
            lower_bound=0,
            upper_bound=None,
            name="link_x_z",
        )

        solver.add_constraint(
            coefficients=np.array([1.0]),
            variables=np.array([data.x_var]),
            lower_bound=10,
            upper_bound=None,
            name="min_x_without_z",
        )


class BaseObjectiveBuilder(AbstractObjectiveBuilder):
    """Adds the always-on objective function terms."""

    def __init__(self, logger: LoggerInterface) -> None:
        super().__init__(logger=logger)
        self.objective_name = "base_objective"
        self.add_analytics_granularity(
            granularity_name="variable",
            analytics_calculator=self._variable_analytics,
        )
        self.add_analytics_granularity(
            granularity_name="total",
            analytics_calculator=self._total_analytics,
        )

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> None:
        solver.add_multiple_objective_terms(
            coefficients=np.array([1.0, 4.0]),
            variables=np.array([data.x_var, data.y_var]),
        )

    def _variable_analytics(self, analytics_data: ExampleOutputData) -> dict[str, float | None]:
        return {"x": analytics_data.x_value, "y": analytics_data.y_value}

    def _total_analytics(self, analytics_data: ExampleOutputData) -> float:
        variable_analytics = self.get_analytics(granularity="variable", analytics_data=analytics_data)
        return variable_analytics["x"] * 1.0 + variable_analytics["y"] * 4.0


class BonusObjectiveBuilder(AbstractObjectiveBuilder):
    """Adds optional objective function terms for z when enabled by settings."""

    def __init__(self, logger: LoggerInterface) -> None:
        super().__init__(logger=logger)
        self.objective_name = "bonus_objective"
        self.add_analytics_granularity(
            granularity_name="variable",
            analytics_calculator=self._variable_analytics,
        )
        self.add_analytics_granularity(
            granularity_name="total",
            analytics_calculator=self._total_analytics,
        )

    def build(self, solver: AbstractOptimizationSolver, data: ExampleInternalData) -> None:
        if data.z_var is not None:
            solver.add_objective_term(coefficient=2.0, variable=data.z_var)

    def _variable_analytics(self, analytics_data: ExampleOutputData) -> dict[str, float | None]:
        return {"z": analytics_data.z_value}

    def _total_analytics(self, analytics_data: ExampleOutputData) -> float:
        variable_analytics = self.get_analytics(granularity="variable", analytics_data=analytics_data)
        return variable_analytics["z"] * 2.0 if variable_analytics["z"] is not None else 0.0


class ExampleMipModel(AbstractMipModel):
    """Concrete model for demonstrating settings-based composition."""

    def build(
        self,
        input_data: ExampleInputData,
        redirect_solver_log: bool = True,
        **kwargs: Any,
    ) -> None:
        super().build(input_data=input_data, redirect_solver_log=redirect_solver_log, **kwargs)
        self._solver.set_optimization_direction(maximization=True)

    def _convert_internal_to_output_data(
        self, internal_unpacked_data: ExampleInternalData, **kwargs: Any
    ) -> ExampleOutputData:
        return ExampleOutputData(
            x_value=float(internal_unpacked_data.x_value or 0.0),
            y_value=float(internal_unpacked_data.y_value or 0.0),
            z_value=float(internal_unpacked_data.z_value) if internal_unpacked_data.z_value is not None else None,
            objective_value=float(self.get_objective_value()),
        )


class ExampleModelFactory(AbstractMipModelFactory):
    """Factory that composes model builders from settings flags."""

    def construct(self, settings: Settings) -> ExampleMipModel:
        """Construct model components based on settings.

        Builder types are collected into sets before instantiation so that multiple settings
        referencing the same builder type never produce duplicate instances.

        :param settings: Feature-switch settings.
        :return: Composed model ready to build and solve.
        """
        variable_builder_types: set[type[AbstractDecisionVariableBuilder]] = {BaseVariableBuilder}
        constraint_builder_types: set[type[AbstractConstraintBuilder]] = {BaseConstraintBuilder}
        objective_builder_types: set[type[AbstractObjectiveBuilder]] = {BaseObjectiveBuilder}

        if settings.add_bonus_variable:
            variable_builder_types.add(BonusVariableBuilder)

        if settings.add_tighter_constraint:
            constraint_builder_types.add(TighterConstraintBuilder)

        if settings.add_bonus_objective:
            objective_builder_types.add(BonusObjectiveBuilder)

        return ExampleMipModel(
            solver=self._solver,
            data_preparator=ExampleDataPreparator(logger=self._logger),
            variable_builders=[variable_builder(logger=self._logger) for variable_builder in variable_builder_types],
            constraint_builders=[
                constraint_builder(logger=self._logger) for constraint_builder in constraint_builder_types
            ],
            objective_builders=[
                objective_builder(logger=self._logger) for objective_builder in objective_builder_types
            ],
            logger=self._logger,
        )


def _run_case(logger: Any, settings: Settings) -> ExampleMipModel:
    model = ExampleModelFactory(logger=logger, solver_type=SolverType.ORTOOLS_SCIP).construct(settings=settings)
    model.build(input_data=ExampleInputData())
    model.solve()
    return model


logger = SemanticLogger().add_log_source(
    log_source_name="SettingsFactoryExample",
    min_log_level=LogLevel.INFO,
    log_handlers=[SafeStreamHandler(sys.stdout)],
    is_default=True,
)

base_settings = Settings()
feature_settings = Settings(add_bonus_variable=True, add_tighter_constraint=True, add_bonus_objective=True)

base_model = _run_case(logger=logger, settings=base_settings)
feature_model = _run_case(logger=logger, settings=feature_settings)

base_output = base_model.get_output_data()
feature_output = feature_model.get_output_data()

print("Base model:")
print(f"x = {base_output.x_value}, y = {base_output.y_value}")
print(f"Objective: {base_output.objective_value}")
print(base_model.get_analytics(granularity="variable"))
print(base_model.get_analytics(granularity="total"))

print("\nFeature-enabled model:")
print(f"x = {feature_output.x_value}, y = {feature_output.y_value}, z = {feature_output.z_value}")
print(f"Objective: {feature_output.objective_value}")
print(feature_model.get_analytics(granularity="variable"))
print(feature_model.get_analytics(granularity="total"))
