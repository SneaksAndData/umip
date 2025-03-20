"""A solver implemented in the Gurobi library."""
from typing import Iterable

from docplex.mp.model import Model
from docplex.mp.linear import Var, LinearExpr
from docplex.mp.constr import LinearConstraint
from docplex.mp.constants import ObjectiveSense
from docplex.mp.model_reader import ModelReader

import numpy as np
import numpy.typing as npt

from adapta.logs import LoggerInterface
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.solver.cplex.enum import CplexStatus
from generic_mip.variable_data_type import VariableDataType


class CplexSolver(AbstractOptimizationSolver[Var, LinearConstraint]):  # pylint: disable=too-many-public-methods
    """A solver implemented in the Cplex library."""

    def __init__(self, logger: LoggerInterface, model_path: str | None = None):
        """
        Initialize the solver.

        :param model_path: The path to the model file to read (mps or lp).
        :param logger: The logger to use.
        """
        super().__init__(logger)
        self._solver = Model()
        if model_path is not None:
            self._solver = ModelReader.read(model_path)
        self._objective = LinearExpr(self._solver)
        self.status = None
        self.solution = None

    def set_variable_hint(self, var: Var, hint: float) -> None:
        raise NotImplementedError()

    def set_multiple_variable_hints(self, vars_: npt.NDArray[Var], hints: npt.NDArray[float]) -> None:
        raise NotImplementedError()

    def add_multiple_objective_terms(
        self, coeffs: npt.NDArray[float], vars_: npt.NDArray[Var], overwrite: bool = True, name: str = None
    ) -> None:
        if name is not None:
            self.add_named_objective(coeffs, vars_, overwrite, name)

        for coeff, var in zip(coeffs, vars_):
            if overwrite:
                self._objective.set_coefficient(var, coeff)
            else:
                self._objective.add_term(var, coeff)

    def add_multiple_variables(
        self, names: npt.NDArray[str], dtype: VariableDataType, lb: float | None = None, ub: float | None = None
    ) -> npt.NDArray[Var]:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        count = len(names)

        if dtype == VariableDataType.INT:
            vars_ = self._solver.integer_var_list(keys=count, lb=lb, ub=ub)
            self._integer_problem = True
        elif dtype == VariableDataType.FLOAT:
            vars_ = self._solver.continuous_var_list(keys=count, lb=lb, ub=ub)
        elif dtype == VariableDataType.BOOL:
            vars_ = self._solver.binary_var_list(keys=count, lb=lb, ub=ub)
            self._integer_problem = True
        else:
            raise ValueError("Unsupported variable data type")
        return vars_

    def add_multiple_constraints(
        self,
        coeffs: npt.NDArray[npt.NDArray[float]] | npt.NDArray[float],
        vars_: npt.NDArray[npt.NDArray[Var]] | npt.NDArray[Var],
        lb: npt.NDArray[float] | None = None,
        ub: npt.NDArray[float] | None = None,
        names: str | None = None,
    ) -> None:
        if coeffs.size == 0:
            return

        if lb is not None:
            for i, coeff in enumerate(coeffs):
                self._solver.add_constraint(self._solver.dot(vars_[i], coeff) >= lb[i])
        if ub is not None:
            for i, coeff in enumerate(coeffs):
                self._solver.add_constraint(self._solver.dot(vars_[i], coeff) <= ub[i])

    def add_constraint(
        self,
        coeffs: npt.NDArray[float] | float,
        vars_: npt.NDArray[Var] | Var,
        lb: float | None = None,
        ub: float | None = None,
        name: str | None = None,
    ) -> int | None:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        constr_expr = self._solver.linear_expr()
        if isinstance(vars_, Iterable):
            for coeff, var in zip(coeffs, vars_):
                constr_expr += coeff * var
        else:
            constr_expr = coeffs * vars_

        constraint_lb = None
        constraint_ub = None
        if lb == ub:
            return self._solver.add_constraint(lb == constr_expr, name)

        if lb != -self.infinity():
            constraint_lb = self._solver.add_constraint(lb <= constr_expr, name)
        if ub != self.infinity():
            constraint_ub = self._solver.add_constraint(constr_expr <= ub, name)
        if constraint_lb is not None:
            return constraint_lb
        return constraint_ub

    def add_variable(self, name: str, dtype: VariableDataType, lb: float | None = None, ub: float | None = None) -> int:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        if dtype == VariableDataType.INT:
            var = self._solver.integer_var(lb, ub, name)
            self._integer_problem = True
        elif dtype == VariableDataType.FLOAT:
            var = self._solver.continuous_var(lb, ub, name)
        elif dtype == VariableDataType.BOOL:
            var = self._solver.binary_var(name)
            self._integer_problem = True
        else:
            raise ValueError("Unsupported variable data type")
        return var

    def get_variable_value(self, var: Var) -> float:
        return self.solution.get_value(var)

    def add_objective_term(self, coeff: float, var: Var, overwrite: bool = True, name: str = None) -> None:
        if name is not None:
            self.add_named_objective(np.array([coeff]), np.array([var]), overwrite, name)

        if overwrite:
            self._objective.set_coefficient(var, coeff)
        else:
            self._objective.add_term(var, coeff)

    def set_optimization_direction(self, maximization: bool) -> None:
        if maximization:
            self._solver.objective_sense = ObjectiveSense.Maximize
        else:
            self._solver.objective_sense = ObjectiveSense.Minimize

    def get_objective_value(self) -> float:
        return self.solution.get_objective_value()

    def solve(self, time_limit: float | None = None, mip_gap_limit: float | None = None) -> str:
        if time_limit is not None:
            self._solver.set_time_limit(time_limit)
        if mip_gap_limit is not None:
            self._solver.parameters.mip.tolerances.mipgap = mip_gap_limit
        if self._solver.objective_sense == ObjectiveSense.Maximize:
            self._solver.maximize(self._objective)
        elif self._solver.objective_sense == ObjectiveSense.Minimize:
            self._solver.minimize(self._objective)
        self.solution = self._solver.solve()
        self.status = str(self._solver.solve_details.status)
        return self.status

    def get_constraint(self, name: str):
        return self._solver.get_constraint_by_name(name)

    def infinity(self) -> float:
        return self._solver.infinity

    def is_optimal(self) -> bool:
        return self.status in [CplexStatus.OPTIMAL.value, CplexStatus.INTEGER_OPTIMAL.value]

    def is_feasible(self) -> bool:
        return self.status == CplexStatus.FEASIBLE.value

    def is_infeasible(self) -> bool:
        return self.status in [CplexStatus.INFEASIBLE.value, CplexStatus.INTEGER_INFEASIBLE.value]

    def is_unbounded(self) -> bool:
        return CplexStatus.UNBOUNDED.value in self.status.lower()

    def is_abnormal(self) -> bool:
        return self.status in [
            CplexStatus.NUMERICAL_DIFFICULTIES.value,
            CplexStatus.ABORTED.value,
            CplexStatus.UNKNOWN.value,
        ]

    def is_not_solved(self) -> bool:
        return self.status in [
            CplexStatus.TIME_LIMIT.value,
            CplexStatus.ITERATION_LIMIT.value,
            CplexStatus.INTEGER_LIMIT.value,
        ]

    def set_solver_setting(self, setting: str) -> None:
        raise ValueError("Not supported in Cplex solver")

    def export_to_file(self, path: str) -> None:
        self._solver.export_as_lp(path)

    def set_verbose(self, verbose: bool) -> None:
        self._solver.log_output = verbose

    def get_constraint_count(self):
        return self._solver.number_of_constraints

    def get_variable_count(self):
        return self._solver.number_of_variables

    def get_variable_count_of_type(self, var_type: VariableDataType):
        if var_type == VariableDataType.FLOAT:
            return self._solver.number_of_continuous_variables
        if var_type == VariableDataType.INT:
            return self._solver.number_of_integer_variables
        if var_type == VariableDataType.BOOL:
            return self._solver.number_of_binary_variables

        raise ValueError(f"Unsupported variable data type: {var_type}")

    def get_objective_terms_count(self):
        return self._objective.number_of_terms()

    def force_update(self):  # there is no need to manually update the model in CPLEX, it is done automatically
        pass

    def get_gap(self) -> float:
        if self.is_infeasible() or self.solution is None:
            return self.infinity()
        return self._solver.solve_details.mip_relative_gap

    def add_objective_offset(self, offset: float, overwrite: bool = True):
        if overwrite:
            self._objective.constant = 0
        self._objective.constant += offset

    def get_dual_value(self, constraint: LinearConstraint) -> float:
        if self._integer_problem:
            raise ValueError("Dual values are not available for integer problems")
        if not self.is_optimal():
            raise ValueError("Dual values are only available for optimal solutions")
        return constraint.dual_value
