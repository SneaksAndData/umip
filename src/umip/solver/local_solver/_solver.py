#  Copyright (c) 2026. ECCO Data & AI and other project contributors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import math

import localsolver as ls
import numpy as np
import numpy.typing as npt
from adapta.logs import LoggerInterface

from umip.abstract_solver import AbstractOptimizationSolver
from umip.enums.variable_domain import VariableDomain
from umip.solver_config import LocalSolverConfig


class LocalSolver(AbstractOptimizationSolver[ls.LSExpression, ls.LSExpression]):  # pylint: disable=too-many-public-methods
    """A solver implemented in the LocalSolver library."""

    def __init__(self, logger: LoggerInterface):
        super().__init__(logger)
        self._solver = ls.LocalSolver()
        self._model = self._solver.get_model()
        self._objective = self._model.create_constant(0)
        self._maximization = True
        self.number_of_variables = 0
        self.number_of_variables_of_type = {variable_type: 0 for variable_type in list(VariableDomain)}
        self.number_of_objective_terms = 0
        self._solution = None

    def __del__(self):
        self._solver.delete()

    def add_constraint(
        self,
        coefficients: npt.NDArray[np.floating | np.integer | np.bool_] | float | bool,
        variables: npt.NDArray[ls.LSExpression] | ls.LSExpression,
        lower_bound: float | bool | None = None,
        upper_bound: float | bool | None = None,
        name: str | None = None,
    ) -> ls.LSExpression | None:
        if lower_bound is None and upper_bound is None:
            return None

        coefficients = self._to_float(value=coefficients)

        expr = self._model.sum(coefficients * variables)

        if name is not None:
            expr.set_name(name)

        constr_lb = (
            self._model.add_constraint(expr >= self._to_float(value=lower_bound)) if lower_bound is not None else None
        )
        constr_ub = (
            self._model.add_constraint(expr <= self._to_float(value=upper_bound)) if upper_bound is not None else None
        )

        return constr_lb or constr_ub

    def get_constraint(self, name: str) -> ls.LSExpression:
        return self._model.get_expression(name)

    def add_multiple_constraints(
        self,
        coefficients: npt.NDArray[npt.NDArray[np.floating | np.integer | np.bool_]]
        | npt.NDArray[np.floating | np.integer | np.bool_],
        variables: npt.NDArray[npt.NDArray[ls.LSExpression]] | npt.NDArray[ls.LSExpression],
        lower_bounds: npt.NDArray[np.floating | np.integer | np.bool_] | None = None,
        upper_bounds: npt.NDArray[np.floating | np.integer | np.bool_] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        if coefficients.size == 0:
            return

        coefficients = self._to_float(value=coefficients)
        lower_bounds = self._to_float(value=lower_bounds) if lower_bounds is not None else None
        upper_bounds = self._to_float(value=upper_bounds) if upper_bounds is not None else None

        if names is not None and len(names) != len(coefficients):
            raise ValueError("The number of names must match the number of constraints")

        num_constrs = len(coefficients)
        for i in range(num_constrs):
            self.add_constraint(
                coefficients=coefficients[i],
                variables=variables[i],
                lower_bound=lower_bounds[i] if lower_bounds is not None else None,
                upper_bound=upper_bounds[i] if upper_bounds is not None else None,
                name=f"{names[i]}" if names is not None else None,
            )

    def add_variable(
        self,
        name: str,
        variable_domain: VariableDomain,
        lower_bound: float | bool | None = None,
        upper_bound: float | bool | None = None,
    ) -> ls.LSExpression:
        lower_bound = self._to_float(value=lower_bound) if lower_bound is not None else -self.infinity()
        upper_bound = self._to_float(value=upper_bound) if upper_bound is not None else self.infinity()
        self.number_of_variables += 1
        self.number_of_variables_of_type[variable_domain] += 1
        if variable_domain == VariableDomain.INTEGER:
            var = self._model.int(math.ceil(lower_bound), math.floor(upper_bound))
            self._integer_problem = True
        elif variable_domain == VariableDomain.BINARY:
            var = self._model.bool()
            self._integer_problem = True
        elif variable_domain == VariableDomain.CONTINUOUS:
            var = self._model.float(lower_bound, upper_bound)
        else:
            raise ValueError(f"Unknown variable domain: {variable_domain}")

        if name is not None:
            var.set_name(name)

        return var

    def add_multiple_variables(
        self,
        names: npt.NDArray[str],
        variable_domain: VariableDomain,
        lower_bound: float | bool | None = None,
        upper_bound: float | bool | None = None,
    ) -> npt.NDArray[ls.LSExpression]:
        lower_bound = self._to_float(value=lower_bound) if lower_bound is not None else -self.infinity()
        upper_bound = self._to_float(value=upper_bound) if upper_bound is not None else self.infinity()
        return np.array(  # pylint: disable=duplicate-code
            [
                self.add_variable(
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    name=f"{name}",
                    variable_domain=variable_domain,
                )
                for name in names
            ]
        )

    def set_variable_hint(self, variable: ls.LSExpression, hint: float | bool) -> None:
        raise NotImplementedError()

    def set_multiple_variable_hints(
        self,
        variables: npt.NDArray[ls.LSExpression],
        hints: npt.NDArray[np.floating | np.integer | np.bool_],
    ) -> None:
        raise NotImplementedError()

    def add_objective_term(
        self,
        coefficient: float | bool,
        variable: ls.LSExpression,
        overwrite: bool = True,
        name: str | None = None,
    ) -> None:
        coefficient = self._to_float(value=coefficient)
        if name is not None:
            self.add_named_objective(np.array([coefficient]), np.array([variable]), overwrite, name)

        if overwrite:
            self._objective += coefficient * variable
            self.number_of_objective_terms += 1
        else:
            raise NotImplementedError()

    def add_multiple_objective_terms(
        self,
        coefficients: npt.NDArray[np.floating | np.integer | np.bool_],
        variables: npt.NDArray[ls.LSExpression],
        overwrite: bool = True,
        name: str | None = None,
    ) -> None:
        coefficients = self._to_float(value=coefficients)
        if name is not None:
            self.add_named_objective(coefficients, variables, overwrite, name)

        if overwrite:
            self._objective = self._model.sum(coefficients * variables) + self._objective
            self.number_of_objective_terms += len(coefficients)
        else:
            raise NotImplementedError()

    def set_optimization_direction(self, maximization: bool) -> None:
        self._maximization = maximization

    def get_objective_value(self) -> float:
        return self._solution.get_value(self._objective)

    def solve(self, time_limit: float | None = None, mip_gap_limit: float | None = None) -> int:
        if time_limit is not None:
            self._solver.get_param().set_time_limit(time_limit)
        if mip_gap_limit is not None:
            raise NotImplementedError()
        self._model.add_objective(
            self._objective,
            ls.LSObjectiveDirection.MAXIMIZE if self._maximization else ls.LSObjectiveDirection.MINIMIZE,
        )
        self._model.close()
        self._solver.solve()
        self._solution = self._solver.get_solution()
        return self._solution.get_status()

    def infinity(self) -> float:
        return 100000000

    def is_optimal(self) -> bool:
        return (
            self._solution.get_status() == ls.LSSolutionStatus.OPTIMAL and self.get_objective_value() <= self.infinity()
        )

    def is_feasible(self) -> bool:
        return (
            self._solution.get_status() == ls.LSSolutionStatus.FEASIBLE
            and self.get_objective_value() <= self.infinity()
        )

    def is_infeasible(self) -> bool:
        return self._solution.get_status() in [
            ls.LSSolutionStatus.INFEASIBLE,
            ls.LSSolutionStatus.INCONSISTENT,
        ]

    def is_abnormal(self) -> bool:
        return False

    def is_unbounded(self) -> bool:
        return (
            self._solution.get_status() == ls.LSSolutionStatus.OPTIMAL and self.get_objective_value() >= self.infinity()
        )

    def get_variable_value(self, var: ls.LSExpression) -> float:
        return self._solution.get_value(var)

    def export_to_file(self, path: str) -> None:
        self._solver.save_environment(path)

    def set_verbose(self, verbose: bool) -> None:
        self._solver.get_param().set_verbosity(2 if verbose else 0)

    def set_solver_setting(self, setting: LocalSolverConfig) -> None:
        raise NotImplementedError()

    def get_variable_count(self):
        return self.number_of_variables

    def get_variable_count_of_type(self, variable_domain: VariableDomain):
        return self.number_of_variables_of_type[variable_domain]

    def get_constraint_count(self):
        return self._model.get_nb_constraints()

    def get_objective_terms_count(self):
        return self.number_of_objective_terms

    def force_update(self):
        pass  # LocalSolver is eager

    def is_not_solved(self) -> bool:
        return False

    def get_gap(self) -> float:
        return self._solution.get_objective_gap(0)

    def get_dual_value(self, constraint: ls.LSExpression) -> float:
        raise NotImplementedError("LocalSolver does not support dual values")

    def get_named_objective(self, name: str) -> float:
        if name in self._named_objectives:
            return float(
                np.sum(
                    [
                        (item.objective_coefficient * self.get_variable_value(item.variable))
                        for item in self._named_objectives[name]
                        if item.objective_coefficient
                    ]
                )
            )
        return 0.0
