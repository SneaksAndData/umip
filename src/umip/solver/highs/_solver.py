"""A solver implemented in the HiGHS library."""

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

import highspy
import numpy as np
import numpy.typing as npt
from adapta.logs import LoggerInterface

from umip.abstract_solver import AbstractOptimizationSolver
from umip.enums.variable_domain import VariableDomain
from umip.solver_config import HighsSolverConfig


class HighsSolver(AbstractOptimizationSolver[int, int]):  # pylint: disable=too-many-public-methods
    """A solver implemented in the HiGHS library."""

    def __init__(self, logger: LoggerInterface, model_path: str | None = None):
        """
        Initialize the solver.

        :param model_path: The path to the model file to read (mps or lp).
        :param logger: The logger to use.
        """
        super().__init__(logger)
        self._dual_values: list[float] | None = None
        self._var_count = 0
        self._constr_count = 0
        self._obj_count = 0
        self._solver = highspy.Highs()
        self.number_of_variables_of_type = {variable_type: 0 for variable_type in list(VariableDomain)}
        if model_path is not None:
            self._solver.readModel(model_path)
        self.status: highspy.HighsModelStatus | None = None
        self._solution: list[float] | None = None

    def set_variable_hint(self, variable: str, hint: float | bool) -> None:
        raise NotImplementedError()

    def _get_var_by_name(self, name: str) -> int:
        """
        Get the variable index (int) by its name, as highs references variables by their index.

        :param name: The name of the variable.
        :return: The index of the variable.
        """
        return self._solver.getColByName(name)[1]

    def _get_vars_by_names(self, names: npt.NDArray[npt.NDArray[str]] | npt.NDArray[str]) -> npt.NDArray[int]:
        """
        Get the variable indices (int) by their names, as highs references variables by their index.

        :param names: The names of the variables.
        :return: The indices of the variables.
        """
        if len(names) == 0:
            return np.array([], dtype=int)
        if names.ndim > 1 or isinstance(names[0], (np.ndarray, list, set)):
            names = np.concatenate(names)
        return np.array([self._solver.getColByName(name)[1] for name in names])

    def _get_constraint_by_name(self, name: str) -> int:
        return self._solver.getRowByName(name)[1]

    def set_multiple_variable_hints(
        self,
        variables: npt.NDArray[str],
        hints: npt.NDArray[np.floating | np.integer | np.bool_],
    ) -> None:
        raise NotImplementedError()

    def add_multiple_objective_terms(
        self,
        coefficients: npt.NDArray[np.floating | np.integer | np.bool_],
        variables: npt.NDArray[str],
        overwrite: bool = True,
        name: str | None = None,
    ) -> None:
        coefficients = self._to_float(value=coefficients)
        if name is not None:
            self.add_named_objective(coefficients, self._get_vars_by_names(variables), overwrite, name)

        for i in range(len(coefficients)):  # pylint: disable=consider-using-enumerate
            self.add_objective_term(coefficients[i], variables[i], overwrite=overwrite)

    def add_multiple_variables(
        self,
        names: npt.NDArray[str],
        variable_domain: VariableDomain,
        lower_bound: float | bool | None = None,
        upper_bound: float | bool | None = None,
    ) -> npt.NDArray[str]:
        lower_bound = self._to_float(value=lower_bound) if lower_bound is not None else -self.infinity()
        upper_bound = self._to_float(value=upper_bound) if upper_bound is not None else self.infinity()

        first_var_number = self._var_count
        self._var_count += len(names)

        lower_bounds = np.full(shape=len(names), fill_value=lower_bound, dtype=np.float64)
        upper_bounds = np.full(shape=len(names), fill_value=upper_bound, dtype=np.float64)
        self._solver.addVars(len(names), lower_bounds, upper_bounds)

        for i, name in enumerate(names):
            self.number_of_variables_of_type[variable_domain] += 1
            if variable_domain == VariableDomain.INTEGER:
                self._integer_problem = True
                self._solver.changeColIntegrality(first_var_number + i, highspy.HighsVarType.kInteger)
            elif variable_domain == VariableDomain.CONTINUOUS:
                self._solver.changeColIntegrality(first_var_number + i, highspy.HighsVarType.kContinuous)
            elif variable_domain == VariableDomain.BINARY:
                self._integer_problem = True
                self._solver.changeColIntegrality(first_var_number + i, highspy.HighsVarType.kInteger)
                self._solver.changeColBounds(first_var_number + i, 0, 1)
            else:
                raise ValueError("Unsupported variable data type")
            self._solver.passColName(first_var_number + i, name)

        return names

    def add_multiple_constraints(
        self,
        coefficients: npt.NDArray[npt.NDArray[np.floating | np.integer | np.bool_]]
        | npt.NDArray[np.floating | np.integer | np.bool_],
        variables: npt.NDArray[npt.NDArray[str]] | npt.NDArray[str],
        lower_bounds: npt.NDArray[np.floating | np.integer | np.bool_] | None = None,
        upper_bounds: npt.NDArray[np.floating | np.integer | np.bool_] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        if lower_bounds is None and upper_bounds is None:
            raise ValueError("Either lb or ub must be specified")

        coefficients = self._to_float(value=coefficients)

        if lower_bounds is None:
            upper_bounds = self._to_float(value=upper_bounds)
            lower_bounds = np.full(len(upper_bounds), -self.infinity())
        elif upper_bounds is None:
            lower_bounds = self._to_float(value=lower_bounds)
            upper_bounds = np.full(len(lower_bounds), self.infinity())
        else:
            lower_bounds = self._to_float(value=lower_bounds)
            upper_bounds = self._to_float(value=upper_bounds)

        if len(lower_bounds) == 0:
            return

        self._constr_count += len(lower_bounds)

        if coefficients.ndim == 1 and not isinstance(coefficients[0], (np.ndarray, list, set)):
            flat_coeffs = coefficients
            flat_vars = self._get_vars_by_names(variables)
            starts = np.arange(0, len(flat_coeffs) + 1)
        else:
            flat_coeffs = np.concatenate(coefficients)
            flat_vars = self._get_vars_by_names(variables)
            starts = np.cumsum([0] + [len(c) for c in coefficients][:-1])

        # In HiGHS, the number of non-zero coefficients is simply the number of coefficients
        # num_cons, lower, upper, num_new_nz, starts, indices, values
        self._solver.addRows(
            len(lower_bounds),
            lower_bounds,
            upper_bounds,
            len(flat_coeffs),
            starts,
            flat_vars,
            flat_coeffs.astype(np.float64),
        )

    def add_constraint(
        self,
        coefficients: npt.NDArray[np.floating | np.integer | np.bool_] | float | bool,
        variables: npt.NDArray[str] | str,
        lower_bound: float | bool | None = None,
        upper_bound: float | bool | None = None,
        name: str | None = None,
    ) -> str | None:
        coefficients = self._to_float(value=coefficients)
        lower_bound = self._to_float(value=lower_bound) if lower_bound is not None else -self.infinity()
        upper_bound = self._to_float(value=upper_bound) if upper_bound is not None else self.infinity()

        constr_number = self._constr_count
        self._constr_count += 1

        if isinstance(coefficients, np.ndarray) & isinstance(variables, np.ndarray):
            if coefficients.size != variables.size:
                raise ValueError("The number of coefficients must be equal to the number of variables")
            coefficients = np.array(coefficients)
            variables = np.array(self._get_vars_by_names(variables))

        elif isinstance(coefficients, (float, int, bool)) and isinstance(variables, str):
            coefficients = np.array([coefficients])
            variables = np.array([self._get_var_by_name(variables)])
        else:
            raise ValueError("Coeffs and vars_ must be of the same type")

        # In HiGHS, the number of non-zero coefficients is simply the number of coefficients
        number_of_non_zero_coefficients = len(coefficients)

        # In some very specific cases, (completely) wrong coefficients are added to the model if not casted to float64
        # lower, upper, num_new_nz, index, value
        self._solver.addRow(
            lower_bound,
            upper_bound,
            number_of_non_zero_coefficients,
            variables,
            coefficients.astype(np.float64),
        )

        name = name if name is not None else str(constr_number)
        self._solver.passRowName(constr_number, name)

        return name

    def add_variable(
        self,
        name: str,
        variable_domain: VariableDomain,
        lower_bound: float | bool | None = None,
        upper_bound: float | bool | None = None,
    ) -> str:
        lower_bound = self._to_float(value=lower_bound) if lower_bound is not None else -self.infinity()
        upper_bound = self._to_float(value=upper_bound) if upper_bound is not None else self.infinity()

        var_number = self._var_count
        self._var_count += 1

        self._solver.addVar(lower_bound, upper_bound)
        self.number_of_variables_of_type[variable_domain] += 1

        if variable_domain == VariableDomain.INTEGER:
            self._solver.changeColIntegrality(var_number, highspy.HighsVarType.kInteger)
            self._integer_problem = True
        elif variable_domain == VariableDomain.CONTINUOUS:
            self._solver.changeColIntegrality(var_number, highspy.HighsVarType.kContinuous)
        elif variable_domain == VariableDomain.BINARY:
            self._solver.changeColIntegrality(var_number, highspy.HighsVarType.kInteger)
            self._solver.changeColBounds(var_number, 0, 1)
            self._integer_problem = True
        else:
            raise ValueError("Unsupported variable data type")

        self._solver.passColName(var_number, name)

        return name

    def get_variable_value(self, var: str) -> float:
        return self._solution[self._solver.getColByName(var)[1]]

    def add_objective_term(
        self,
        coefficient: float | bool,
        variable: str,
        overwrite: bool = True,
        name: str | None = None,
    ) -> None:
        coefficient = self._to_float(value=coefficient)
        _, old_coeff, _, _, _ = self._solver.getCol(self._get_var_by_name(variable))  # status, cost, lb, ub, index
        if name is not None:
            self.add_named_objective(
                np.array([coefficient]),
                np.array([self._get_var_by_name(variable)]),
                overwrite,
                name,
            )

        if old_coeff == 0 and coefficient != 0:
            self._obj_count += 1
        elif old_coeff != 0 and coefficient == 0:
            self._obj_count -= 1

        if overwrite:
            self._solver.changeColCost(self._get_var_by_name(variable), coefficient)
        else:
            self._solver.changeColCost(self._get_var_by_name(variable), coefficient + old_coeff)

    def get_named_objective(self, name: str) -> float:
        if name in self._named_objectives:
            return float(
                np.sum(
                    [
                        (item.objective_coefficient * self._solution[item.variable])
                        if item.variable + 1 <= len(self._solution)
                        else 0.0
                        for item in self._named_objectives[name]
                    ]
                )
            )
        return 0.0

    def set_optimization_direction(self, maximization: bool) -> None:
        if maximization:
            self._solver.changeObjectiveSense(highspy.ObjSense.kMaximize)
        else:
            self._solver.changeObjectiveSense(highspy.ObjSense.kMinimize)

    def get_objective_value(self) -> float:
        info: highspy.HighsInfo = self._solver.getInfo()
        return info.objective_function_value

    def solve(self, time_limit: float | None = None, mip_gap_limit: float | None = None) -> int:
        if time_limit is not None:
            self._solver.setOptionValue("time_limit", time_limit)
        if mip_gap_limit is not None:
            self._solver.setOptionValue("mip_rel_gap", mip_gap_limit)
        self._solver.run()

        self.status = self._solver.getModelStatus()

        # This list conversion is necessary because of performance issues in the C implementation
        self._solution = list(self._solver.getSolution().col_value)
        self._dual_values = list(self._solver.getSolution().row_dual)

        return self.status

    def get_constraint(self, name: str) -> int:
        _, constr = self._solver.getRowByName(name)
        return constr

    def infinity(self) -> float:
        return highspy.kHighsInf

    def is_optimal(self) -> bool:
        return self.status in [
            highspy.HighsModelStatus.kOptimal,
            highspy.HighsModelStatus.kModelEmpty,
        ]

    def is_feasible(self) -> bool:
        return not self.is_infeasible() and not self.is_abnormal() and not self.is_unbounded()

    def is_infeasible(self) -> bool:
        return self.status == highspy.HighsModelStatus.kInfeasible

    def is_unbounded(self) -> bool:
        return self.status in [
            highspy.HighsModelStatus.kUnbounded,
            highspy.HighsModelStatus.kUnboundedOrInfeasible,
        ]

    def is_abnormal(self) -> bool:
        return self.status in [
            highspy.HighsModelStatus.kLoadError,
            highspy.HighsModelStatus.kModelError,
            highspy.HighsModelStatus.kPresolveError,
            highspy.HighsModelStatus.kSolveError,
            highspy.HighsModelStatus.kPostsolveError,
            highspy.HighsModelStatus.kUnknown,
        ]

    def is_not_solved(self) -> bool:
        return self.status in [
            highspy.HighsModelStatus.kTimeLimit,
            highspy.HighsModelStatus.kIterationLimit,
            highspy.HighsModelStatus.kSolutionLimit,
        ]

    def set_solver_setting(self, setting: HighsSolverConfig) -> None:
        for option_name, option_value in setting.to_highs_options().items():
            self._solver.setOptionValue(option_name, option_value)

    def export_to_file(self, path: str) -> None:
        self._solver.writeModel(path)

    def set_verbose(self, verbose: bool) -> None:
        self._solver.setOptionValue("log_to_console", verbose)
        self._solver.setOptionValue("output_flag", verbose)

    def get_constraint_count(self):
        return self._solver.getNumRow()

    def get_variable_count(self):
        return self._solver.getNumCol()

    def get_variable_count_of_type(self, variable_domain: VariableDomain):
        return self.number_of_variables_of_type[variable_domain]

    def get_objective_terms_count(self):
        return self._obj_count

    def force_update(self):
        pass  # HiGHs is always eager

    def get_gap(self) -> float:
        info: highspy.HighsInfo = self._solver.getInfo()
        bound = info.mip_dual_bound
        objective_value = info.objective_function_value
        if objective_value == 0 and self.is_not_solved():
            return self.infinity()
        if objective_value == 0 and bound == 0:
            return 0
        return abs(bound - objective_value) / abs(objective_value)

    def add_objective_offset(self, offset: float | bool, overwrite: bool = True):
        offset = self._to_float(value=offset)
        if overwrite:
            self._solver.changeObjectiveOffset(offset)
        else:
            _, old_offset = self._solver.getObjectiveOffset()
            self._solver.changeObjectiveOffset(old_offset + offset)

    def get_dual_value(self, constraint: str) -> float:
        if self._integer_problem:
            raise ValueError("Dual values are not available for integer problems")
        if not self.is_optimal():
            raise ValueError("Dual values are only available for optimal solutions")
        return self._dual_values[self._get_constraint_by_name(constraint)]
