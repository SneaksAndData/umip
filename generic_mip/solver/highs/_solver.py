"""A solver implemented in the HiGHS library."""
import highspy
import numpy.typing as npt
import numpy as np
from adapta.logs import LoggerInterface
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.enums.variable_data_type import VariableDataType


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
        self.number_of_variables_of_type = dict(zip(list(VariableDataType), np.zeros(len(VariableDataType), dtype=int)))
        if model_path is not None:
            self._solver.readModel(model_path)
        self.status: highspy.HighsModelStatus | None = None
        self._solution: list[float] | None = None

    def set_variable_hint(self, var: str, hint: float) -> None:
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
        if names.ndim > 1 or isinstance(names[0], (np.ndarray, list, set)):
            names = np.concatenate(names)
        return np.array([self._solver.getColByName(name)[1] for name in names])

    def _get_constraint_by_name(self, name: str) -> int:
        return self._solver.getRowByName(name)[1]

    def set_multiple_variable_hints(self, vars_: npt.NDArray[str], hints: npt.NDArray[float]) -> None:
        raise NotImplementedError()

    def add_multiple_objective_terms(
        self, coeffs: npt.NDArray[float], vars_: npt.NDArray[str], overwrite: bool = True, name: str = None
    ) -> None:
        if name is not None:
            self.add_named_objective(coeffs, self._get_vars_by_names(vars_), overwrite, name)

        for i in range(len(coeffs)):  # pylint: disable=consider-using-enumerate
            self.add_objective_term(coeffs[i], vars_[i], overwrite=overwrite)

    def add_multiple_variables(
        self,
        names: npt.NDArray[str],
        dtype: VariableDataType,
        lb: float | None = None,
        ub: float | None = None,
    ) -> npt.NDArray[str]:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        first_var_number = self._var_count
        self._var_count += len(names)

        self._solver.addVars(len(names), lb, ub)

        for i, name in enumerate(names):
            self.number_of_variables_of_type[dtype] += 1
            if dtype == VariableDataType.INT:
                self._integer_problem = True
                self._solver.changeColIntegrality(first_var_number + i, highspy.HighsVarType.kInteger)
            elif dtype == VariableDataType.FLOAT:
                self._solver.changeColIntegrality(first_var_number + i, highspy.HighsVarType.kContinuous)
            elif dtype == VariableDataType.BOOL:
                self._integer_problem = True
                self._solver.changeColIntegrality(first_var_number + i, highspy.HighsVarType.kInteger)
                self._solver.changeColBounds(first_var_number + i, 0, 1)
            else:
                raise ValueError("Unsupported variable data type")
            self._solver.passColName(first_var_number + i, name)

        return names

    def add_multiple_constraints(
        self,
        coeffs: npt.NDArray[npt.NDArray[float]] | npt.NDArray[float],
        vars_: npt.NDArray[npt.NDArray[str]] | npt.NDArray[str],
        lb: npt.NDArray[float] | None = None,
        ub: npt.NDArray[float] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        if lb is None and ub is None:
            raise ValueError("Either lb or ub must be specified")

        if lb is None:
            lb = np.full(len(ub), -self.infinity())
        if ub is None:
            ub = np.full(len(lb), self.infinity())

        if len(lb) == 0:
            return

        self._constr_count += len(lb)

        if coeffs.ndim == 1 and not isinstance(coeffs[0], (np.ndarray, list, set)):
            flat_coeffs = coeffs
            flat_vars = self._get_vars_by_names(vars_)
            starts = np.arange(0, len(flat_coeffs) + 1)
        else:
            flat_coeffs = np.concatenate(coeffs)
            flat_vars = self._get_vars_by_names(vars_)
            starts = np.cumsum([0] + [len(c) for c in coeffs][:-1])

        # In HiGHS, the number of non-zero coefficients is simply the number of coefficients
        # num_cons, lower, upper, num_new_nz, starts, indices, values
        self._solver.addRows(
            len(lb),
            lb,
            ub,
            len(flat_coeffs),
            starts,
            flat_vars,
            flat_coeffs.astype(np.float64),
        )

    def add_constraint(
        self,
        coeffs: npt.NDArray[float] | float,
        vars_: npt.NDArray[str] | str,
        lb: float | None = None,
        ub: float | None = None,
        name: str | None = None,
    ) -> str | None:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        constr_number = self._constr_count
        self._constr_count += 1

        if isinstance(coeffs, np.ndarray) & isinstance(vars_, np.ndarray):
            if coeffs.size != vars_.size:
                raise ValueError("The number of coefficients must be equal to the number of variables")
            coeffs = np.array(coeffs)
            vars_ = np.array(self._get_vars_by_names(vars_))

        elif isinstance(coeffs, (float, int, bool)) and isinstance(vars_, str):
            coeffs = np.array([coeffs])
            vars_ = np.array([self._get_var_by_name(vars_)])
        else:
            raise ValueError("Coeffs and vars_ must be of the same type")

        # In HiGHS, the number of non-zero coefficients is simply the number of coefficients
        number_of_non_zero_coefficients = len(coeffs)

        # In some very specific cases, (completely) wrong coefficients are added to the model if not casted to float64
        # lower, upper, num_new_nz, index, value
        self._solver.addRow(lb, ub, number_of_non_zero_coefficients, vars_, coeffs.astype(np.float64))

        name = name if name is not None else str(constr_number)
        self._solver.passRowName(constr_number, name)

        return name

    def add_variable(self, name: str, dtype: VariableDataType, lb: float | None = None, ub: float | None = None) -> str:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        var_number = self._var_count
        self._var_count += 1

        self._solver.addVar(lb, ub)
        self.number_of_variables_of_type[dtype] += 1

        if dtype == VariableDataType.INT:
            self._solver.changeColIntegrality(var_number, highspy.HighsVarType.kInteger)
            self._integer_problem = True
        elif dtype == VariableDataType.FLOAT:
            self._solver.changeColIntegrality(var_number, highspy.HighsVarType.kContinuous)
        elif dtype == VariableDataType.BOOL:
            self._solver.changeColIntegrality(var_number, highspy.HighsVarType.kInteger)
            self._solver.changeColBounds(var_number, 0, 1)
            self._integer_problem = True
        else:
            raise ValueError("Unsupported variable data type")

        self._solver.passColName(var_number, name)

        return name

    def get_variable_value(self, var: str) -> float:
        return self._solution[self._solver.getColByName(var)[1]]

    def add_objective_term(self, coeff: float, var: str, overwrite: bool = True, name: str = None) -> None:
        _, old_coeff, _, _, _ = self._solver.getCol(self._get_var_by_name(var))  # status, cost, lb, ub, index
        if name is not None:
            self.add_named_objective(np.array([coeff]), np.array([self._get_var_by_name(var)]), overwrite, name)

        if old_coeff == 0 and coeff != 0:
            self._obj_count += 1
        elif old_coeff != 0 and coeff == 0:
            self._obj_count -= 1

        if overwrite:
            self._solver.changeColCost(self._get_var_by_name(var), coeff)
        else:
            self._solver.changeColCost(self._get_var_by_name(var), coeff + old_coeff)

    def get_named_objective(self, name: str) -> float:
        return np.sum(
            [(item.objective_coefficient * self._solution[item.variable]) for item in self._named_objectives[name]]
        )

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
        return self.status in [highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kModelEmpty]

    def is_feasible(self) -> bool:
        return not self.is_infeasible() and not self.is_abnormal() and not self.is_unbounded()

    def is_infeasible(self) -> bool:
        return self.status == highspy.HighsModelStatus.kInfeasible

    def is_unbounded(self) -> bool:
        return self.status in [highspy.HighsModelStatus.kUnbounded, highspy.HighsModelStatus.kUnboundedOrInfeasible]

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

    def set_solver_setting(self, setting: str) -> None:
        raise ValueError("Not supported in HiGHS solver")

    def export_to_file(self, path: str) -> None:
        self._solver.writeModel(path)

    def set_verbose(self, verbose: bool) -> None:
        self._solver.setOptionValue("log_to_console", verbose)
        self._solver.setOptionValue("output_flag", verbose)

    def get_constraint_count(self):
        return self._solver.getNumRow()

    def get_variable_count(self):
        return self._solver.getNumCol()

    def get_variable_count_of_type(self, var_type: VariableDataType):
        return self.number_of_variables_of_type[var_type]

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

    def add_objective_offset(self, offset: float, overwrite: bool = True):
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
