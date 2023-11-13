"""A solver implemented in the HiGHS library."""
from typing import Optional, Union, List
import highspy
import numpy.typing as npt
import numpy as np
from adapta.logs import SemanticLogger
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.variable_data_type import VariableDataType


class HighsSolver(AbstractOptimizationSolver[int, int]):  # pylint: disable=too-many-public-methods
    """A solver implemented in the HiGHS library."""

    def __init__(self, logger: SemanticLogger, model_path: Optional[str] = None):
        """
        Initialize the solver.

        :param model_path: The path to the model file to read (mps or lp).
        :param logger: The logger to use.
        """
        super().__init__(logger)
        self._dual_values: Optional[List[float]] = None
        self._var_count = 0
        self._constr_count = 0
        self._obj_count = 0
        self._solver = highspy.Highs()
        if model_path is not None:
            self._solver.readModel(model_path)
        self.status: Optional[highspy.HighsModelStatus] = None
        self._solution: Optional[List[float]] = None

    def set_variable_hint(self, var: int, hint: float) -> None:
        raise NotImplementedError()

    def set_multiple_variable_hints(self, vars_: npt.NDArray[int], hints: npt.NDArray[float]) -> None:
        raise NotImplementedError()

    def add_multiple_objective_terms(
        self, coeffs: npt.NDArray[float], vars_: npt.NDArray[int], overwrite: bool = True
    ) -> None:
        for i in range(len(coeffs)):  # pylint: disable=consider-using-enumerate
            self.add_objective_term(coeffs[i], vars_[i], overwrite=overwrite)

    def add_multiple_variables(
        self, count: int, name: str, dtype: VariableDataType, lb: Optional[float] = None, ub: Optional[float] = None
    ) -> npt.NDArray[int]:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        first_var_number = self._var_count
        self._var_count += count

        self._solver.addVars(count, lb, ub)

        if dtype == VariableDataType.INT:
            self._integer_problem = True
            for i in range(count):
                self._solver.changeColIntegrality(first_var_number + i, highspy.HighsVarType.kInteger)
        elif dtype == VariableDataType.FLOAT:
            for i in range(count):
                self._solver.changeColIntegrality(first_var_number + i, highspy.HighsVarType.kContinuous)
        elif dtype == VariableDataType.BOOL:
            self._integer_problem = True
            for i in range(count):
                self._solver.changeColIntegrality(first_var_number + i, highspy.HighsVarType.kInteger)
                self._solver.changeColBounds(first_var_number + i, 0, 1)
        else:
            raise ValueError("Unsupported variable data type")

        return np.arange(first_var_number, first_var_number + count)

    def add_multiple_constraints(
        self,
        coeffs: Union[npt.NDArray[npt.NDArray[float]], npt.NDArray[float]],
        vars_: Union[npt.NDArray[npt.NDArray[int]], npt.NDArray[int]],
        lb: Optional[npt.NDArray[float]] = None,
        ub: Optional[npt.NDArray[float]] = None,
        name: Optional[str] = None,
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
            flat_vars = vars_
            starts = np.arange(0, len(flat_coeffs) + 1)
        else:
            flat_coeffs = np.concatenate(coeffs)
            flat_vars = np.concatenate(vars_)
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
        coeffs: Union[npt.NDArray[float], float],
        vars_: Union[npt.NDArray[int], int],
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Optional[int]:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        constr_number = self._constr_count
        self._constr_count += 1

        if isinstance(coeffs, (float, int, bool)):
            coeffs = np.array([coeffs])
            vars_ = np.array([vars_])

        # In HiGHS, the number of non-zero coefficients is simply the number of coefficients
        number_of_non_zero_coefficients = len(coeffs)

        # In some very specific cases, (completely) wrong coefficients are added to the model if not casted to float64
        # lower, upper, num_new_nz, index, value
        self._solver.addRow(lb, ub, number_of_non_zero_coefficients, vars_, coeffs.astype(np.float64))

        if name is not None:
            self._solver.passRowName(constr_number, name)

        return constr_number

    def add_variable(
        self, name: str, dtype: VariableDataType, lb: Optional[float] = None, ub: Optional[float] = None
    ) -> int:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        var_number = self._var_count
        self._var_count += 1

        self._solver.addVar(lb, ub)

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

        if name is not None:
            self._solver.passColName(var_number, name)

        return var_number

    def get_variable_value(self, var: int) -> float:
        return self._solution[var]

    def add_objective_term(self, coeff: float, var: int, overwrite: bool = True) -> None:
        _, old_coeff, _, _, _ = self._solver.getCol(var)  # status, cost, lb, ub, index

        if old_coeff == 0 and coeff != 0:
            self._obj_count += 1
        elif old_coeff != 0 and coeff == 0:
            self._obj_count -= 1

        if overwrite:
            self._solver.changeColCost(var, coeff)
        else:
            self._solver.changeColCost(var, coeff + old_coeff)

    def set_optimization_direction(self, maximization: bool) -> None:
        if maximization:
            self._solver.changeObjectiveSense(highspy.ObjSense.kMaximize)
        else:
            self._solver.changeObjectiveSense(highspy.ObjSense.kMinimize)

    def get_objective_value(self) -> float:
        info: highspy.HighsInfo = self._solver.getInfo()
        return info.objective_function_value

    def solve(self, time_limit: Optional[float] = None) -> int:
        if time_limit is not None:
            self._solver.setOptionValue("time_limit", time_limit)

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
        return self.status == highspy.HighsModelStatus.kFeasible

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

    def get_dual_value(self, constraint: int) -> float:
        if self._integer_problem:
            raise ValueError("Dual values are not available for integer problems")
        if not self.is_optimal():
            raise ValueError("Dual values are only available for optimal solutions")
        return self._dual_values[constraint]
