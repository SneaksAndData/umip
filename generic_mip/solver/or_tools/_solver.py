"""A solver implemented in the Google OR-Tools library."""
from typing import Optional, Union, Iterable
from ortools.linear_solver import pywraplp
import numpy.typing as npt
import numpy as np
from adapta.logs import SemanticLogger
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.solver.or_tools._solver_engine import OrToolsSolverEngine
from generic_mip.variable_data_type import VariableDataType


class OrToolsSolver(AbstractOptimizationSolver[pywraplp.Variable, pywraplp.Constraint]):  # pylint: disable=too-many-public-methods
    """A solver implemented in the Google OR-Tools library."""
    def __init__(self, solver_engine: OrToolsSolverEngine, logger: SemanticLogger):
        super().__init__(logger)
        self._solver: pywraplp.Solver = pywraplp.Solver.CreateSolver(solver_engine.value)
        self._solver.EnableOutput()
        self._objective: pywraplp.Objective = self._solver.Objective()
        self.status = None

    def set_variable_hint(self, var: pywraplp.Variable, hint: float) -> None:
        self._solver.SetHint([var], [hint])

    def set_multiple_variable_hints(self, vars_: npt.NDArray[pywraplp.Variable], hints: npt.NDArray[float]) -> None:
        self._solver.SetHint(vars_, hints)

    def add_multiple_variables(self, count: int, name: str, dtype: VariableDataType, lb: Optional[float] = None, ub: Optional[float] = None) -> npt.NDArray[pywraplp.Variable]:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        return np.array([
            self.add_variable(
                lb=lb,
                ub=ub,
                name=f'{name}{i}',
                dtype=dtype
            )
            for i in range(count)
        ])

    def add_multiple_constraints(
        self,
        coeffs: Union[npt.NDArray[npt.NDArray[float]], npt.NDArray[float]],
        vars_: Union[npt.NDArray[npt.NDArray[pywraplp.Variable]], npt.NDArray[pywraplp.Variable]],
        lb: Optional[npt.NDArray[float]] = None,
        ub: Optional[npt.NDArray[float]] = None,
        name: Optional[str] = None
    ) -> None:
        if coeffs.size == 0:
            return

        num_constrs = len(coeffs)
        for i in range(num_constrs):
            constr: pywraplp.Constraint = self._solver.Constraint(
                lb[i] if lb is not None else -self._solver.infinity(),
                ub[i] if ub is not None else self._solver.infinity()
            )
            if coeffs.ndim == 1 and not isinstance(coeffs[0], (np.ndarray, list, set)):
                constr.SetCoefficient(vars_[i], coeffs[i])
            else:
                for j in range(len(coeffs[i])):
                    constr.SetCoefficient(vars_[i][j], coeffs[i][j])

    def add_constraint(
        self,
        coeffs: Union[npt.NDArray[float], float],
        vars_: Union[npt.NDArray[pywraplp.Variable], pywraplp.Variable],
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        name: Optional[str] = None
    ) -> Optional[pywraplp.Constraint]:
        if lb is None and ub is None:
            return None

        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        constr: pywraplp.Constraint = self._solver.Constraint(lb, ub, name) if name is not None else self._solver.Constraint(lb, ub)

        if isinstance(vars_, Iterable):
            for (coeff, var) in zip(coeffs, vars_):
                constr.SetCoefficient(var, coeff)
        else:
            constr.SetCoefficient(vars_, coeffs)

        return constr

    def add_variable(self, name: str, dtype: VariableDataType, lb: Optional[float] = None, ub: Optional[float] = None) -> pywraplp.Variable:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        if dtype == VariableDataType.INT:
            return self._solver.IntVar(lb, ub, name)
        if dtype == VariableDataType.FLOAT:
            return self._solver.NumVar(lb, ub, name)
        if dtype == VariableDataType.BOOL:
            return self._solver.BoolVar(name)
        raise ValueError("Unsupported variable data type")

    def get_variable_value(self, var: pywraplp.Variable) -> float:
        return var.SolutionValue()

    def add_objective_term(self, coeff: float, var: pywraplp.Variable) -> None:
        self._objective.SetCoefficient(var, coeff)

    def add_multiple_objective_terms(self, coeffs: npt.NDArray[float], vars_: npt.NDArray[pywraplp.Variable]) -> None:
        for i in range(len(coeffs)):  # pylint: disable=consider-using-enumerate
            self.add_objective_term(coeffs[i], vars_[i])

    def set_optimization_direction(self, maximization: bool) -> None:
        self._objective.SetOptimizationDirection(maximization)

    def get_objective_value(self) -> float:
        return self._objective.Value()

    def solve(self) -> int:
        self.status = self._solver.Solve()
        return self.status

    def get_constraint(self, name: str) -> pywraplp.Constraint:
        return self._solver.LookupConstraint(name)

    def infinity(self) -> float:
        return self._solver.infinity()

    def is_optimal(self) -> bool:
        return self.status == pywraplp.Solver.OPTIMAL

    def is_infeasible(self) -> bool:
        return self.status == pywraplp.Solver.INFEASIBLE

    def is_unbounded(self) -> bool:
        return self.status == pywraplp.Solver.UNBOUNDED

    def is_abnormal(self) -> bool:
        return self.status == pywraplp.Solver.ABNORMAL

    def set_solver_setting(self, setting: str) -> None:
        self._solver.SetSolverSpecificParametersAsString(setting)

    def export_to_file(self, path: str) -> None:
        if path.lower().endswith('.lp'):
            file_content = self._solver.ExportModelAsLpFormat(False)
        else:
            file_content = self._solver.ExportModelAsMpsFormat(False, False)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(file_content)

    def set_verbose(self, verbose: bool) -> None:
        if verbose:
            self._solver.EnableOutput()
        else:
            self._solver.SuppressOutput()

    def get_variable_count(self):
        return self._solver.NumVariables()

    def get_constraint_count(self):
        return self._solver.NumConstraints()

    def get_objective_terms_count(self):
        return len([coeff for var in self._solver.variables() if (coeff := self._objective.GetCoefficient(var)) != 0])

    def force_update(self):
        # OR Tools uses eager updates.
        pass
