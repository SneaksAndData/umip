import math
from typing import Optional, Union
import numpy.typing as npt
import numpy as np
from adapta.logs import SemanticLogger
import localsolver as ls
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.variable_data_type import VariableDataType


class LocalSolver(
    AbstractOptimizationSolver[ls.LSExpression, ls.LSExpression]
):  # pylint: disable=too-many-public-methods
    """A solver implemented in the LocalSolver library."""

    def __init__(self, logger: SemanticLogger):
        super().__init__(logger)
        self._solver = ls.LocalSolver()
        self._model = self._solver.get_model()
        self._objective = self._model.create_constant(0)
        self._maximization = True
        self.number_of_variables = 0
        self.number_of_objective_terms = 0
        self._solution = None

    def __del__(self):
        self._solver.delete()

    def add_constraint(
        self,
        coeffs: Union[npt.NDArray[float], float],
        vars_: Union[npt.NDArray[ls.LSExpression], ls.LSExpression],
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Optional[ls.LSExpression]:
        if lb is None and ub is None:
            return None

        expr = self._model.sum(coeffs * vars_)

        if name is not None:
            expr.set_name(name)

        constr_lb = self._model.add_constraint(expr >= lb) if lb is not None else None
        constr_ub = self._model.add_constraint(expr <= ub) if ub is not None else None

        return constr_lb or constr_ub

    def get_constraint(self, name: str) -> ls.LSExpression:
        return self._model.get_expression(name)

    def add_multiple_constraints(
        self,
        coeffs: Union[npt.NDArray[npt.NDArray[float]], npt.NDArray[float]],
        vars_: Union[npt.NDArray[npt.NDArray[ls.LSExpression]], npt.NDArray[ls.LSExpression]],
        lb: Optional[npt.NDArray[float]] = None,
        ub: Optional[npt.NDArray[float]] = None,
        name: Optional[str] = None,
    ) -> None:
        if coeffs.size == 0:
            return

        num_constrs = len(coeffs)
        for i in range(num_constrs):
            self.add_constraint(
                coeffs=coeffs[i],
                vars_=vars_[i],
                lb=lb[i] if lb is not None else None,
                ub=ub[i] if ub is not None else None,
                name=f"{name}{i}" if name is not None else None,
            )

    def add_variable(
        self, name: str, dtype: VariableDataType, lb: Optional[float] = None, ub: Optional[float] = None
    ) -> ls.LSExpression:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        self.number_of_variables += 1
        if dtype == VariableDataType.INT:
            var = self._model.int(math.ceil(lb), math.floor(ub))
        elif dtype == VariableDataType.BOOL:
            var = self._model.bool()
        elif dtype == VariableDataType.FLOAT:
            var = self._model.float(lb, ub)
        else:
            raise ValueError(f"Unknown variable data type: {dtype}")

        if name is not None:
            var.set_name(name)

        return var

    def add_multiple_variables(
        self, count: int, name: str, dtype: VariableDataType, lb: Optional[float] = None, ub: Optional[float] = None
    ) -> npt.NDArray[ls.LSExpression]:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        return np.array([self.add_variable(lb=lb, ub=ub, name=f"{name}{i}", dtype=dtype) for i in range(count)])

    def set_variable_hint(self, var: ls.LSExpression, hint: float) -> None:
        raise NotImplementedError()

    def set_multiple_variable_hints(self, vars_: npt.NDArray[ls.LSExpression], hints: npt.NDArray[float]) -> None:
        raise NotImplementedError()

    def add_objective_term(self, coeff: float, var: ls.LSExpression, overwrite: bool = True) -> None:
        if overwrite:
            self._objective += coeff * var
            self.number_of_objective_terms += 1
        else:
            raise NotImplementedError()

    def add_multiple_objective_terms(
        self, coeffs: npt.NDArray[float], vars_: npt.NDArray[ls.LSExpression], overwrite: bool = True
    ) -> None:
        if overwrite:
            self._objective = self._model.sum(coeffs * vars_) + self._objective
            self.number_of_objective_terms += len(coeffs)
        else:
            raise NotImplementedError()

    def set_optimization_direction(self, maximization: bool) -> None:
        self._maximization = maximization

    def get_objective_value(self) -> float:
        return self._solution.get_value(self._objective)

    def solve(self, time_limit: Optional[float] = None) -> int:
        if time_limit is not None:
            self._solver.get_param().set_time_limit(time_limit)
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

    def is_infeasible(self) -> bool:
        return self._solution.get_status() in [ls.LSSolutionStatus.INFEASIBLE, ls.LSSolutionStatus.INCONSISTENT]

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

    def set_solver_setting(self, setting: str) -> None:
        raise NotImplementedError()

    def get_variable_count(self):
        return self.number_of_variables

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
