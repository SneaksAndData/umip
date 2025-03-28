import math
import numpy.typing as npt
import numpy as np
from adapta.logs import LoggerInterface
import localsolver as ls
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.enums.variable_data_type import VariableDataType


class LocalSolver(
    AbstractOptimizationSolver[ls.LSExpression, ls.LSExpression]
):  # pylint: disable=too-many-public-methods
    """A solver implemented in the LocalSolver library."""

    def __init__(self, logger: LoggerInterface):
        super().__init__(logger)
        self._solver = ls.LocalSolver()
        self._model = self._solver.get_model()
        self._objective = self._model.create_constant(0)
        self._maximization = True
        self.number_of_variables = 0
        self.number_of_variables_of_type = dict(zip(list(VariableDataType), np.zeros(len(VariableDataType), dtype=int)))
        self.number_of_objective_terms = 0
        self._solution = None

    def __del__(self):
        self._solver.delete()

    def add_constraint(
        self,
        coeffs: npt.NDArray[float] | float,
        vars_: npt.NDArray[ls.LSExpression] | ls.LSExpression,
        lb: float | None = None,
        ub: float | None = None,
        name: str | None = None,
    ) -> ls.LSExpression | None:
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
        coeffs: npt.NDArray[npt.NDArray[float]] | npt.NDArray[float],
        vars_: npt.NDArray[npt.NDArray[ls.LSExpression]] | npt.NDArray[ls.LSExpression],
        lb: npt.NDArray[float] | None = None,
        ub: npt.NDArray[float] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        if coeffs.size == 0:
            return

        if names is not None and len(names) != len(coeffs):
            raise ValueError("The number of names must match the number of constraints")

        num_constrs = len(coeffs)
        for i in range(num_constrs):
            self.add_constraint(
                coeffs=coeffs[i],
                vars_=vars_[i],
                lb=lb[i] if lb is not None else None,
                ub=ub[i] if ub is not None else None,
                name=f"{names[i]}" if names is not None else None,
            )

    def add_variable(
        self, name: str, dtype: VariableDataType, lb: float | None = None, ub: float | None = None
    ) -> ls.LSExpression:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        self.number_of_variables += 1
        self.number_of_variables_of_type[dtype] += 1
        if dtype == VariableDataType.INT:
            var = self._model.int(math.ceil(lb), math.floor(ub))
            self._integer_problem = True
        elif dtype == VariableDataType.BOOL:
            var = self._model.bool()
            self._integer_problem = True
        elif dtype == VariableDataType.FLOAT:
            var = self._model.float(lb, ub)
        else:
            raise ValueError(f"Unknown variable data type: {dtype}")

        if name is not None:
            var.set_name(name)

        return var

    def add_multiple_variables(
        self, names: npt.NDArray[str], dtype: VariableDataType, lb: float | None = None, ub: float | None = None
    ) -> npt.NDArray[ls.LSExpression]:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        return np.array([self.add_variable(lb=lb, ub=ub, name=f"{name}", dtype=dtype) for name in names])

    def set_variable_hint(self, var: ls.LSExpression, hint: float) -> None:
        raise NotImplementedError()

    def set_multiple_variable_hints(self, vars_: npt.NDArray[ls.LSExpression], hints: npt.NDArray[float]) -> None:
        raise NotImplementedError()

    def add_objective_term(self, coeff: float, var: ls.LSExpression, overwrite: bool = True, name: str = None) -> None:
        if name is not None:
            self.add_named_objective(np.array([coeff]), np.array([var]), overwrite, name)

        if overwrite:
            self._objective += coeff * var
            self.number_of_objective_terms += 1
        else:
            raise NotImplementedError()

    def add_multiple_objective_terms(
        self, coeffs: npt.NDArray[float], vars_: npt.NDArray[ls.LSExpression], overwrite: bool = True, name: str = None
    ) -> None:
        if name is not None:
            self.add_named_objective(coeffs, vars_, overwrite, name)

        if overwrite:
            self._objective = self._model.sum(coeffs * vars_) + self._objective
            self.number_of_objective_terms += len(coeffs)
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

    def get_variable_count_of_type(self, var_type: VariableDataType):
        return self.number_of_variables_of_type[var_type]

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
