from typing import Iterable

import numpy as np
import numpy.typing as npt
import pyscipopt
from pyscipopt import Model
from adapta.logs import LoggerInterface
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.enums.variable_data_type import VariableDataType


class ScipSolver(
    AbstractOptimizationSolver[pyscipopt.Variable, pyscipopt.Constraint]
):  # pylint: disable=too-many-public-methods
    """A solver implemented in the SCIP library."""

    def __init__(self, logger: LoggerInterface, model_path: str | None = None):
        super().__init__(logger)
        self._solver = Model()
        if model_path is not None:
            self._solver = self._solver.readProblem(model_path)
        self.number_of_variables_of_type = {variable_type: 0 for variable_type in list(VariableDataType)}
        self._objective = 0
        self._objective_term_count = 0

    def add_constraint(
        self,
        coeffs: npt.NDArray[float] | float,
        vars_: npt.NDArray[pyscipopt.Variable] | pyscipopt.Variable,
        lb: float | None = None,
        ub: float | None = None,
        name: str | None = None,
    ) -> pyscipopt.Constraint | None:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        name = name if name is not None else ""

        if isinstance(coeffs, Iterable):
            constr_expr = 0
            for coeff, var in zip(coeffs, vars_):
                constr_expr += coeff * var
        else:
            constr_expr = coeffs * vars_

        if lb == ub:
            return self._solver.addCons(cons=(constr_expr == lb), name=name)
        if lb != -self.infinity() and ub != self.infinity():
            return self._solver.addCons(cons=lb <= (constr_expr <= ub), name=name)
        if lb != -self.infinity():
            return self._solver.addCons(cons=(constr_expr >= lb), name=name)

        return self._solver.addCons(cons=(constr_expr <= ub), name=name)

    def add_multiple_constraints(
        self,
        coeffs: npt.NDArray[npt.NDArray[float]] | npt.NDArray[float],
        vars_: npt.NDArray[npt.NDArray[pyscipopt.Variable]] | npt.NDArray[pyscipopt.Variable],
        lb: npt.NDArray[float] | None = None,
        ub: npt.NDArray[float] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        if coeffs.size == 0:
            return

        for i, coeff in enumerate(coeffs):
            lowerbound = lb[i] if lb is not None else None
            upperbound = ub[i] if ub is not None else None
            name = names[i] if names is not None else None
            self.add_constraint(coeff, vars_[i], lb=lowerbound, ub=upperbound, name=name)

    def get_constraint(self, name: str) -> pyscipopt.Constraint:
        for constraint in self._solver.getConss():
            if constraint.name == name:
                return constraint

        raise ValueError(f"Constraint with name {name} not found")

    def add_variable(
        self, name: str, dtype: VariableDataType, lb: float | None = None, ub: float | None = None
    ) -> pyscipopt.Variable:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        self.number_of_variables_of_type[dtype] += 1

        if dtype == VariableDataType.INT:
            self._integer_problem = True
            variable_type = "I"
        elif dtype == VariableDataType.BOOL:
            self._integer_problem = True
            variable_type = "B"
        elif dtype == VariableDataType.FLOAT:
            variable_type = "C"
        else:
            raise ValueError("Unsupported variable data type")

        return self._solver.addVar(name=name, lb=lb, ub=ub, vtype=variable_type)

    def add_multiple_variables(
        self,
        names: npt.NDArray[str],
        dtype: VariableDataType,
        lb: float | None = None,
        ub: float | None = None,
    ) -> npt.NDArray[pyscipopt.Variable]:
        return np.array([self.add_variable(lb=lb, ub=ub, name=f"{name}", dtype=dtype) for name in names])

    def set_variable_hint(self, var: pyscipopt.Variable, hint: float) -> None:
        partial_solution = self._solver.createPartialSol()
        self._solver.setSolVal(partial_solution, var, hint)

    def set_multiple_variable_hints(self, vars_: npt.NDArray[pyscipopt.Variable], hints: npt.NDArray[float]) -> None:
        partial_solution = self._solver.createPartialSol()

        for i in range(vars_.size):
            self._solver.setSolVal(partial_solution, vars_[i], hints[i])

    def add_objective_term(
        self, coeff: float, var: pyscipopt.Variable, overwrite: bool = True, name: str = None
    ) -> None:
        if name is not None:
            self.add_named_objective(np.array([coeff]), np.array([var]), overwrite, name)

        self._objective_term_count += 1 if coeff != 0 else 0

        if overwrite:
            raise ValueError("ScipSolver.add_objective_term() is not supported with overwrite = true ")

        self._objective += coeff * var

    def add_multiple_objective_terms(
        self,
        coeffs: npt.NDArray[float],
        vars_: npt.NDArray[pyscipopt.Variable],
        overwrite: bool = True,
        name: str = None,
    ) -> None:
        if name is not None:
            self.add_named_objective(coeffs, vars_, overwrite, name)

        for coeff, var in zip(coeffs, vars_):
            self._objective_term_count += 1 if coeff != 0 else 0
            if overwrite:
                raise ValueError("ScipSolver.add_objective_term() is not supported with overwrite = true ")

            self._objective += coeff * var

    def set_optimization_direction(self, maximization: bool) -> None:
        if maximization:
            self._solver.setMaximize()
        else:
            self._solver.setMinimize()

    def get_objective_value(self) -> float:
        return self._solver.getObjVal()

    def solve(self, time_limit: float | None = None, mip_gap_limit: float | None = None) -> str:
        if time_limit is not None:
            self._solver.setParam("limits/time", time_limit)
        if mip_gap_limit is not None:
            self._solver.setParam("limits/gap", mip_gap_limit)

        offset = self._solver.getObjoffset()
        self._solver.setObjective(expr=self._objective, sense=self._solver.getObjectiveSense())
        self._solver.addObjoffset(offset)
        self._solver.optimize()

        return self._solver.getStatus()

    def set_param(self, setting: str, value: int) -> None:
        """
        Set solver parameters for the SCIP solver
        :param setting: Name of parameter to set
        :param value: Value of parameter to set
        """
        self._solver.setParam(setting, value)

    def infinity(self) -> float:
        return self._solver.infinity()

    def is_optimal(self) -> bool:
        return self._solver.getStatus().lower() == "optimal"

    def is_feasible(self) -> bool:
        return len(self._solver.getSols()) > 0

    def is_infeasible(self) -> bool:
        return self._solver.getStatus().lower() == "infeasible"

    def is_abnormal(self) -> bool:
        return self._solver.getStatus() in [
            "unknown",
            "userinterrupt",
        ]

    def is_unbounded(self) -> bool:
        return self._solver.getStatus().lower() == "unbounded"

    def is_not_solved(self) -> bool:
        return self._solver.getStatus().lower() in [
            "nodelimit",
            "totalnodelimit",
            "stallnodelimit",
            "timelimit",
            "memlimit",
            "gaplimit",
            "primallimit",
            "duallimit",
            "sollimit",
            "bestsollimit",
            "restartlimit",
        ]

    def get_variable_value(self, var: pyscipopt.Variable) -> float:
        return self._solver.getVal(var)

    def export_to_file(self, path: str) -> None:
        self._solver.writeProblem(filename=path)

    def set_verbose(self, verbose: bool) -> None:
        quiet = not verbose
        self._solver.hideOutput(quiet)

    def get_variable_count(self) -> int:
        return self._solver.getNVars()

    def get_variable_count_of_type(self, var_type: VariableDataType) -> int:
        return self.number_of_variables_of_type[var_type]

    def get_constraint_count(self) -> int:
        return len(self._solver.getConss())

    def get_objective_terms_count(self) -> int:
        return self._objective_term_count

    def force_update(self) -> int:
        pass

    def get_gap(self) -> float:
        return self._solver.getGap()

    def add_objective_offset(self, offset: float, overwrite: bool = True) -> None:
        if overwrite:
            offset -= self._solver.getObjoffset()

        self._solver.addObjoffset(offset)

    def get_dual_value(self, constraint: pyscipopt.Constraint) -> float:
        raise ValueError("Cannot get dual variables from the SCIP solver")
