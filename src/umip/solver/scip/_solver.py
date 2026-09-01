from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pyscipopt
from adapta.logs import LoggerInterface
from pyscipopt import Model

from umip.abstract_solver import AbstractOptimizationSolver
from umip.enums.variable_domain import VariableDomain


class ScipSolver(AbstractOptimizationSolver[pyscipopt.Variable, pyscipopt.Constraint]):  # pylint: disable=too-many-public-methods
    """A solver implemented in the SCIP library."""

    def __init__(self, logger: LoggerInterface, model_path: str | None = None):
        super().__init__(logger)
        self._solver = Model()
        if model_path is not None:
            self._solver = self._solver.readProblem(model_path)
        self.number_of_variables_of_type = {variable_type: 0 for variable_type in list(VariableDomain)}
        self._objective = 0
        self._objective_term_count = 0

    def add_constraint(
        self,
        coefficients: npt.NDArray[np.floating | np.integer | np.bool_] | float | bool,
        variables: npt.NDArray[pyscipopt.Variable] | pyscipopt.Variable,
        lower_bound: float | bool | None = None,
        upper_bound: float | bool | None = None,
        name: str | None = None,
    ) -> pyscipopt.Constraint | None:
        coefficients = self._to_float(value=coefficients)
        lower_bound = self._to_float(value=lower_bound) if lower_bound is not None else -self.infinity()
        upper_bound = self._to_float(value=upper_bound) if upper_bound is not None else self.infinity()
        name = name if name is not None else ""

        if isinstance(coefficients, Iterable):
            constr_expr = 0
            for coeff, var in zip(coefficients, variables):
                constr_expr += coeff * var
        else:
            constr_expr = coefficients * variables

        if lower_bound == upper_bound:
            return self._solver.addCons(cons=(constr_expr == lower_bound), name=name)
        if lower_bound != -self.infinity() and upper_bound != self.infinity():
            return self._solver.addCons(cons=lower_bound <= (constr_expr <= upper_bound), name=name)
        if lower_bound != -self.infinity():
            return self._solver.addCons(cons=(constr_expr >= lower_bound), name=name)

        return self._solver.addCons(cons=(constr_expr <= upper_bound), name=name)

    def add_multiple_constraints(
        self,
        coefficients: npt.NDArray[npt.NDArray[np.floating | np.integer | np.bool_]]
        | npt.NDArray[np.floating | np.integer | np.bool_],
        variables: npt.NDArray[npt.NDArray[pyscipopt.Variable]] | npt.NDArray[pyscipopt.Variable],
        lower_bounds: npt.NDArray[np.floating | np.integer | np.bool_] | None = None,
        upper_bounds: npt.NDArray[np.floating | np.integer | np.bool_] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        if coefficients.size == 0:
            return

        coefficients = self._to_float(value=coefficients)

        for i, coeff in enumerate(coefficients):
            lowerbound = self._to_float(value=lower_bounds[i]) if lower_bounds is not None else None
            upperbound = self._to_float(value=upper_bounds[i]) if upper_bounds is not None else None
            name = names[i] if names is not None else None
            self.add_constraint(
                coeff,
                variables[i],
                lower_bound=lowerbound,
                upper_bound=upperbound,
                name=name,
            )

    def get_constraint(self, name: str) -> pyscipopt.Constraint:
        for constraint in self._solver.getConss():
            if constraint.name == name:
                return constraint

        raise ValueError(f"Constraint with name {name} not found")

    def add_variable(
        self,
        name: str,
        variable_domain: VariableDomain,
        lower_bound: float | bool | None = None,
        upper_bound: float | bool | None = None,
    ) -> pyscipopt.Variable:
        lower_bound = self._to_float(value=lower_bound) if lower_bound is not None else -self.infinity()
        upper_bound = self._to_float(value=upper_bound) if upper_bound is not None else self.infinity()
        self.number_of_variables_of_type[variable_domain] += 1

        if variable_domain == VariableDomain.INTEGER:
            self._integer_problem = True
            variable_type = "I"
        elif variable_domain == VariableDomain.BINARY:
            self._integer_problem = True
            variable_type = "B"
        elif variable_domain == VariableDomain.CONTINUOUS:
            variable_type = "C"
        else:
            raise ValueError("Unsupported variable data type")

        return self._solver.addVar(name=name, lb=lower_bound, ub=upper_bound, vtype=variable_type)

    def add_multiple_variables(
        self,
        names: npt.NDArray[str],
        variable_domain: VariableDomain,
        lower_bound: float | bool | None = None,
        upper_bound: float | bool | None = None,
    ) -> npt.NDArray[pyscipopt.Variable]:
        return np.array(
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

    def set_variable_hint(self, variable: pyscipopt.Variable, hint: float | bool) -> None:
        hint = self._to_float(value=hint)
        partial_solution = self._solver.createPartialSol()
        self._solver.setSolVal(partial_solution, variable, hint)

    def set_multiple_variable_hints(
        self,
        variables: npt.NDArray[pyscipopt.Variable],
        hints: npt.NDArray[np.floating | np.integer | np.bool_],
    ) -> None:
        hints = self._to_float(value=hints)
        partial_solution = self._solver.createPartialSol()

        for i in range(variables.size):
            self._solver.setSolVal(partial_solution, variables[i], hints[i])

    def add_objective_term(
        self,
        coefficient: float | bool,
        variable: pyscipopt.Variable,
        overwrite: bool = True,
        name: str | None = None,
    ) -> None:
        coefficient = self._to_float(value=coefficient)
        if name is not None:
            self.add_named_objective(np.array([coefficient]), np.array([variable]), overwrite, name)

        self._objective_term_count += 1 if coefficient != 0 else 0

        if overwrite:
            raise ValueError("ScipSolver.add_objective_term() is not supported with overwrite = true ")

        self._objective += coefficient * variable

    def add_multiple_objective_terms(
        self,
        coefficients: npt.NDArray[np.floating | np.integer | np.bool_],
        variables: npt.NDArray[pyscipopt.Variable],
        overwrite: bool = True,
        name: str | None = None,
    ) -> None:
        coefficients = self._to_float(value=coefficients)
        if name is not None:
            self.add_named_objective(coefficients, variables, overwrite, name)

        for coeff, var in zip(coefficients, variables):
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

    def get_variable_count_of_type(self, variable_domain: VariableDomain) -> int:
        return self.number_of_variables_of_type[variable_domain]

    def get_constraint_count(self) -> int:
        return len(self._solver.getConss())

    def get_objective_terms_count(self) -> int:
        return self._objective_term_count

    def force_update(self) -> int:
        pass

    def get_gap(self) -> float:
        return self._solver.getGap()

    def add_objective_offset(self, offset: float | bool, overwrite: bool = True) -> None:
        offset = self._to_float(value=offset)
        if overwrite:
            offset -= self._solver.getObjoffset()

        self._solver.addObjoffset(offset)

    def get_dual_value(self, constraint: pyscipopt.Constraint) -> float:
        raise ValueError("Cannot get dual variables from the SCIP solver")

    def get_named_objective(self, name: str) -> float:
        if name in self._named_objectives:
            return float(
                np.sum(
                    [
                        (
                            0.0
                            if abs(item.objective_coefficient) <= 1e-9
                            else item.objective_coefficient * self.get_variable_value(item.variable)
                        )
                        for item in self._named_objectives[name]
                        if item.objective_coefficient
                    ]
                )
            )
        return 0.0
