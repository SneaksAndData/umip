"""A solver implemented in the Gurobi library."""

from typing import Iterable
import gurobipy as gp
import numpy.typing as npt
import numpy as np
from scipy.sparse import coo_matrix
from adapta.logs import LoggerInterface
from umip.abstract_solver import AbstractOptimizationSolver
from umip.enums.variable_domain import VariableDomain
from umip.solver_config import GurobiSolverConfig


class GurobiSolver(AbstractOptimizationSolver[gp.Var, gp.Constr]):  # pylint: disable=too-many-public-methods
    """A solver implemented in the Gurobi library."""

    def __init__(self, logger: LoggerInterface, model_path: str | None = None):
        """
        Initialize the solver.

        :param model_path: The path to the model file to read (mps or lp).
        :param logger: The logger to use.
        """
        super().__init__(logger)
        if model_path is not None:
            self._solver = gp.read(model_path)
        else:
            self._solver = gp.Model()
        self._objective = gp.LinExpr()
        self._solver.setParam(gp.GRB.Param.LogToConsole, 0)
        self.status = None

    def set_variable_hint(self, variable: gp.Var, hint: float | int | bool) -> None:
        raise NotImplementedError()

    def set_multiple_variable_hints(
        self,
        variables: npt.NDArray[gp.Var],
        hints: npt.NDArray[np.floating | np.integer | np.bool_],
    ) -> None:
        raise NotImplementedError()

    def add_multiple_objective_terms(
        self,
        coefficients: npt.NDArray[np.floating | np.integer | np.bool_],
        variables: npt.NDArray[gp.Var],
        overwrite: bool = True,
        name: str = None,
    ) -> None:
        coefficients = self._to_float(value=coefficients)
        if overwrite:
            for var in variables:
                self._objective.remove(var)

        if name is not None:
            self.add_named_objective(coefficients, variables, overwrite, name)
        self._objective.addTerms(coefficients, variables)

    def add_multiple_variables(
        self,
        names: npt.NDArray[str],
        variable_domain: VariableDomain,
        lower_bound: float | int | bool | None = None,
        upper_bound: float | int | bool | None = None,
    ) -> npt.NDArray[gp.Var]:
        lower_bound = (
            self._to_float(value=lower_bound)
            if lower_bound is not None
            else -self.infinity()
        )
        upper_bound = (
            self._to_float(value=upper_bound)
            if upper_bound is not None
            else self.infinity()
        )

        if variable_domain == VariableDomain.INTEGER:
            vars_ = self._solver.addMVar(
                shape=(len(names),),
                lb=lower_bound,
                ub=upper_bound,
                obj=0.0,
                vtype=gp.GRB.INTEGER,
                name=names,
            )
            self._integer_problem = True
        elif variable_domain == VariableDomain.CONTINUOUS:
            vars_ = self._solver.addMVar(
                shape=(len(names),),
                lb=lower_bound,
                ub=upper_bound,
                obj=0.0,
                vtype=gp.GRB.CONTINUOUS,
                name=names,
            )
        elif variable_domain == VariableDomain.BINARY:
            vars_ = self._solver.addMVar(
                shape=(len(names),),
                lb=lower_bound,
                ub=upper_bound,
                obj=0.0,
                vtype=gp.GRB.BINARY,
                name=names,
            )
            self._integer_problem = True
        else:
            raise ValueError("Unsupported variable data type")

        return vars_.tolist()

    def add_multiple_constraints(
        self,
        coefficients: npt.NDArray[np.floating | np.integer | np.bool_],
        variables: npt.NDArray[npt.NDArray[gp.Var]] | npt.NDArray[gp.Var],
        lower_bounds: npt.NDArray[np.floating | np.integer | np.bool_] | None = None,
        upper_bounds: npt.NDArray[np.floating | np.integer | np.bool_] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        if coefficients.size == 0:
            return

        coefficients = self._to_float(value=coefficients)

        if coefficients.ndim == 1 and not isinstance(coefficients[0], np.ndarray):
            coefficients = np.asarray([coefficients]).T
            variables = np.asarray([variables]).T

        coeff_list = np.concatenate(coefficients)
        var_vector = np.concatenate(variables)

        matrix_rows = [
            j for i, coeff in enumerate(coefficients) for j in [i] * len(coeff)
        ]
        matrix_cols = list(range(len(var_vector)))
        matrix_data = coeff_list

        coeff_matrix = coo_matrix(
            (matrix_data.astype(float), (matrix_rows, matrix_cols)),
            shape=(len(coefficients), len(var_vector)),
        )
        var_vector = var_vector.tolist()

        if lower_bounds is not None:
            lower_bounds = self._to_float(value=lower_bounds)
            self._solver.addMConstr(
                coeff_matrix, var_vector, gp.GRB.GREATER_EQUAL, lower_bounds.tolist()
            )
        if upper_bounds is not None:
            upper_bounds = self._to_float(value=upper_bounds)
            self._solver.addMConstr(
                coeff_matrix, var_vector, gp.GRB.LESS_EQUAL, upper_bounds.tolist()
            )

    def add_constraint(
        self,
        coefficients: npt.NDArray[np.floating | np.integer | np.bool_]
        | float
        | int
        | bool,
        variables: npt.NDArray[gp.Var] | gp.Var,
        lower_bound: float | int | bool | None = None,
        upper_bound: float | int | bool | None = None,
        name: str | None = None,
    ) -> gp.Constr | None:
        coefficients = self._to_float(value=coefficients)
        lower_bound = (
            self._to_float(value=lower_bound)
            if lower_bound is not None
            else -self.infinity()
        )
        upper_bound = (
            self._to_float(value=upper_bound)
            if upper_bound is not None
            else self.infinity()
        )

        constr_expr = gp.LinExpr()
        if isinstance(variables, Iterable):
            for coeff, var in zip(coefficients, variables):
                constr_expr += coeff * var
        else:
            constr_expr = coefficients * variables

        if lower_bound == upper_bound:
            return self._solver.addConstr(lower_bound == constr_expr, name)

        constr_lb, constr_ub = None, None
        if lower_bound != -self.infinity():
            constr_lb: gp.Constr = self._solver.addConstr(
                lower_bound <= constr_expr, name
            )
        if upper_bound != self.infinity():
            constr_ub: gp.Constr = self._solver.addConstr(
                constr_expr <= upper_bound, name
            )

        return constr_lb or constr_ub

    def add_variable(
        self,
        name: str,
        variable_domain: VariableDomain,
        lower_bound: float | int | bool | None = None,
        upper_bound: float | int | bool | None = None,
    ) -> gp.Var:
        lower_bound = (
            self._to_float(value=lower_bound)
            if lower_bound is not None
            else -self.infinity()
        )
        upper_bound = (
            self._to_float(value=upper_bound)
            if upper_bound is not None
            else self.infinity()
        )
        if variable_domain == VariableDomain.INTEGER:
            var = self._solver.addVar(
                lower_bound, upper_bound, 0, gp.GRB.INTEGER, name, None
            )
            self._integer_problem = True
        elif variable_domain == VariableDomain.CONTINUOUS:
            var = self._solver.addVar(
                lower_bound, upper_bound, 0, gp.GRB.CONTINUOUS, name, None
            )
        elif variable_domain == VariableDomain.BINARY:
            var = self._solver.addVar(
                lower_bound, upper_bound, 0, gp.GRB.BINARY, name, None
            )
            self._integer_problem = True
        else:
            raise ValueError("Unsupported variable domain")
        return var

    def get_variable_value(self, var: gp.Var) -> float:
        return var.x

    def add_objective_term(
        self,
        coefficient: float | int | bool,
        variable: gp.Var,
        overwrite: bool = True,
        name: str = None,
    ) -> None:
        coefficient = self._to_float(value=coefficient)
        if overwrite:
            # Might have bad performance?
            self._objective.remove(variable)

        if name is not None:
            self.add_named_objective(
                np.array([coefficient]), np.array([variable]), overwrite, name
            )

        self._objective.addTerms([coefficient], [variable])

    def set_optimization_direction(self, maximization: bool) -> None:
        self._solver.setAttr(
            gp.GRB.Attr.ModelSense, gp.GRB.MAXIMIZE if maximization else gp.GRB.MINIMIZE
        )

    def get_objective_value(self) -> float:
        return self._solver.getObjective().getValue()

    def solve(
        self, time_limit: float | None = None, mip_gap_limit: float | None = None
    ) -> int:
        if time_limit is not None:
            self._solver.setParam(gp.GRB.Param.TimeLimit, time_limit)
        if mip_gap_limit is not None:
            self._solver.setParam(gp.GRB.Param.MIPGap, mip_gap_limit)
        self._solver.setObjective(self._objective)
        self._solver.optimize()
        self.status = self._solver.getAttr(gp.GRB.Attr.Status)
        return self.status

    def get_constraint(self, name: str) -> gp.Constr:
        return self._solver.getConstrByName(name)

    def infinity(self) -> float:
        return gp.GRB.INFINITY

    def is_optimal(self) -> bool:
        return self.status == gp.GRB.OPTIMAL

    def is_feasible(self) -> bool:
        return self.status == gp.GRB.OPTIMAL.SUBOPTIMAL

    def is_infeasible(self) -> bool:
        return self.status == gp.GRB.INFEASIBLE

    def is_unbounded(self) -> bool:
        return self.status == gp.GRB.UNBOUNDED

    def is_abnormal(self) -> bool:
        return self.status in [gp.GRB.NUMERIC, gp.GRB.INF_OR_UNBD, gp.GRB.CUTOFF]

    def is_not_solved(self) -> bool:
        return self.status in [
            gp.GRB.TIME_LIMIT,
            gp.GRB.NODE_LIMIT,
            gp.GRB.ITERATION_LIMIT,
            gp.GRB.INTERRUPTED,
            gp.GRB.SUBOPTIMAL,
            gp.GRB.USER_OBJ_LIMIT,
            gp.GRB.WORK_LIMIT,
        ]

    def set_solver_setting(self, setting: GurobiSolverConfig) -> None:
        raise ValueError("Not supported in Gurobi solver")

    def export_to_file(self, path: str) -> None:
        self._solver.write(path)

    def set_verbose(self, verbose: bool) -> None:
        if verbose:
            self._solver.setParam(gp.GRB.Param.LogToConsole, 1)
        else:
            self._solver.setParam(gp.GRB.Param.LogToConsole, 0)

    def get_constraint_count(self):
        return len(self._solver.getConstrs())

    def get_variable_count(self):
        return len(self._solver.getVars())

    def get_variable_count_of_type(self, variable_domain: VariableDomain):
        if variable_domain == VariableDomain.CONTINUOUS:
            return self._solver.NumVars - self._solver.NumIntVars
        if variable_domain == VariableDomain.INTEGER:
            return self._solver.NumIntVars - self._solver.NumBinVars
        if variable_domain == VariableDomain.BINARY:
            return self._solver.NumBinVars

        raise ValueError(f"Unsupported variable data type: {variable_domain}")

    def get_objective_terms_count(self):
        return self._objective.size()

    def force_update(self):
        self._solver.update()

    def get_gap(self) -> float:
        bound = self._solver.getAttr(gp.GRB.Attr.ObjBound)
        objective_value = self._solver.getAttr(gp.GRB.Attr.ObjVal)
        if objective_value == 0 and self.is_not_solved():
            return self.infinity()
        if objective_value == 0 and bound == 0:
            return 0
        return abs(bound - objective_value) / abs(objective_value)

    def add_objective_offset(self, offset: float | int | bool, overwrite: bool = True):
        offset = self._to_float(value=offset)
        if overwrite:
            self._objective -= self._objective.getConstant()
        self._objective += offset

    def get_dual_value(self, constraint: gp.Constr) -> float:
        if self._integer_problem:
            raise ValueError("Dual values are not available for integer problems")
        if not self.is_optimal():
            raise ValueError("Dual values are only available for optimal solutions")
        return constraint.getAttr(gp.GRB.Attr.Pi)

    def get_named_objective(self, name: str) -> float:
        if name in self._named_objectives:
            return float(
                np.sum(
                    [
                        (
                            item.objective_coefficient
                            * self.get_variable_value(item.variable)
                        )
                        for item in self._named_objectives[name]
                        if item.objective_coefficient
                    ]
                )
            )
        return 0.0
