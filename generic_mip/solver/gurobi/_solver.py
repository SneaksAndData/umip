"""A solver implemented in the Gurobi library."""
from typing import Iterable
import gurobipy as gp
import numpy.typing as npt
import numpy as np
from scipy.sparse import coo_matrix
from adapta.logs import LoggerInterface
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.variable_data_type import VariableDataType


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

    def set_variable_hint(self, var: gp.Var, hint: float) -> None:
        raise NotImplementedError()

    def set_multiple_variable_hints(self, vars_: npt.NDArray[gp.Var], hints: npt.NDArray[float]) -> None:
        raise NotImplementedError()

    def add_multiple_objective_terms(
        self, coeffs: npt.NDArray[float], vars_: npt.NDArray[gp.Var], overwrite: bool = True, name: str = None
    ) -> None:
        if overwrite:
            for var in vars_:
                self._objective.remove(var)

        if name is not None:
            self.add_named_objective(coeffs, vars_, overwrite, name)
        self._objective.addTerms(coeffs, vars_)

    def add_multiple_variables(
        self,
        names: npt.NDArray[str],
        dtype: VariableDataType,
        lb: float | None = None,
        ub: float | None = None,
    ) -> npt.NDArray[gp.Var]:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        if dtype == VariableDataType.INT:
            vars_ = self._solver.addMVar(shape=(len(names),), lb=lb, ub=ub, obj=0.0, vtype=gp.GRB.INTEGER, name=names)
            self._integer_problem = True
        elif dtype == VariableDataType.FLOAT:
            vars_ = self._solver.addMVar(
                shape=(len(names),), lb=lb, ub=ub, obj=0.0, vtype=gp.GRB.CONTINUOUS, name=names
            )
        elif dtype == VariableDataType.BOOL:
            vars_ = self._solver.addMVar(shape=(len(names),), lb=lb, ub=ub, obj=0.0, vtype=gp.GRB.BINARY, name=names)
            self._integer_problem = True
        else:
            raise ValueError("Unsupported variable data type")

        return vars_.tolist()

    def add_multiple_constraints(
        self,
        coeffs: npt.NDArray[npt.NDArray[float]] | npt.NDArray[float],
        vars_: npt.NDArray[npt.NDArray[gp.Var]] | npt.NDArray[gp.Var],
        lb: npt.NDArray[float] | None = None,
        ub: npt.NDArray[float] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        if coeffs.size == 0:
            return

        if coeffs.ndim == 1 and not isinstance(coeffs[0], np.ndarray):
            coeffs = np.asarray([coeffs]).T
            vars_ = np.asarray([vars_]).T

        coeff_list = np.concatenate(coeffs)
        var_vector = np.concatenate(vars_)

        matrix_rows = [j for i, coeff in enumerate(coeffs) for j in [i] * len(coeff)]
        matrix_cols = list(range(len(var_vector)))
        matrix_data = coeff_list

        coeff_matrix = coo_matrix(
            (matrix_data.astype(float), (matrix_rows, matrix_cols)), shape=(len(coeffs), len(var_vector))
        )
        var_vector = var_vector.tolist()

        if lb is not None:
            self._solver.addMConstr(coeff_matrix, var_vector, gp.GRB.GREATER_EQUAL, lb.tolist())
        if ub is not None:
            self._solver.addMConstr(coeff_matrix, var_vector, gp.GRB.LESS_EQUAL, ub.tolist())

    def add_constraint(
        self,
        coeffs: npt.NDArray[float] | float,
        vars_: npt.NDArray[gp.Var] | gp.Var,
        lb: float | None = None,
        ub: float | None = None,
        name: str | None = None,
    ) -> gp.Constr | None:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        constr_expr = gp.LinExpr()
        if isinstance(vars_, Iterable):
            for coeff, var in zip(coeffs, vars_):
                constr_expr += coeff * var
        else:
            constr_expr = coeffs * vars_

        if lb == ub:
            return self._solver.addConstr(lb == constr_expr, name)

        constr_lb, constr_ub = None, None
        if lb != -self.infinity():
            constr_lb: gp.Constr = self._solver.addConstr(lb <= constr_expr, name)
        if ub != self.infinity():
            constr_ub: gp.Constr = self._solver.addConstr(constr_expr <= ub, name)

        return constr_lb or constr_ub

    def add_variable(
        self, name: str, dtype: VariableDataType, lb: float | None = None, ub: float | None = None
    ) -> gp.Var:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()
        if dtype == VariableDataType.INT:
            var = self._solver.addVar(lb, ub, 0, gp.GRB.INTEGER, name, None)
            self._integer_problem = True
        elif dtype == VariableDataType.FLOAT:
            var = self._solver.addVar(lb, ub, 0, gp.GRB.CONTINUOUS, name, None)
        elif dtype == VariableDataType.BOOL:
            var = self._solver.addVar(lb, ub, 0, gp.GRB.BINARY, name, None)
            self._integer_problem = True
        else:
            raise ValueError("Unsupported variable data type")
        return var

    def get_variable_value(self, var: gp.Var) -> float:
        return var.x

    def add_objective_term(self, coeff: float, var: gp.Var, overwrite: bool = True, name: str = None) -> None:
        if overwrite:
            # Might have bad performance?
            self._objective.remove(var)

        if name is not None:
            self.add_named_objective(np.array([coeff]), np.array([var]), overwrite, name)

        self._objective.addTerms([coeff], [var])

    def set_optimization_direction(self, maximization: bool) -> None:
        self._solver.setAttr(gp.GRB.Attr.ModelSense, gp.GRB.MAXIMIZE if maximization else gp.GRB.MINIMIZE)

    def get_objective_value(self) -> float:
        return self._solver.getObjective().getValue()

    def solve(self, time_limit: float | None = None, mip_gap_limit: float | None = None) -> int:
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

    def set_solver_setting(self, setting: str) -> None:
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

    def get_variable_count_of_type(self, var_type: VariableDataType):
        if var_type == VariableDataType.FLOAT:
            return self._solver.NumVars - self._solver.NumIntVars
        if var_type == VariableDataType.INT:
            return self._solver.NumIntVars - self._solver.NumBinVars
        if var_type == VariableDataType.BOOL:
            return self._solver.NumBinVars

        raise ValueError(f"Unsupported variable data type: {var_type}")

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

    def add_objective_offset(self, offset: float, overwrite: bool = True):
        if overwrite:
            self._objective -= self._objective.getConstant()
        self._objective += offset

    def get_dual_value(self, constraint: gp.Constr) -> float:
        if self._integer_problem:
            raise ValueError("Dual values are not available for integer problems")
        if not self.is_optimal():
            raise ValueError("Dual values are only available for optimal solutions")
        return constraint.getAttr(gp.GRB.Attr.Pi)
