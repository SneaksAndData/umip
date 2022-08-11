"""A solver implemented in the Gurobi library."""
from typing import Optional, Union, Iterable
import gurobipy as gp
import numpy.typing as npt
import numpy as np
from scipy.sparse import coo_matrix
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.variable_data_type import VariableDataType


class GurobiSolver(AbstractOptimizationSolver[gp.Var, gp.Constr]):  # pylint: disable=too-many-public-methods
    """A solver implemented in the Gurobi library."""
    def __init__(self):
        self._solver = gp.Model()
        self._objective = gp.LinExpr()
        self._solver.setParam(gp.GRB.Param.LogToConsole, 0)
        self.status = None

    def set_variable_hint(self, var: gp.Var, hint: float) -> None:
        raise NotImplementedError()

    def set_multiple_variable_hints(self, vars_: npt.NDArray[gp.Var], hints: npt.NDArray[float]) -> None:
        raise NotImplementedError()

    def add_multiple_objective_terms(self, coeffs: npt.NDArray[float], vars_: npt.NDArray[gp.Var]) -> None:
        self._objective += coeffs.dot(vars_)

    def add_multiple_variables(self, count: int, lb: float, ub: float, name: str, dtype: VariableDataType) -> npt.NDArray[gp.Var]:
        return self._solver.addMVar(shape=(count,), lb=lb, ub=ub, obj=0.0, vtype=gp.GRB.INTEGER, name=name).tolist()

    def add_multiple_constraints(
        self,
        lb: Union[None, npt.NDArray[float]],
        ub: Union[None, npt.NDArray[float]],
        coeffs: Union[npt.NDArray[npt.NDArray[float]], npt.NDArray[float]],
        vars_: Union[npt.NDArray[npt.NDArray[gp.Var]], npt.NDArray[gp.Var]],
        name: Optional[str] = None
    ) -> None:
        if coeffs.ndim == 1 and not isinstance(coeffs[0], np.ndarray):
            coeffs = np.asarray([coeffs]).T
            vars_ = np.asarray([vars_]).T

        coeff_list = np.concatenate([np.asarray(coeff) for coeff in coeffs])
        var_vector = np.concatenate([np.asarray(var) for var in vars_])

        matrix_rows = [j for i, coeff in enumerate(coeffs) for j in [i]*len(coeff)]
        matrix_cols = list(range(len(var_vector)))
        matrix_data = coeff_list

        # coeff_matrix = hstack([diags(coeff, 0) for coeff in coeffs])
        coeff_matrix = coo_matrix((matrix_data, (matrix_rows, matrix_cols)), shape=(len(coeffs),len(var_vector)))
        var_vector = var_vector.tolist()

        if lb is not None:
            self._solver.addMConstr(coeff_matrix, var_vector, gp.GRB.GREATER_EQUAL, lb.tolist())
        if ub is not None:
            self._solver.addMConstr(coeff_matrix, var_vector, gp.GRB.LESS_EQUAL, ub.tolist())

    def add_constraint(
        self,
        lb: Optional[float],
        ub: Optional[float],
        coeffs: Union[npt.NDArray[float], float],
        vars_:  Union[npt.NDArray[any], any],
        name: Optional[str] = None
    ) -> gp.Constr:
        lb = lb if lb is not None else -self.infinity()
        ub = ub if ub is not None else self.infinity()

        constr_expr = gp.LinExpr()
        if isinstance(vars_, Iterable):
            for (coeff, var) in zip(coeffs, vars_):
                constr_expr += coeff*var
        else:
            constr_expr = coeffs*vars_

        if lb == ub:
            return self._solver.addConstr(lb == constr_expr, name)

        constr_lb, constr_ub = None, None
        if lb != -self.infinity():
            constr_lb: gp.Constr = self._solver.addConstr(lb <= constr_expr, name)
        if ub != self.infinity():
            constr_ub: gp.Constr = self._solver.addConstr(constr_expr <= ub, name)

        return constr_lb or constr_ub

    def add_variable(self, lb: float, ub: float, name: str, dtype: VariableDataType) -> gp.Var:
        if dtype == VariableDataType.INT:
            var = self._solver.addVar(lb, ub, 0, gp.GRB.INTEGER, name, None)
        elif dtype == VariableDataType.FLOAT:
            var = self._solver.addVar(lb, ub, 0, gp.GRB.CONTINUOUS, name, None)
        elif dtype == VariableDataType.BOOL:
            var = self._solver.addVar(lb, ub, 0, gp.GRB.BINARY, name, None)
        else:
            raise ValueError("Unsupported variable data type")
        return var

    def get_variable_value(self, var: gp.Var) -> float:
        return var.x

    def add_objective_term(self, coeff: float, var: gp.Var) -> None:
        self._objective += coeff*var

    def set_optimization_direction(self, maximization: bool) -> None:
        self._solver.setAttr(gp.GRB.Attr.ModelSense, gp.GRB.MAXIMIZE if maximization else gp.GRB.MINIMIZE)

    def get_objective_value(self) -> float:
        return self._solver.getObjective().getValue()

    def solve(self) -> int:
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

    def is_infeasible(self) -> bool:
        return self.status == gp.GRB.INFEASIBLE

    def is_unbounded(self) -> bool:
        return self.status == gp.GRB.UNBOUNDED

    def is_abnormal(self) -> bool:
        return self.status not in (gp.GRB.OPTIMAL, gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED)

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

    def get_objective_terms_count(self):
        return self._objective.size()

    def force_update(self):
        self._solver.update()
