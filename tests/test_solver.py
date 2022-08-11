import pytest
import numpy as np
from typing import Type, Callable
from ortools.linear_solver import pywraplp
from generic_mip.solver import GurobiSolver, OrToolsSolver
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.variable_data_type import VariableDataType


@pytest.mark.parametrize("solver", [GurobiSolver(), OrToolsSolver(solver=pywraplp.Solver.CreateSolver('SCIP'))])
def test_add_var_and_constr(solver: AbstractOptimizationSolver):
    """
    Testing that adding a variable, a constraint, and an objective term is reflected in the solver.
    """
    var = solver.add_variable(lb=0, ub=1, name='x', dtype=VariableDataType.FLOAT)
    # Notice that setting both lb and ub may result in 2 constraints in some implementations
    solver.add_constraint(lb=0, ub=None, coeffs=1, vars_=var, name='c1')
    solver.add_objective_term(coeff=1, var=var)
    solver.force_update()

    assert solver.get_constraint_count() == 1
    assert solver.get_variable_count() == 1
    assert solver.get_objective_terms_count() == 1


@pytest.mark.parametrize("solver", [GurobiSolver(), OrToolsSolver(solver=pywraplp.Solver.CreateSolver('SCIP'))])
def test_add_multiple_var_and_constr(solver: AbstractOptimizationSolver):
    """
    Testing that adding 3 variables, 3 constraints, and 3 objective terms is reflected in the solver.
    """
    vars_ = solver.add_multiple_variables(count=3, lb=0, ub=1, name='x', dtype=VariableDataType.FLOAT)
    # Notice that setting both lb and ub may result in 2 constraints in some implementations
    solver.add_multiple_constraints(
        lb=np.array([0.0, 0.0, 0.0]),
        ub=None,
        coeffs=np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
        vars_=np.array([vars_]*3),
        name='c1'
    )
    solver.add_multiple_objective_terms(coeffs=np.array([1.0, 1.0, 1.0]), vars_=vars_)
    solver.force_update()

    assert len(vars_) == 3
    assert solver.get_variable_count() == 3
    assert solver.get_constraint_count() == 3
    assert solver.get_objective_terms_count() == 3


@pytest.mark.parametrize("solver_class, get_solver_args", [
    (GurobiSolver, lambda: {}),
    (OrToolsSolver, lambda: {'solver': pywraplp.Solver.CreateSolver('SCIP')})
])
@pytest.mark.parametrize("dtype", [VariableDataType.FLOAT, VariableDataType.INT, VariableDataType.BOOL])
@pytest.mark.parametrize("maximisation", [True, False])
def test_optimal_solution(solver_class: Type[AbstractOptimizationSolver], get_solver_args: Callable[[], dict], dtype: VariableDataType, maximisation: bool):
    """
    Testing that the solver returns the known optimal solution, and it is reflected in the optimisation status.
    """
    solver = solver_class(**get_solver_args())
    solver.set_optimization_direction(maximization=maximisation)
    var = solver.add_variable(lb=0.0, ub=100.0, name='x', dtype=dtype)
    solver.add_constraint(lb=0.0, ub=1.0, coeffs=np.array([1.0]), vars_=np.array([var]), name='c1')
    solver.add_objective_term(coeff=1.0, var=var)
    solver.solve()

    assert solver.is_optimal()
    assert not solver.is_abnormal()
    assert not solver.is_unbounded()
    assert not solver.is_infeasible()
    assert solver.get_objective_value() == int(maximisation)
    assert solver.get_variable_value(var) == int(maximisation)


@pytest.mark.parametrize("solver_class, get_solver_args", [
    (GurobiSolver, lambda: {}),
    (OrToolsSolver, lambda: {'solver': pywraplp.Solver.CreateSolver('SCIP')})
])
@pytest.mark.parametrize("dtype", [VariableDataType.FLOAT, VariableDataType.INT])
def test_unbounded_problem(solver_class: Type[AbstractOptimizationSolver], get_solver_args: Callable[[], dict], dtype: VariableDataType):
    """
    Testing that the solver recognises an unbounded solution, and it is reflected in the optimisation status.
    """
    solver = solver_class(**get_solver_args())
    solver.set_optimization_direction(maximization=True)
    var = solver.add_variable(lb=0.0, ub=solver.infinity(), name='x', dtype=dtype)
    solver.add_objective_term(coeff=1.0, var=var)
    solver.solve()

    assert not solver.is_optimal()
    assert not solver.is_abnormal()
    assert solver.is_unbounded()
    assert not solver.is_infeasible()


@pytest.mark.parametrize("solver_class, get_solver_args", [
    (GurobiSolver, lambda: {}),
    (OrToolsSolver, lambda: {'solver': pywraplp.Solver.CreateSolver('SCIP')})
])
@pytest.mark.parametrize("dtype", [VariableDataType.FLOAT, VariableDataType.INT])
def test_infeasible_problem(solver_class: Type[AbstractOptimizationSolver], get_solver_args: Callable[[], dict], dtype: VariableDataType):
    """
    Testing that the solver recognises an infeasible solution, and it is reflected in the optimisation status.
    """
    solver = solver_class(**get_solver_args())
    var = solver.add_variable(lb=0.0, ub=1.0, name='x', dtype=dtype)
    solver.add_constraint(lb=5.0, ub=6.0, coeffs=np.array([1.0]), vars_=np.array([var]), name='c1')
    solver.add_objective_term(coeff=1.0, var=var)
    solver.solve()

    assert not solver.is_optimal()
    assert not solver.is_abnormal()
    assert not solver.is_unbounded()
    assert solver.is_infeasible()


@pytest.mark.parametrize("solver", [GurobiSolver(), OrToolsSolver(solver=pywraplp.Solver.CreateSolver('SCIP'))])
@pytest.mark.parametrize("verbose", [True, False])
def test_set_verbose(capfd, solver: AbstractOptimizationSolver, verbose: bool):
    """
    Testing that the solver verbosity is set correctly.
    """
    var = solver.add_variable(lb=0.0, ub=100.0, name='x', dtype=VariableDataType.FLOAT)
    solver.add_constraint(lb=0.0, ub=1.0, coeffs=np.array([1.0]), vars_=np.array([var]), name='c1')
    solver.add_objective_term(coeff=1.0, var=var)
    solver.set_verbose(verbose=verbose)
    solver.solve()

    out, err = capfd.readouterr()

    if verbose:
        assert len(out) > 0
    else:
        assert len(out) == 0
