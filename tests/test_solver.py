import pytest
import numpy as np
from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.solver.gurobi import GurobiSolver
from generic_mip.variable_data_type import VariableDataType


@pytest.mark.parametrize("solver", ["Gurobi", "OrTools"], indirect=True)
def test_add_var_and_constr(solver: AbstractOptimizationSolver):
    """
    Testing that adding a variable, a constraint, and an objective term is reflected in the solver.
    """
    var = solver.add_variable(lb=0, ub=1, name="x", dtype=VariableDataType.FLOAT)
    # Notice that setting both lb and ub may result in 2 constraints in some implementations
    solver.add_constraint(lb=0, ub=None, coeffs=1, vars_=var, name="c1")
    solver.add_objective_term(coeff=1, var=var, overwrite=False)
    solver.force_update()

    assert solver.get_constraint_count() == 1
    assert solver.get_variable_count() == 1
    assert solver.get_objective_terms_count() == 1


@pytest.mark.parametrize("solver", ["Gurobi", "OrTools"], indirect=True)
def test_add_multiple_var_and_constr(solver: AbstractOptimizationSolver):
    """
    Testing that adding 3 variables, 3 constraints, and 3 objective terms is reflected in the solver.
    """
    vars_ = solver.add_multiple_variables(count=3, lb=0, ub=1, name="x", dtype=VariableDataType.FLOAT)
    # Notice that setting both lb and ub may result in 2 constraints in some implementations
    solver.add_multiple_constraints(
        lb=np.array([0.0, 0.0, 0.0]),
        ub=None,
        coeffs=np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
        vars_=np.array([vars_] * 3),
        name="c1",
    )
    solver.add_multiple_objective_terms(coeffs=np.array([1.0, 1.0, 1.0]), vars_=vars_, overwrite=False)
    solver.force_update()

    assert len(vars_) == 3
    assert solver.get_variable_count() == 3
    assert solver.get_constraint_count() == 3
    assert solver.get_objective_terms_count() == 3


@pytest.mark.parametrize("solver", ["Gurobi", "OrTools"], indirect=True)
@pytest.mark.parametrize("dtype", [VariableDataType.FLOAT, VariableDataType.INT, VariableDataType.BOOL])
@pytest.mark.parametrize("maximisation", [True, False])
def test_optimal_solution(solver: AbstractOptimizationSolver, dtype: VariableDataType, maximisation: bool):
    """
    Testing that the solver returns the known optimal solution, and it is reflected in the optimisation status.
    """
    solver.set_optimization_direction(maximization=maximisation)
    var = solver.add_variable(lb=0.0, ub=100.0, name="x", dtype=dtype)
    solver.add_constraint(lb=0.0, ub=1.0, coeffs=np.array([1.0]), vars_=np.array([var]), name="c1")
    solver.add_objective_term(coeff=1.0, var=var, overwrite=False)
    solver.solve()

    assert solver.is_optimal()
    assert not solver.is_abnormal()
    assert not solver.is_unbounded()
    assert not solver.is_infeasible()
    assert solver.get_objective_value() == int(maximisation)
    assert solver.get_variable_value(var) == int(maximisation)
    assert solver.get_gap() == 0.0


@pytest.mark.parametrize("solver", ["Gurobi", "OrTools"], indirect=True)
@pytest.mark.parametrize("dtype", [VariableDataType.FLOAT, VariableDataType.INT])
def test_unbounded_problem(solver: AbstractOptimizationSolver, dtype: VariableDataType):
    """
    Testing that the solver recognises an unbounded solution, and it is reflected in the optimisation status.
    """
    solver.set_optimization_direction(maximization=True)
    var = solver.add_variable(lb=0.0, ub=solver.infinity(), name="x", dtype=dtype)
    solver.add_objective_term(coeff=1.0, var=var, overwrite=False)
    solver.solve()

    assert not solver.is_optimal()
    assert not solver.is_abnormal()
    assert solver.is_unbounded()
    assert not solver.is_infeasible()


@pytest.mark.parametrize("solver", ["Gurobi", "OrTools"], indirect=True)
@pytest.mark.parametrize("dtype", [VariableDataType.FLOAT, VariableDataType.INT])
def test_infeasible_problem(solver: AbstractOptimizationSolver, dtype: VariableDataType):
    """
    Testing that the solver recognises an infeasible solution, and it is reflected in the optimisation status.
    """
    var = solver.add_variable(lb=0.0, ub=1.0, name="x", dtype=dtype)
    solver.add_constraint(lb=5.0, ub=6.0, coeffs=np.array([1.0]), vars_=np.array([var]), name="c1")
    solver.add_objective_term(coeff=1.0, var=var, overwrite=False)
    solver.solve()

    assert not solver.is_optimal()
    assert not solver.is_abnormal()
    assert not solver.is_unbounded()
    assert solver.is_infeasible()


@pytest.mark.parametrize("solver", ["Gurobi", "OrTools"], indirect=True)
@pytest.mark.parametrize("verbose", [True, False])
def test_set_verbose(capfd, solver: AbstractOptimizationSolver, verbose: bool):
    """
    Testing that the solver verbosity is set correctly.
    """
    var = solver.add_variable(lb=0.0, ub=100.0, name="x", dtype=VariableDataType.FLOAT)
    solver.add_constraint(lb=0.0, ub=1.0, coeffs=np.array([1.0]), vars_=np.array([var]), name="c1")
    solver.add_objective_term(coeff=1.0, var=var, overwrite=False)
    solver.set_verbose(verbose=verbose)
    solver.solve()

    out, err = capfd.readouterr()

    if verbose:
        assert len(out) > 0
    else:
        assert len(out) == 0


@pytest.mark.parametrize("solver", ["OrTools"], indirect=True)
def test_time_limit_gap(solver: AbstractOptimizationSolver):
    """
    Testing that solving a computationally difficult problem stops at the time limit and
    provides a non-zero gap because it is not done solving.
    """
    number_of_cities = 100
    cities = list(range(0, number_of_cities))

    c = [[10.0 for _ in cities] for _ in cities]
    x = [
        [solver.add_variable(lb=0.0, ub=1.0, name=f"x{i},{j}", dtype=VariableDataType.BOOL) for j in cities]
        for i in cities
    ]
    u = [None] + [
        solver.add_variable(lb=1.0, ub=number_of_cities - 1, name=f"u{i}", dtype=VariableDataType.INT)
        for i in cities[1:]
    ]

    for i in cities:
        for j in cities:
            if i != j:
                solver.add_objective_term(coeff=c[i][j], var=x[i][j], overwrite=False)

    for i in cities:
        solver.add_constraint(
            lb=1.0,
            ub=1.0,
            coeffs=np.array([1.0 for j in cities if i != j]),
            vars_=np.array([x[i][j] for j in cities if i != j]),
        )

    for j in cities:
        solver.add_constraint(
            lb=1.0,
            ub=1.0,
            coeffs=np.array([1.0 for i in cities if i != j]),
            vars_=np.array([x[i][j] for i in cities if i != j]),
        )

    for i in cities[1:]:
        for j in cities[1:]:
            if i != j:
                solver.add_constraint(
                    lb=None,
                    ub=number_of_cities - 2,
                    coeffs=np.array([1.0, -1.0, number_of_cities - 1]),
                    vars_=np.array([u[i], u[j], x[i][j]]),
                )

    solver.set_solver_setting("presolving/maxrounds=0")
    solver.set_verbose(verbose=True)
    solver.solve(time_limit=2.0)
    gap = solver.get_gap()

    assert gap is not None and gap > 0


@pytest.mark.parametrize("solver", ["OrTools", "Gurobi"], indirect=True)
@pytest.mark.parametrize("overwrite", [True, False])
def test_add_objective_term(solver: AbstractOptimizationSolver, overwrite: bool):
    """
    Testing that the solver setting for OrTools overwrite in objective function is set correctly.
    """
    x = solver.add_variable(lb=0, ub=100, name="x", dtype=VariableDataType.INT)
    y = solver.add_variable(lb=0, ub=100, name="y", dtype=VariableDataType.INT)

    solver.add_multiple_objective_terms(coeffs=np.array([1.0, 4.0]), vars_=np.array([x, y]), overwrite=overwrite)
    solver.add_multiple_objective_terms(coeffs=np.array([0.0, 0.0]), vars_=np.array([x, y]), overwrite=overwrite)

    solver.add_constraint(lb=None, ub=100, coeffs=np.array([1.0, 1.0]), vars_=np.array([x, y]), name="my_constraint")
    solver.add_constraint(lb=None, ub=20, coeffs=np.array([1.0]), vars_=np.array([y]), name="my_constraint")
    solver.set_optimization_direction(True)
    solver.set_verbose(True)
    solver.solve()

    obj_value = solver.get_objective_value()

    if overwrite:
        assert obj_value == 0.0
    else:
        assert obj_value == 160.0


@pytest.mark.parametrize("solver", ["Gurobi", "OrTools"], indirect=True)
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("overwrite", [False, True])
def test_optimal_solution(solver: AbstractOptimizationSolver, offset: int, overwrite: bool):
    """
    Testing that the solver returns the known optimal solution, and it is reflected in the optimisation status.
    """
    solver.set_optimization_direction(maximization=False)
    var = solver.add_variable(lb=0.0, ub=100.0, name="x", dtype=VariableDataType.FLOAT)
    solver.add_constraint(lb=0.0, ub=1.0, coeffs=np.array([1.0]), vars_=np.array([var]), name="c1")
    solver.add_objective_term(coeff=1.0, var=var, overwrite=True)
    solver.add_objective_offset(offset=offset, overwrite=overwrite)
    solver.add_objective_offset(offset=offset, overwrite=overwrite)
    solver.solve()

    assert solver.get_objective_value() == offset + offset * (1 - overwrite)
