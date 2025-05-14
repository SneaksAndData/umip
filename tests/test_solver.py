"""
Generally, Gurobi and LocalSolver are not tested because they are not open source and require a license to run.
Open source implementations are tested below.
"""
import pytest
import numpy as np

from generic_mip.abstract_solver import AbstractOptimizationSolver
from generic_mip.enums.constraint_type import ConstraintType
from generic_mip.enums.variable_data_type import VariableDataType


@pytest.mark.parametrize(
    "solver",
    [
        "OrTools",
        "Highs",
    ],
    indirect=True,
)
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


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
@pytest.mark.parametrize(
    "ctype", [ConstraintType.LESS_THAN_OR_EQUAL, ConstraintType.EQUAL, ConstraintType.GREATER_THAN_OR_EQUAL]
)
def test_add_constraint_of_type(solver: AbstractOptimizationSolver, ctype: ConstraintType):
    """
    Testing that adding a constraint of a constraint type is reflected in the solver.
    """
    # Arrange
    var = solver.add_variable(lb=0, ub=1, name="x", dtype=VariableDataType.FLOAT)
    solver.add_constraint_of_type(constraint_type=ctype, right_hand_side=1, coeffs=1, vars_=var, name="c1")
    solver.force_update()

    # Act & Assert
    assert solver.get_constraint_count() == 1


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_add_multiple_var_and_constr(solver: AbstractOptimizationSolver):
    """
    Testing that adding 3 variables, 3 constraints, and 3 objective terms is reflected in the solver.
    """
    vars_ = solver.add_multiple_variables(
        lb=0, ub=1, names=np.array(["x_1", "x_2", "x_3"]), dtype=VariableDataType.FLOAT
    )
    # Notice that setting both lb and ub may result in 2 constraints in some implementations
    solver.add_multiple_constraints(
        lb=np.array([0.0, 0.0, 0.0]),
        ub=None,
        coeffs=np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
        vars_=np.array([vars_] * 3),
        names=np.array(["c1", "c2", "c3"]),
    )
    solver.add_multiple_objective_terms(coeffs=np.array([1.0, 1.0, 1.0]), vars_=vars_, overwrite=False)
    solver.force_update()

    assert len(vars_) == 3
    assert solver.get_variable_count() == 3
    assert solver.get_constraint_count() == 3
    assert solver.get_objective_terms_count() == 3


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_add_multiple_objectives_with_names__named_objectives_is_as_expected(solver: AbstractOptimizationSolver):
    # Arrange
    vars_ = solver.add_multiple_variables(
        lb=0, ub=1, names=np.array(["x_1", "x_2", "x_3"]), dtype=VariableDataType.BOOL
    )

    objective1 = "Objective1"
    objective2 = "Objective2"

    solver.add_multiple_objective_terms(coeffs=np.array([1.0, 1.0, 1.0]), vars_=vars_, overwrite=False, name=objective1)
    solver.add_multiple_objective_terms(coeffs=np.array([2.0, 2.0, 2.0]), vars_=vars_, overwrite=False, name=objective2)
    solver.force_update()
    solver.set_optimization_direction(True)
    solver.solve()

    # Act & Assert
    assert solver.get_named_objective(objective1) == 3.0
    assert solver.get_named_objective(objective2) == 6.0
    assert solver.get_objective_value() == 9.0


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_add_objectives_with_names__named_objectives_is_as_expected(solver: AbstractOptimizationSolver):
    # Arrange
    vars_ = solver.add_multiple_variables(
        lb=0, ub=1, names=np.array(["x_1", "x_2", "x_3"]), dtype=VariableDataType.BOOL
    )

    objective1 = "Objective1"
    objective2 = "Objective2"

    for var in vars_:
        solver.add_objective_term(coeff=1, var=var, overwrite=False, name=objective1)
        solver.add_objective_term(coeff=2, var=var, overwrite=False, name=objective2)

    solver.force_update()
    solver.set_optimization_direction(True)
    solver.solve()

    # Act & Assert
    assert solver.get_named_objective(objective1) == 3.0
    assert solver.get_named_objective(objective2) == 6.0
    assert solver.get_objective_value() == 9.0


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_add_objectives_with_names__raises_error_when_overwrite(solver: AbstractOptimizationSolver):
    # Arrange
    vars_ = solver.add_multiple_variables(
        lb=0, ub=1, names=np.array(["x_1", "x_2", "x_3"]), dtype=VariableDataType.BOOL
    )

    # Act & Assert
    with pytest.raises(ValueError) as valueError:
        solver.add_multiple_objective_terms(coeffs=np.array([1.0, 1.0, 1.0]), vars_=vars_, overwrite=True, name="Test")


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
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
    assert solver.is_feasible()
    assert not solver.is_abnormal()
    assert not solver.is_unbounded()
    assert not solver.is_infeasible()
    assert solver.get_objective_value() == int(maximisation)
    assert solver.get_variable_value(var) == int(maximisation)
    # assert solver.get_gap() == 0.0


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
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


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
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


@pytest.mark.parametrize("solver", ["OrTools", "Scip"], indirect=True)
@pytest.mark.parametrize("verbose", [True, False])
def test_set_verbose(capfd, solver: AbstractOptimizationSolver, verbose: bool):
    """
    Testing that the solver verbosity is set correctly.
    Highs always print at least one line of output, thus, it is not tested here.
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


@pytest.mark.parametrize("solver", ["OrTools", "Scip"], indirect=True)
def test_time_limit_gap(solver: AbstractOptimizationSolver):
    """
    Testing that solving a computationally difficult problem stops at the time limit and
    provides a non-zero gap because it is not done solving.
    """
    number_of_cities = 200
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
        solver.add_multiple_objective_terms(coeffs=np.array(c[i]), vars_=np.array(x[i]), overwrite=False)

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
    if type(solver).__name__ == "OrToolsSolver":
        solver.set_solver_setting("presolving/maxrounds=0")
    if type(solver).__name__ == "ScipSolver":
        solver.set_param("presolving/maxrounds", 0)
    solver.set_verbose(verbose=True)
    solver.solve(time_limit=1)
    gap = solver.get_gap()

    assert gap is not None and gap > 0


@pytest.mark.parametrize("solver", ["OrTools", "Highs"], indirect=True)
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


@pytest.mark.parametrize("solver", ["OrTools", "Highs"], indirect=True)
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("overwrite", [False, True])
def test_objective_offset(solver: AbstractOptimizationSolver, offset: int, overwrite: bool):
    """
    Testing that offset in objective function is set correctly both when overwriting and adding.
    """
    solver.set_optimization_direction(maximization=False)
    var = solver.add_variable(lb=0.0, ub=100.0, name="x", dtype=VariableDataType.FLOAT)
    solver.add_constraint(lb=0.0, ub=1.0, coeffs=np.array([1.0]), vars_=np.array([var]), name="c1")
    solver.add_objective_term(coeff=1.0, var=var, overwrite=False)
    solver.add_objective_offset(offset=offset, overwrite=overwrite)
    solver.add_objective_offset(offset=offset, overwrite=overwrite)
    solver.solve()

    assert solver.get_objective_value() == offset + offset * (1 - overwrite)


@pytest.mark.parametrize("solver", ["Scip"], indirect=True)
@pytest.mark.parametrize("offset", [1])
def test_objective_offset_for_scip(solver: AbstractOptimizationSolver, offset: int):
    """
    Testing that offset in objective function is set correctly both when overwriting and adding.
    """
    solver.set_optimization_direction(maximization=False)
    var = solver.add_variable(lb=0.0, ub=100.0, name="x", dtype=VariableDataType.FLOAT)
    solver.add_constraint(lb=0.0, ub=1.0, coeffs=np.array([1.0]), vars_=np.array([var]), name="c1")
    solver.add_objective_term(coeff=1.0, var=var, overwrite=False)
    solver.add_objective_offset(offset=offset, overwrite=False)
    solver.add_objective_offset(offset=offset, overwrite=False)
    solver.solve()

    assert solver.get_objective_value() == offset + offset


@pytest.mark.parametrize("solver", ["Highs"], indirect=True)
def test_get_dual_value(solver: AbstractOptimizationSolver):
    """
    Testing that the dual value is retrieved correctly.
    OR Tools does not support dual values, so it is not tested.
    """
    solver.set_optimization_direction(maximization=True)
    var = solver.add_variable(lb=-solver.infinity(), ub=solver.infinity(), name="x", dtype=VariableDataType.FLOAT)

    constr = solver.add_constraint(
        lb=-solver.infinity(), ub=1.0, coeffs=np.array([1.0]), vars_=np.array([var]), name="c1"
    )

    solver.add_objective_term(coeff=10, var=var, overwrite=False)
    solver.solve()
    assert solver.get_dual_value(constr) == 10


@pytest.mark.parametrize("solver", ["Highs", "Scip"], indirect=True)
def test_integer_problem(solver: AbstractOptimizationSolver):
    """
    Testing that the dual value is retrieved correctly.
    OR Tools does not support dual values, so it is not tested.
    """
    solver.set_optimization_direction(maximization=True)
    var = solver.add_variable(lb=-solver.infinity(), ub=solver.infinity(), name="x", dtype=VariableDataType.INT)
    constr = solver.add_constraint(
        lb=-solver.infinity(), ub=1.0, coeffs=np.array([1.0]), vars_=np.array([var]), name="c1"
    )
    solver.add_objective_term(coeff=10, var=var, overwrite=False)
    solver.solve()
    with pytest.raises(ValueError):
        assert solver.get_dual_value(constr)


@pytest.mark.parametrize("solver", ["Highs", "OrTools", "Scip"], indirect=True)
def test_get_variable_count_of_type(solver: AbstractOptimizationSolver):
    # Arrange
    number_of_continuous_variables = 5
    number_of_binary_variables = 7
    number_of_integer_variables = 10
    total_variables_count = number_of_continuous_variables + number_of_binary_variables + number_of_integer_variables
    [
        solver.add_variable(lb=0, ub=0, dtype=VariableDataType.FLOAT, name="C[" + str(i) + "]")
        for i in range(number_of_continuous_variables)
    ]
    [
        solver.add_variable(lb=0, ub=0, dtype=VariableDataType.BOOL, name="B[" + str(i) + "]")
        for i in range(number_of_binary_variables)
    ]
    [
        solver.add_variable(lb=0, ub=0, dtype=VariableDataType.INT, name="I[" + str(i) + "]")
        for i in range(number_of_integer_variables)
    ]

    # Act & Assert
    assert solver.get_variable_count_of_type(VariableDataType.FLOAT) == number_of_continuous_variables
    assert solver.get_variable_count_of_type(VariableDataType.BOOL) == number_of_binary_variables
    assert solver.get_variable_count_of_type(VariableDataType.INT) == number_of_integer_variables
    assert solver.get_variable_count() == total_variables_count


@pytest.mark.parametrize("solver", ["Highs", "OrTools", "Scip"], indirect=True)
def test_mip_gap_limit_gap(solver: AbstractOptimizationSolver):
    """
    The travelling salesman problem (TSP) is trying to find the most cost-efficient route for the salesman to visit
    each city exactly once and returns to the origin city. It is an NP-hard problem in combinatorial optimization.
    This test considers 100 cities and travel costs between cities are assigned randomly within (0,1), the
    randomness is controlled by seed. The goal of the test is to solve a computationally difficult TSP and stop at
    the mip gap limit while providing a gap that is equal or lower than mip_gap_limit.
    """
    number_of_cities = 100
    cities = list(range(0, number_of_cities))

    np.random.seed(number_of_cities)
    c = [[np.random.rand() for _ in cities] for _ in cities]
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

    solver.set_verbose(verbose=True)
    solver.solve(mip_gap_limit=0.7)
    gap = solver.get_gap()

    assert gap is not None and gap <= 0.7
