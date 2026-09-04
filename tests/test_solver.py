"""
Generally, Gurobi and LocalSolver are not tested because they are not open source and require a license to run.
Open source implementations are tested below.
"""

#  Copyright (c) 2026. ECCO Data & AI and other project contributors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np
import pytest

from umip import AbstractOptimizationSolver, VariableDomain
from umip.enums.constraint_type import ConstraintType
from umip.solver_config import OrToolsScipSolverConfig


@pytest.mark.parametrize(
    "solver",
    ["OrTools", "Highs", "Scip"],
    indirect=True,
)
def test_add_var_and_constr(solver: AbstractOptimizationSolver):
    """
    Testing that adding a variable, a constraint, and an objective term is reflected in the solver.
    """
    var = solver.add_variable(
        lower_bound=0,
        upper_bound=1,
        name="x",
        variable_domain=VariableDomain.CONTINUOUS,
    )
    # Notice that setting both lb and ub may result in 2 constraints in some implementations
    solver.add_constraint(lower_bound=0, upper_bound=None, coefficients=1, variables=var, name="c1")
    solver.add_objective_term(coefficient=1, variable=var, overwrite=False)
    solver.force_update()

    assert solver.get_constraint_count() == 1
    assert solver.get_variable_count() == 1
    assert solver.get_objective_terms_count() == 1


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
@pytest.mark.parametrize(
    "ctype",
    [
        ConstraintType.LESS_THAN_OR_EQUAL,
        ConstraintType.EQUAL,
        ConstraintType.GREATER_THAN_OR_EQUAL,
    ],
)
def test_add_constraint_of_type(solver: AbstractOptimizationSolver, ctype: ConstraintType):
    """
    Testing that adding a constraint of a constraint type is reflected in the solver.
    """
    # Arrange
    var = solver.add_variable(
        lower_bound=0,
        upper_bound=1,
        name="x",
        variable_domain=VariableDomain.CONTINUOUS,
    )
    solver.add_constraint_of_type(
        constraint_type=ctype,
        right_hand_side=1,
        coefficients=1,
        variables=var,
        name="c1",
    )
    solver.force_update()

    # Act & Assert
    assert solver.get_constraint_count() == 1


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_add_multiple_var_and_constr(solver: AbstractOptimizationSolver):
    """
    Testing that adding 3 variables, 3 constraints, and 3 objective terms is reflected in the solver.
    """
    vars_ = solver.add_multiple_variables(
        lower_bound=0,
        upper_bound=1,
        names=np.array(["x_1", "x_2", "x_3"]),
        variable_domain=VariableDomain.CONTINUOUS,
    )
    # Notice that setting both lb and ub may result in 2 constraints in some implementations
    solver.add_multiple_constraints(
        lower_bounds=np.array([0.0, 0.0, 0.0]),
        upper_bounds=None,
        coefficients=np.array(
            [
                np.array([-1.0, 1.0, 1.0]),
                np.array([1.0, -1.0, 1.0]),
                np.array([1.0, 1.0, -1.0]),
            ]
        ),
        variables=np.array([vars_] * 3),
        names=np.array(["c1", "c2", "c3"]),
    )
    solver.add_multiple_objective_terms(coefficients=np.array([1.0, 1.0, 1.0]), variables=vars_, overwrite=False)
    solver.force_update()

    assert len(vars_) == 3
    assert solver.get_variable_count() == 3
    assert solver.get_constraint_count() == 3
    assert solver.get_objective_terms_count() == 3


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_add_multiple_objectives_with_names__named_objectives_is_as_expected(
    solver: AbstractOptimizationSolver,
):
    # Arrange
    vars_ = solver.add_multiple_variables(
        lower_bound=0,
        upper_bound=1,
        names=np.array(["x_1", "x_2", "x_3"]),
        variable_domain=VariableDomain.BINARY,
    )

    objective1 = "Objective1"
    objective2 = "Objective2"

    solver.add_multiple_objective_terms(
        coefficients=np.array([1.0, 1.0, 1.0]),
        variables=vars_,
        overwrite=False,
        name=objective1,
    )
    solver.add_multiple_objective_terms(
        coefficients=np.array([2.0, 2.0, 2.0]),
        variables=vars_,
        overwrite=False,
        name=objective2,
    )
    solver.force_update()
    solver.set_optimization_direction(True)
    solver.solve()

    # Act & Assert
    assert solver.get_named_objective(objective1) == 3.0
    assert solver.get_named_objectives()[objective1] == 3.0
    assert solver.get_named_objective(objective2) == 6.0
    assert solver.get_named_objectives()[objective2] == 6.0
    assert sum(list(solver.get_named_objectives().values())) == solver.get_objective_value()
    assert solver.get_objective_value() == 9.0


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_add_multiple_objectives_with_names__named_objectives_with_no_term_added(
    solver: AbstractOptimizationSolver,
):
    # Arrange
    objective1 = "Objective1"

    solver.add_multiple_objective_terms(
        coefficients=np.array([]),
        variables=np.array([]),
        overwrite=False,
        name=objective1,
    )
    solver.force_update()
    solver.set_optimization_direction(True)
    solver.solve()

    # Act & Assert
    assert solver.get_named_objective(objective1) == 0.0
    assert solver.get_named_objectives()[objective1] == 0.0
    assert sum(list(solver.get_named_objectives().values())) == solver.get_objective_value()
    assert solver.get_objective_value() == 0.0


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_add_objectives_with_names__named_objectives_is_as_expected(
    solver: AbstractOptimizationSolver,
):
    # Arrange
    vars_ = solver.add_multiple_variables(
        lower_bound=0,
        upper_bound=1,
        names=np.array(["x_1", "x_2", "x_3"]),
        variable_domain=VariableDomain.BINARY,
    )

    objective1 = "Objective1"
    objective2 = "Objective2"

    for var in vars_:
        solver.add_objective_term(coefficient=1, variable=var, overwrite=False, name=objective1)
        solver.add_objective_term(coefficient=2, variable=var, overwrite=False, name=objective2)

    solver.force_update()
    solver.set_optimization_direction(True)
    solver.solve()

    # Act & Assert
    assert solver.get_named_objective(objective1) == 3.0
    assert solver.get_named_objective(objective2) == 6.0
    assert solver.get_objective_value() == 9.0


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_add_objectives_with_names__raises_error_when_overwrite(
    solver: AbstractOptimizationSolver,
):
    # Arrange
    vars_ = solver.add_multiple_variables(
        lower_bound=0,
        upper_bound=1,
        names=np.array(["x_1", "x_2", "x_3"]),
        variable_domain=VariableDomain.BINARY,
    )

    # Act & Assert
    with pytest.raises(ValueError) as _:
        solver.add_multiple_objective_terms(
            coefficients=np.array([1.0, 1.0, 1.0]),
            variables=vars_,
            overwrite=True,
            name="Test",
        )


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
@pytest.mark.parametrize("dtype", [VariableDomain.CONTINUOUS, VariableDomain.INTEGER, VariableDomain.BINARY])
@pytest.mark.parametrize("maximisation", [True, False])
def test__solver_functional__optimal_solution(
    solver: AbstractOptimizationSolver, dtype: VariableDomain, maximisation: bool
):
    """
    Testing that the solver returns the known optimal solution, and it is reflected in the optimisation status.
    """
    # Arrange
    solver.set_optimization_direction(maximization=maximisation)
    var = solver.add_variable(lower_bound=0.0, upper_bound=100.0, name="x", variable_domain=dtype)
    solver.add_constraint(
        lower_bound=0.0,
        upper_bound=1.0,
        coefficients=np.array([1.0]),
        variables=np.array([var]),
        name="c1",
    )
    solver.add_objective_term(coefficient=1.0, variable=var, overwrite=False)

    # Act
    solver.solve()

    # Assert
    assert solver.is_optimal()
    assert solver.is_feasible()
    assert not solver.is_abnormal()
    assert not solver.is_unbounded()
    assert not solver.is_infeasible()
    assert solver.get_objective_value() == int(maximisation)
    assert solver.get_variable_value(var) == int(maximisation)
    # assert solver.get_gap() == 0.0  # outcommented because HiGHS does not provide a correct bound


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
@pytest.mark.parametrize("dtype", [VariableDomain.CONTINUOUS, VariableDomain.INTEGER])
def test_solver_functional__unbounded_problem(solver: AbstractOptimizationSolver, dtype: VariableDomain):
    """
    Testing that the solver recognises an unbounded solution, and it is reflected in the optimisation status.
    """
    # Arrange
    solver.set_optimization_direction(maximization=True)
    var = solver.add_variable(lower_bound=0.0, upper_bound=solver.infinity(), name="x", variable_domain=dtype)
    solver.add_objective_term(coefficient=1.0, variable=var, overwrite=False)

    # Act
    solver.solve()

    # Assert
    assert not solver.is_optimal()
    assert not solver.is_abnormal()
    assert solver.is_unbounded()
    assert not solver.is_infeasible()


@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
@pytest.mark.parametrize("dtype", [VariableDomain.CONTINUOUS, VariableDomain.INTEGER])
def test_solver_functional__infeasible_problem(solver: AbstractOptimizationSolver, dtype: VariableDomain):
    """
    Testing that the solver recognises an infeasible solution, and it is reflected in the optimisation status.
    """
    # Arrange
    var = solver.add_variable(lower_bound=0.0, upper_bound=1.0, name="x", variable_domain=dtype)
    solver.add_constraint(
        lower_bound=5.0,
        upper_bound=6.0,
        coefficients=np.array([1.0]),
        variables=np.array([var]),
        name="c1",
    )
    solver.add_objective_term(coefficient=1.0, variable=var, overwrite=False)

    # Act
    solver.solve()

    # Assert
    assert not solver.is_optimal()
    assert not solver.is_abnormal()
    assert not solver.is_unbounded()
    assert solver.is_infeasible()


@pytest.mark.parametrize("solver", ["OrTools", "Scip"], indirect=True)
@pytest.mark.parametrize("verbose", [True, False])
def test__set_verbose(capfd, solver: AbstractOptimizationSolver, verbose: bool):
    """
    Testing that the solver verbosity is set correctly.
    Highs always print at least one line of output, thus, it is not tested here.
    """
    # Arrange
    var = solver.add_variable(
        lower_bound=0.0,
        upper_bound=100.0,
        name="x",
        variable_domain=VariableDomain.CONTINUOUS,
    )
    solver.add_constraint(
        lower_bound=0.0,
        upper_bound=1.0,
        coefficients=np.array([1.0]),
        variables=np.array([var]),
        name="c1",
    )
    solver.add_objective_term(coefficient=1.0, variable=var, overwrite=False)

    # Act
    solver.set_verbose(verbose=verbose)
    solver.solve()

    out, _ = capfd.readouterr()

    # Assert
    if verbose:
        assert len(out) > 0
    else:
        assert len(out) == 0


@pytest.mark.parametrize("solver", ["OrTools", "Highs"], indirect=True)
@pytest.mark.parametrize("overwrite", [True, False])
def test__add_objective_term__overwrite_term_when_desired(solver: AbstractOptimizationSolver, overwrite: bool):
    """
    Testing that the solver setting for OrTools overwrite in objective function is set correctly.
    """
    x = solver.add_variable(lower_bound=0, upper_bound=100, name="x", variable_domain=VariableDomain.INTEGER)

    solver.add_objective_term(coefficient=1.0, variable=x, overwrite=overwrite)
    solver.add_objective_term(coefficient=0.0, variable=x, overwrite=overwrite)

    solver.set_optimization_direction(True)
    solver.set_verbose(True)
    solver.solve()

    obj_value = solver.get_objective_value()

    if overwrite:
        assert obj_value == 0.0
    else:
        assert obj_value == 100.0


@pytest.mark.parametrize("solver", ["OrTools", "Highs"], indirect=True)
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("overwrite", [False, True])
def test__add_objective_offset__overwrite_if_desired(solver: AbstractOptimizationSolver, offset: int, overwrite: bool):
    """
    Testing that offset in objective function is set correctly both when overwriting and adding.
    """
    # Arrange
    solver.set_optimization_direction(maximization=False)
    var = solver.add_variable(
        lower_bound=0.0,
        upper_bound=100.0,
        name="x",
        variable_domain=VariableDomain.CONTINUOUS,
    )
    solver.add_constraint(
        lower_bound=0.0,
        upper_bound=1.0,
        coefficients=np.array([1.0]),
        variables=np.array([var]),
        name="c1",
    )
    solver.add_objective_term(coefficient=1.0, variable=var, overwrite=False)

    # Act
    solver.add_objective_offset(offset=offset, overwrite=overwrite)
    solver.add_objective_offset(offset=offset, overwrite=overwrite)
    solver.solve()

    # Assert
    assert solver.get_objective_value() == offset + offset * (1 - overwrite)


@pytest.mark.parametrize("solver", ["Scip"], indirect=True)
@pytest.mark.parametrize("offset", [1])
def test__add_objective_offset__for_scip__overwrite_false(solver: AbstractOptimizationSolver, offset: int):
    """
    Testing that offset in objective function is set correctly both when overwriting and adding.
    """
    # Arrange
    solver.set_optimization_direction(maximization=False)
    var = solver.add_variable(
        lower_bound=0.0,
        upper_bound=100.0,
        name="x",
        variable_domain=VariableDomain.CONTINUOUS,
    )
    solver.add_constraint(
        lower_bound=0.0,
        upper_bound=1.0,
        coefficients=np.array([1.0]),
        variables=np.array([var]),
        name="c1",
    )
    solver.add_objective_term(coefficient=1.0, variable=var, overwrite=False)

    # Act
    solver.add_objective_offset(offset=offset, overwrite=False)
    solver.add_objective_offset(offset=offset, overwrite=False)
    solver.solve()

    # Assert
    assert solver.get_objective_value() == offset + offset


@pytest.mark.parametrize("solver", ["Highs"], indirect=True)
def test__get_dual_value__lp__retrieved_correctly(solver: AbstractOptimizationSolver):
    """
    Testing that the dual value is retrieved correctly.
    OR Tools does not support dual values, so it is not tested.
    """
    # Arrange
    solver.set_optimization_direction(maximization=True)
    var = solver.add_variable(
        lower_bound=-solver.infinity(),
        upper_bound=solver.infinity(),
        name="x",
        variable_domain=VariableDomain.CONTINUOUS,
    )

    constr = solver.add_constraint(
        lower_bound=-solver.infinity(),
        upper_bound=1.0,
        coefficients=np.array([1.0]),
        variables=np.array([var]),
        name="c1",
    )

    solver.add_objective_term(coefficient=10, variable=var, overwrite=False)
    solver.solve()

    # Act & Assert
    assert solver.get_dual_value(constr) == 10


@pytest.mark.parametrize("solver", ["Highs", "Scip"], indirect=True)
def test__get_dual_value__integer_problem__raises_value_error(
    solver: AbstractOptimizationSolver,
):
    """
    Testing that a value error is raised when trying to retrieve the dual value, as the problem has integer
    decision variables.
    OR Tools does not support dual values, so it is not tested.
    """
    # Arrange
    solver.set_optimization_direction(maximization=True)
    var = solver.add_variable(
        lower_bound=-solver.infinity(),
        upper_bound=solver.infinity(),
        name="x",
        variable_domain=VariableDomain.INTEGER,
    )
    constr = solver.add_constraint(
        lower_bound=-solver.infinity(),
        upper_bound=1.0,
        coefficients=np.array([1.0]),
        variables=np.array([var]),
        name="c1",
    )
    solver.add_objective_term(coefficient=10, variable=var, overwrite=False)
    solver.solve()

    # Act & Assert
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
        solver.add_variable(
            lower_bound=0,
            upper_bound=0,
            variable_domain=VariableDomain.CONTINUOUS,
            name="C[" + str(i) + "]",
        )
        for i in range(number_of_continuous_variables)
    ]
    [
        solver.add_variable(
            lower_bound=0,
            upper_bound=0,
            variable_domain=VariableDomain.BINARY,
            name="B[" + str(i) + "]",
        )
        for i in range(number_of_binary_variables)
    ]
    [
        solver.add_variable(
            lower_bound=0,
            upper_bound=0,
            variable_domain=VariableDomain.INTEGER,
            name="I[" + str(i) + "]",
        )
        for i in range(number_of_integer_variables)
    ]

    # Act & Assert
    assert solver.get_variable_count_of_type(VariableDomain.CONTINUOUS) == number_of_continuous_variables
    assert solver.get_variable_count_of_type(VariableDomain.BINARY) == number_of_binary_variables
    assert solver.get_variable_count_of_type(VariableDomain.INTEGER) == number_of_integer_variables
    assert solver.get_variable_count() == total_variables_count


@pytest.mark.parametrize("solver", ["OrTools", "Scip"], indirect=True)
def test__get_gap__time_limit_reached__gap_greater_than_zero(
    solver: AbstractOptimizationSolver,
):
    """
    The travelling salesman problem (TSP) is trying to find the most cost-efficient route for the salesman to visit
    each city exactly once and returns to the origin city. It is an NP-hard problem in combinatorial optimization.
    This test considers 100 cities and travel costs between cities are assigned randomly within (0,1), the
    randomness is controlled by seed. The goal of the test is to solve a computationally difficult TSP and stop at
    the time limit, resulting in a gap greater than zero.
    """
    # Arrange
    number_of_cities = 200
    cities = list(range(number_of_cities))

    c = [[10.0 for _ in cities] for _ in cities]
    x = [
        [
            solver.add_variable(
                lower_bound=0.0,
                upper_bound=1.0,
                name=f"x{i},{j}",
                variable_domain=VariableDomain.BINARY,
            )
            for j in cities
        ]
        for i in cities
    ]
    u = [None] + [
        solver.add_variable(
            lower_bound=1.0,
            upper_bound=number_of_cities - 1,
            name=f"u{i}",
            variable_domain=VariableDomain.INTEGER,
        )
        for i in cities[1:]
    ]

    for i in cities:
        solver.add_multiple_objective_terms(coefficients=np.array(c[i]), variables=np.array(x[i]), overwrite=False)

    for i in cities:
        solver.add_constraint(
            lower_bound=1.0,
            upper_bound=1.0,
            coefficients=np.array([1.0 for j in cities if i != j]),
            variables=np.array([x[i][j] for j in cities if i != j]),
        )

    for j in cities:
        solver.add_constraint(
            lower_bound=1.0,
            upper_bound=1.0,
            coefficients=np.array([1.0 for i in cities if i != j]),
            variables=np.array([x[i][j] for i in cities if i != j]),
        )

    for i in cities[1:]:
        for j in cities[1:]:
            if i != j:
                solver.add_constraint(
                    lower_bound=None,
                    upper_bound=number_of_cities - 2,
                    coefficients=np.array([1.0, -1.0, number_of_cities - 1]),
                    variables=np.array([u[i], u[j], x[i][j]]),
                )
    if type(solver).__name__ == "OrToolsSolver":
        solver.set_solver_setting(
            setting=OrToolsScipSolverConfig(presolving_max_rounds=0),
        )
    if type(solver).__name__ == "ScipSolver":
        solver.set_param("presolving/maxrounds", 0)
    solver.set_verbose(verbose=True)

    # Act
    solver.solve(time_limit=1)
    gap = solver.get_gap()

    # Assert
    assert gap is not None and gap > 0


@pytest.mark.parametrize("solver", ["Highs", "OrTools", "Scip"], indirect=True)
def test_get_gap__gap_limit_reached__gap_equal_to_limit(
    solver: AbstractOptimizationSolver,
):
    """
    The travelling salesman problem (TSP) is trying to find the most cost-efficient route for the salesman to visit
    each city exactly once and returns to the origin city. It is an NP-hard problem in combinatorial optimization.
    This test considers 100 cities and travel costs between cities are assigned randomly within (0,1), the
    randomness is controlled by seed. The goal of the test is to solve a computationally difficult TSP and stop at
    the mip gap limit while providing a gap that is equal or lower than mip_gap_limit.
    """
    # Arrange
    number_of_cities = 100
    cities = list(range(number_of_cities))

    np.random.seed(number_of_cities)
    c = [[np.random.rand() for _ in cities] for _ in cities]
    x = [
        [
            solver.add_variable(
                lower_bound=0.0,
                upper_bound=1.0,
                name=f"x{i},{j}",
                variable_domain=VariableDomain.BINARY,
            )
            for j in cities
        ]
        for i in cities
    ]
    u = [None] + [
        solver.add_variable(
            lower_bound=1.0,
            upper_bound=number_of_cities - 1,
            name=f"u{i}",
            variable_domain=VariableDomain.INTEGER,
        )
        for i in cities[1:]
    ]

    for i in cities:
        for j in cities:
            if i != j:
                solver.add_objective_term(coefficient=c[i][j], variable=x[i][j], overwrite=False)

    for i in cities:
        solver.add_constraint(
            lower_bound=1.0,
            upper_bound=1.0,
            coefficients=np.array([1.0 for j in cities if i != j]),
            variables=np.array([x[i][j] for j in cities if i != j]),
        )

    for j in cities:
        solver.add_constraint(
            lower_bound=1.0,
            upper_bound=1.0,
            coefficients=np.array([1.0 for i in cities if i != j]),
            variables=np.array([x[i][j] for i in cities if i != j]),
        )

    for i in cities[1:]:
        for j in cities[1:]:
            if i != j:
                solver.add_constraint(
                    lower_bound=None,
                    upper_bound=number_of_cities - 2,
                    coefficients=np.array([1.0, -1.0, number_of_cities - 1]),
                    variables=np.array([u[i], u[j], x[i][j]]),
                )

    solver.set_verbose(verbose=True)
    solver.solve(mip_gap_limit=0.7)

    # Act
    gap = solver.get_gap()

    # Assert
    assert gap is not None and gap <= 0.7


# this fails sometimes because Highs is stupid, just rerun
@pytest.mark.parametrize("solver", ["OrTools", "Highs", "Scip"], indirect=True)
def test_objective_terms_with_low_coefficient__expected_objective_analytics_to_be_equal(
    solver: AbstractOptimizationSolver,
):
    # Arrange
    vars_ = solver.add_multiple_variables(
        lower_bound=1,
        upper_bound=10,
        names=np.array(["x_1", "x_2", "x_3", "x_4", "x_5"]),
        variable_domain=VariableDomain.INTEGER,
    )

    objective_name = "Objective1"

    solver.add_multiple_objective_terms(
        coefficients=np.array([1.0, 1e-6, 1e-9, -1e-6, -1e-9]),
        variables=vars_,
        overwrite=False,
        name=objective_name,
    )
    solver.force_update()
    solver.set_optimization_direction(False)
    solver.solve()

    # Act & Assert
    assert sum(list(solver.get_named_objectives().values())) == solver.get_objective_value()
