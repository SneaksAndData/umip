"""Abstract definition of a solver."""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import numpy as np
import numpy.typing as npt
from adapta.logs import LoggerInterface
from generic_mip.enums.constraint_type import ConstraintType
from generic_mip.enums.variable_data_type import VariableDataType
from generic_mip.variable_with_objective_coefficient import VariableWithObjectiveCoefficient

CT = TypeVar("CT")  # Constraint type
VT = TypeVar("VT")  # Variable type


class AbstractOptimizationSolver(ABC, Generic[VT, CT]):  # pylint: disable=too-many-public-methods
    """
    The optimization solver contains all optimization related states, variables, constraints and objectives.
    This class serves as a generic API for various implementations through solver specific APIs.
    """

    def __init__(self, logger: LoggerInterface):
        """
        Initialize the solver.
        :param logger: The logger to use. Notice that most solvers call C-based libraries that do not use the logger.
            If these libraries produce output, it will need to be caught by redirecting their logs.
            Log redirect is not performed in this class due to the overhead it would add to each method call.
        """
        self._logger = logger
        self._named_objectives = {}
        self._integer_problem = False

    @abstractmethod
    def add_constraint(
        self,
        coeffs: npt.NDArray[float] | float,
        vars_: npt.NDArray[VT] | VT,
        lb: float | None = None,
        ub: float | None = None,
        name: str | None = None,
    ) -> CT | None:
        """
        Add a single constraint:
        lb <= c_1x_1 + c_2x_2 + ... <= ub
        or
        lb <= cx <= ub

        :param lb: Lower bound.
        :param ub: Upper bound.
        :param coeffs: List of coefficients: [c_1, c_2, ...] or a single coefficient: c.
            Each index in the list must correspond to the same index in the vars_ list.
        :param vars_: List of variables: [x_1, x_2, ...] or a single variable: x.
            Each index in the list must correspond to the same index in the coeffs list.
        :param name: Name of constraint.
        :return: The constraint.
        """

    @abstractmethod
    def get_constraint(self, name: str) -> CT:
        """
        Retrieves a constraint by name.
        :param name: Name of the constraint.
        :return: The constraint.
        """

    @abstractmethod
    def add_multiple_constraints(
        self,
        coeffs: npt.NDArray[npt.NDArray[float]] | npt.NDArray[float],
        vars_: npt.NDArray[npt.NDArray[VT]] | npt.NDArray[VT],
        lb: npt.NDArray[float] | None = None,
        ub: npt.NDArray[float] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        """
        Adds multiple constraints at once, such that:
            lb_1 <= c_11x_11 + c_12x_12 + ... <= ub_1
            lb_2 <= c_21x_21 + c_22x_22 + ... <= ub_2
            ...
        or such that:
            lb_1 <= c_1x_1 <= ub_1
            lb_2 <= c_2x_2 <= ub_2
            ...

        :param lb: A list of lower bounds, one for each constraint.
        :param ub: A list of upper bounds, one for each constraint.
        :param coeffs: A list of lists of constraint coefficients. Each inner list represent a constraint,
            and each element of the inner list is a coefficient of a variable in the constraint.
            Example input: [[c_11,c_12,...],[c_21,c_22,...],...] or [c_1, c_2, ...]
        :param vars_: A list of lists of constraint coefficients. Each inner list represent a constraint,
            and each element of the inner list is a variable in the constraint. The dimensions must exactly
            match those of the parameter coeffs. Example: The coefficient in index [1,2] must belong to the variable in
            index [1,2].
            Example input: [[x_11,x_12,...],[x_21,x_22,...],...] or [x_1, x_2, ...]
        :param names: Names of the constraints
        """

    def add_constraint_of_type(
        self,
        constraint_type: ConstraintType,
        coeffs: npt.NDArray[float] | float,
        vars_: npt.NDArray[VT] | VT,
        right_hand_side: float | None = None,
        name: str | None = None,
    ) -> CT | None:
        """
        Add a single constraint of the form
            cx ? ub

        :param constraint_type: The relation to be used in the constraint. (Either >=, = or <=)
        :param right_hand_side: The right hand side of the constraint.
        :param coeffs: List of coefficients: [c_1, c_2, ...] or a single coefficient: c.
            Each index in the list must correspond to the same index in the vars_ list.
        :param vars_: List of variables: [x_1, x_2, ...] or a single variable: x.
            Each index in the list must correspond to the same index in the coeffs list.
        :param name: Name of constraint.
        :return: The constraint.
        """
        if constraint_type == ConstraintType.LESS_THAN_OR_EQUAL:
            return self.add_constraint(coeffs=coeffs, vars_=vars_, ub=right_hand_side, name=name)
        if constraint_type == ConstraintType.EQUAL:
            return self.add_constraint(coeffs=coeffs, vars_=vars_, lb=right_hand_side, ub=right_hand_side, name=name)
        if constraint_type == ConstraintType.GREATER_THAN_OR_EQUAL:
            return self.add_constraint(coeffs=coeffs, vars_=vars_, lb=right_hand_side, name=name)

        raise ValueError(f"Unsupported constraint type: {constraint_type}")

    def add_multiple_constraints_of_type(
        self,
        constraint_type: ConstraintType,
        coeffs: npt.NDArray[npt.NDArray[float]] | npt.NDArray[float],
        vars_: npt.NDArray[npt.NDArray[VT]] | npt.NDArray[VT],
        right_hand_sides: npt.NDArray[float] | None = None,
        names: npt.NDArray[str] | None = None,
    ) -> None:
        """
        Adds multiple constraints of the form
            c_1x_1 ? ub_1
            c_2x_2 ? ub_2
            ...

        :param constraint_type: The relation to be used in the constraint. (Either >=, = or <=)
        :param right_hand_sides: The right hand side sof the constraints.
        :param coeffs: A list of lists of constraint coefficients. Each inner list represent a constraint,
            and each element of the inner list is a coefficient of a variable in the constraint.
            Example input: [[c_11,c_12,...],[c_21,c_22,...],...] or [c_1, c_2, ...]
        :param vars_: A list of lists of constraint coefficients. Each inner list represent a constraint,
            and each element of the inner list is a variable in the constraint. The dimensions must exactly
            match those of the parameter coeffs. Example: The coefficient in index [1,2] must belong to the variable in
            index [1,2].
            Example input: [[x_11,x_12,...],[x_21,x_22,...],...] or [x_1, x_2, ...]
        :param names: Names of the constraints
        """
        if constraint_type == ConstraintType.LESS_THAN_OR_EQUAL:
            return self.add_multiple_constraints(coeffs=coeffs, vars_=vars_, ub=right_hand_sides, names=names)
        if constraint_type == ConstraintType.EQUAL:
            return self.add_multiple_constraints(
                coeffs=coeffs, vars_=vars_, lb=right_hand_sides, ub=right_hand_sides, names=names
            )
        if constraint_type == ConstraintType.GREATER_THAN_OR_EQUAL:
            return self.add_multiple_constraints(coeffs=coeffs, vars_=vars_, lb=right_hand_sides, names=names)

        raise ValueError(f"Unsupported constraint type: {constraint_type}")

    @abstractmethod
    def add_variable(self, name: str, dtype: VariableDataType, lb: float | None = None, ub: float | None = None) -> VT:
        """
        Adds variable to the model.

        :param lb: Lower bound.
        :param ub: Upper bound.
        :param name: Name of the variable.
        :param dtype: The type of variable.
        :return: The variable.
        """

    @abstractmethod
    def add_multiple_variables(
        self,
        names: npt.NDArray[str],
        dtype: VariableDataType,
        lb: float | None = None,
        ub: float | None = None,
    ) -> npt.NDArray[VT]:
        """
        Adds multiple variables to the model.

        :param names: Names of the variables.
        :param lb: Lower bound.
        :param ub: Upper bound.
        :param dtype: The type of variables.
        :return: The variables.
        """

    @abstractmethod
    def set_variable_hint(self, var: VT, hint: float) -> None:
        """
        Adds solution hint to decision variable.

        :param var: Variable to add hint to.
        :param hint: Solution hint.
        :return:
        """

    @abstractmethod
    def set_multiple_variable_hints(self, vars_: npt.NDArray[VT], hints: npt.NDArray[float]) -> None:
        """
        Adds solution hints to multiple decision variables. Number of variables and hints must be identical.

        :param vars_: Variables to add hint to.
        :param hints: Solution hints.
        :return:
        """

    @abstractmethod
    def add_objective_term(self, coeff: float, var: VT, overwrite: bool = True, name: str = None) -> None:
        """
        Adds a single objective term.

        :param coeff: The coefficient of the term.
        :param var: The variable of the term.
        :return:
        """

    @abstractmethod
    def add_multiple_objective_terms(
        self, coeffs: npt.NDArray[float], vars_: npt.NDArray[VT], overwrite: bool = True, name: str = None
    ) -> None:
        """
        Adds multiple objective terms at once: c_1x_1 + c_2x_2 + ...

        :param coeffs: The coefficients of the term. Example: [c_1, c_2, ...].
            Each index must match an index in the vars_ list.
        :param vars_: The variables of the term. Example: [x_1, x_2, ...].
            Each index must match an index in the coeffs list.
        :return:
        """

    def add_named_objective(
        self, coeffs: npt.NDArray[float], vars_: npt.NDArray[VT], overwrite: bool, name: str
    ) -> None:
        """
        Add/update the coefficients and variables of a named objective term

        :param coeffs: The coefficients of the term. Example: [c_1, c_2, ...].
            Each index must match an index in the vars_ list.
        :param vars_: The variables of the term. Example: [x_1, x_2, ...].
            Each index must match an index in the coeffs list.
        :param overwrite: Whether to overwrite the coefficients or not.
        :param name: The name of the objective
        :return:
        """
        if overwrite:
            raise ValueError("Add_named_objective is not supported with overwrite = true")

        elements_to_add = [VariableWithObjectiveCoefficient(vars_[i], coeffs[i]) for i in range(len(coeffs))]
        if name in self._named_objectives:
            self._named_objectives[name].extend(elements_to_add)
        else:
            self._named_objectives[name] = elements_to_add

    def get_named_objective(self, name: str) -> float:
        """
        Get the value of a named objective term

        :param name: The name of the objective
        :return: objective value for the named objective.
        """
        return np.sum(
            [
                (item.objective_coefficient * self.get_variable_value(item.variable))
                for item in self._named_objectives[name]
            ]
        )

    @abstractmethod
    def set_optimization_direction(self, maximization: bool) -> None:
        """
        Sets the direction of the optimization.

        :param maximization: If True, set as maximization problem. If False, set as minimization problem.
        :return:
        """

    @abstractmethod
    def get_objective_value(self) -> float:
        """
        Get objective value of the optimization.
        :return: The objective value.
        """

    @abstractmethod
    def solve(self, time_limit: float | None = None, mip_gap_limit: float | None = None) -> int:
        """
        Solve the optimization problem. If both time_limit and mip_gap_limit are provided, the optimization would stop
        when reach the tighter limit.
        :param time_limit: The time limit in seconds. None means no time limit.
        :param mip_gap_limit: The optimality gap in percent, and in format of float. None means no mip gap limit.
        :return: The status of the optimization.
        """

    @abstractmethod
    def infinity(self) -> float:
        """
        Returns the solver specific implementation of infinity.
        :return: Infinity.
        """

    @abstractmethod
    def is_optimal(self) -> bool:
        """
        Whether the model is solved to optimality.
        :return: Whether the model is solved to optimality.
        """

    @abstractmethod
    def is_feasible(self) -> bool:
        """
        Whether the model has found a feasible but not the optimal solution.
        :return: Whether the model is solved to a feasible solution.
        """

    @abstractmethod
    def is_infeasible(self) -> bool:
        """
        Whether the model is infeasible. Only callable after calling solve().
        :return: Whether the model is infeasible.
        """

    @abstractmethod
    def is_abnormal(self) -> bool:
        """
        Whether the model is abnormal. Only callable after calling solve().
        :return: Whether the model is abnormal.
        """

    @abstractmethod
    def is_unbounded(self) -> bool:
        """
        Whether the model is unbounded. Only callable after calling solve().
        :return: Whether the model is unbounded.
        """

    @abstractmethod
    def is_not_solved(self) -> bool:
        """
        Whether the model is not finished solving. Only callable after calling solve().
        :return: Whether the model is not finished solving.
        """

    @abstractmethod
    def get_variable_value(self, var: VT) -> float:
        """
        Get value of a decision variable.
        :param var: Variable to get value from.
        :return: The value.
        """

    @abstractmethod
    def export_to_file(self, path: str) -> None:
        """
        Export model to file.
        :param path: The file path to export to.
        :return:
        """

    @abstractmethod
    def set_verbose(self, verbose: bool) -> None:
        """
        Set verbose mode of the solving process.
        :param verbose: Whether to enable verbose mode or not.
        :return:
        """

    def set_solver_setting(self, setting: str) -> None:
        """
        Set solver setting.
        :param setting: The setting to set.
        :return:
        """

    @abstractmethod
    def get_variable_count(self) -> int:
        """
        Get number of variables in the model.
        :return: Number of variables.
        """

    @abstractmethod
    def get_variable_count_of_type(self, var_type: VariableDataType) -> int:
        """
        Get number of variables of the specified type in the model.
        :param var_type: The variable type.
        :return: Number of variables of the provided type.
        """

    @abstractmethod
    def get_constraint_count(self) -> int:
        """
        Get number of constraints in the model.
        :return: Number of constraints.
        """

    @abstractmethod
    def get_objective_terms_count(self) -> int:
        """
        Get number of objective terms in the model.
        :return: Number of objective terms.
        """

    @abstractmethod
    def force_update(self):
        """
        Some models add variables, constraints and objective lazily.
        Use this method to force an update of the model, if the specific implementation uses lazy updates.
        E.g. Gurobi uses lazy updates.
        :return:
        """

    @abstractmethod
    def get_gap(self) -> float:
        """
        Returns the gap of the solution.
        :return: The gap.
        """

    @abstractmethod
    def add_objective_offset(self, offset: float, overwrite: bool = True):
        """
        Set the objective offset.
        :param offset: The offset to set.
        :param overwrite: Whether to overwrite the current offset or add to it.
        :return
        """

    @abstractmethod
    def get_dual_value(self, constraint: CT) -> float:
        """
        Get dual value of a constraint.
        :param constraint: Constraint to get dual value from.
        :return: The dual value.
        """
