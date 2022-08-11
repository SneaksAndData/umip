"""Abstract definition of a solver."""
from abc import ABC, abstractmethod, ABCMeta
from typing import Optional, Union
import numpy.typing as npt
from generic_mip.variable_data_type import VariableDataType


class AbstractOptimizationSolver(ABC, metaclass=ABCMeta):  # pylint: disable=too-many-public-methods
    """
    The optimization solver contains all optimization related states, variables, constraints and objectives.
    This class serves as a generic API for various implementations through solver specific APIs.
    """
    @abstractmethod
    def add_constraint(
        self,
        lb: Optional[float],
        ub: Optional[float],
        coeffs: Union[npt.NDArray[float], float],
        vars_:  Union[npt.NDArray[any], any],
        name: Optional[str] = None
    ) -> any:
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
    def get_constraint(self, name: str) -> any:
        """
        Retrieves a constraint by name.
        :param name: Name of the constraint.
        :return: The constraint.
        """

    @abstractmethod
    def add_multiple_constraints(
        self,
        lb: Optional[npt.NDArray[float]],
        ub: Optional[npt.NDArray[float]],
        coeffs: Union[npt.NDArray[npt.NDArray[float]], npt.NDArray[float]],
        vars_: Union[npt.NDArray[npt.NDArray[any]], npt.NDArray[any]],
        name: Optional[str] = None
    ) -> any:
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
        :param name: Name of the constraints
        """

    @abstractmethod
    def add_variable(self, lb: float, ub: float, name: str, dtype: VariableDataType) -> any:
        """
        Adds variable to the model.

        :param lb: Lower bound.
        :param ub: Upper bound.
        :param name: Name of the variable.
        :param dtype: The type of variable.
        :return: The variable.
        """

    @abstractmethod
    def add_multiple_variables(self, count: int, lb: float, ub: float, name: str, dtype: VariableDataType) -> any:
        """
        Adds multiple variables to the model.

        :param count: Number of variables to create.
        :param lb: Lower bound.
        :param ub: Upper bound.
        :param name: Name of the variables.
        :param dtype: The type of variables.
        :return: The variables.
        """

    @abstractmethod
    def set_variable_hint(self, var: any, hint: float) -> None:
        """
        Adds solution hint to decision variable.

        :param var: Variable to add hint to.
        :param hint: Solution hint.
        :return:
        """

    @abstractmethod
    def set_multiple_variable_hints(self, vars_: npt.NDArray[any], hints: npt.NDArray[float]) -> None:
        """
        Adds solution hints to multiple decision variables. Number of variables and hints must be identical.

        :param vars_: Variables to add hint to.
        :param hints: Solution hints.
        :return:
        """

    @abstractmethod
    def add_objective_term(self, coeff: float, var: any) -> None:
        """
        Adds a single objective term.

        :param coeff: The coefficient of the term.
        :param var: The variable of the term.
        :return:
        """

    @abstractmethod
    def add_multiple_objective_terms(self, coeffs: npt.NDArray[float], vars_: npt.NDArray[any]) -> None:
        """
        Adds multiple objective terms at once: c_1x_1 + c_2x_2 + ...

        :param coeffs: The coefficients of the term. Example: [c_1, c_2, ...].
            Each index must match an index in the vars_ list.
        :param vars_: The variables of the term. Example: [x_1, x_2, ...].
            Each index must match an index in the coeffs list.
        :return:
        """

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
    def solve(self) -> any:
        """
        Solve the optimization problem.
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
        Whether the model is solved to optimality. Only callable after calling solve().
        :return: Whether the model is solved to optimality.
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
    def get_variable_value(self, var: any) -> float:
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

    def set_solver_setting(self, setting: str):
        """
        Set solver setting.
        :param setting: The setting to set.
        :return:
        """

    @abstractmethod
    def get_variable_count(self):
        """
        Get number of variables in the model.
        :return: Number of variables.
        """

    @abstractmethod
    def get_constraint_count(self):
        """
        Get number of constraints in the model.
        :return: Number of constraints.
        """

    @abstractmethod
    def get_objective_terms_count(self):
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
