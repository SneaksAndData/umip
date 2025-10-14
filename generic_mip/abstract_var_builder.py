"""Abstract definition of a variable builder."""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from adapta.logs import LoggerInterface
import pandas as pd
import polars as pl
import numpy as np
from generic_mip.enums.variable_data_type import VariableDataType
from generic_mip.abstract_solver import AbstractOptimizationSolver

T = TypeVar("T")
VT = TypeVar("VT")  # Variable type


class AbstractDecisionVariableBuilder(ABC, Generic[T]):
    """A variable builder has the responsibility of building one or more decision variables."""

    def __init__(self, logger: LoggerInterface):
        """
        Initialize the variable builder.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, data: dict[str, T]) -> dict[str, T]:
        """
        Builds the decision variables on the given model and the given data.

        :param solver: The solver to use to build the variables.
        :param data: The data (e.g. dataframes) providing parameters for the variables.
        :return: The dataframes decorated with the created decision variables.
        """

    @abstractmethod
    def unpack(self, solver: AbstractOptimizationSolver, data: dict[str, T]) -> dict[str, T]:
        """
        Unpacks the decision variables after optimization and inserts variable values in the dataframes.

        :param solver: The solver to get the variable values from.
        :param data: The data (e.g. dataframes) containing the variables.
        :return: The dataframes decorated with the values of the decision variables.
        """

    def _build_column_variables_pandas(
        self,
        solver: AbstractOptimizationSolver,
        data: pd.DataFrame,
        destination_column: str,
        variable_dtype: VariableDataType,
        lb: np.array,
        ub: np.array,
        names: np.array,
        indicators: np.array,
    ) -> pd.DataFrame:
        """
        Builds decision variables on a pandas DataFrame by applying a solver's add_variable method and add this to a
        new column in the dataframe.

        :param solver: The solver object used to retrieve the variable values.
        :param data: The DataFrame containing the data.
        :param destination_column: The name of the column that should contain the decision variables.
        :param variable_dtype: The type of variable (int, bool, float).
        :param lb: The lower bound values.
        :param ub: The upper bound values.
        :param names: The names of the variables.
        :param indicators: The indicator for whether a variables should be created or not.
        :return: The updated pandas DataFrame with the decision variable values in the destination_column.
        """

        if data.empty:
            # if dataframe is empty, we should not create any variables
            return data.assign(**{destination_column: None})

        data = data.assign(
            **{
                "lb": lb,
                "ub": ub,
                "var_name": names,
                "indicators": indicators,
            }
        )

        data["var_information"] = data[["lb", "ub", "var_name", "indicators"]].values.tolist()

        # Building variables, where y[0] is lb, y[1] is ub, y[2] is var_name and y[3] is indicators
        data = data.assign(
            **{
                destination_column: lambda x: x["var_information"].apply(
                    lambda y: solver.add_variable(lb=y[0], ub=y[1], name=y[2], dtype=variable_dtype) if y[3] else None
                ),
            }
        )

        data = data.drop(columns=["lb", "ub", "var_name", "indicators", "var_information"])

        return data

    def _build_column_variables_polars(
        self,
        solver: AbstractOptimizationSolver,
        data: pl.DataFrame,
        destination_column: str,
        variable_dtype: VariableDataType,
        lb: np.array,
        ub: np.array,
        names: np.array,
        indicators: np.array,
    ) -> pl.DataFrame:
        """
        Builds decision variables on a polars DataFrame by applying a solver's add_variable method and add this to a
        new column in the dataframe.

        :param solver: The solver object used to retrieve the variable values.
        :param data: The DataFrame containing the data.
        :param destination_column: The name of the column that should contain the decision variables.
        :param variable_dtype: The type of variable (int, bool, float).
        :param lb: The lower bound values.
        :param ub: The upper bound values.
        :param names: The names of the variables.
        :param indicators: The indicator for whether a variables should be created or not.
        :return: The updated polars DataFrame with the decision variable values in the destination_column.
        """

        if data.is_empty():
            # if dataframe is empty, we should not create any variables
            return data.with_columns(pl.lit(None).alias(destination_column))

        def build_variables(row: dict) -> VT:
            return solver.add_variable(lb=row["lb"], ub=row["ub"], name=row["var_name"], dtype=variable_dtype)

        data = data.with_columns(
            **{
                "lb": lb,
                "ub": ub,
                "var_name": names,
                "indicators": indicators,
            }
        )

        # Building variables:
        data = data.with_columns(
            pl.when(pl.col("indicators"))
            .then(pl.struct(["lb", "ub", "var_name"]).map_elements(build_variables, return_dtype=pl.datatypes.Object))
            .otherwise(None)
            .alias(destination_column)
        )

        data = data.drop(["lb", "ub", "var_name", "indicators"])

        return data

    def build_column_variables(
        self,
        solver: AbstractOptimizationSolver,
        data: pd.DataFrame | pl.DataFrame,
        destination_column: str,
        variable_dtype: VariableDataType,
        index_name_columns: list[str] | None = None,
        lower_bound: float | str | None = None,
        upper_bound: float | str | None = None,
        var_name: str | None = None,
        filter_column: str | None = None,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Builds decision variables from a DataFrame by applying a solver's add_variable method and add this to a new
        column in the dataframe.

        :param solver: The solver object used to retrieve the variable values.
        :param data: The DataFrame containing the data.
        :param destination_column: The name of the column that should contain the decision variables.
        :param variable_dtype: The type of variable (int, bool, float).
        :param index_name_columns: list of column names for indexing the variable name (optional, default naming will
        be after row number).
        :param lower_bound: The lower bound of the variable. If float, the lower bound will be assigned to all
        variables. If str, the column will be used as lower bound values (optional, default is -inf).
        both lower_bound and lower_bound_column is defined, we use the lower_bound_column values.
        :param upper_bound: The upper bound of the variable. If float, the upper bound will be assigned to all
        variables. If str, the column will be used as upper bound values (optional, default is inf).
        both upper_bound and upper_bound_column is defined, we use the upper_bound_column values.
        :param var_name: The name of the variable (optional, default is destination_column).
        :param filter_column: The name of the column with True/False values to filter the rows (optional, default is
        creating variables for all rows).
        :return: The updated DataFrame with the decision variable values in the destination_column.
        """
        # Input checks:
        if any(column in data.columns for column in ["lb", "ub", "var_name", "indicators", "var_information"]):
            raise ValueError(
                "DataFrame must not contain column names equal to 'lb', 'ub', 'var_name', "
                "'indicators' and 'var_information'"
            )

        # Check if DataFrame is Pandas or Polars:
        is_pandas = isinstance(data, pd.DataFrame)
        is_polars = isinstance(data, pl.DataFrame)

        # Setup var name prefix:
        var_name_prefix = self._get_var_name_prefix(var_name, destination_column)

        # Find upper and lower bounds:
        lb = self._get_bound(lower_bound, solver, data, variable_dtype, bound_type="lower")
        ub = self._get_bound(upper_bound, solver, data, variable_dtype, bound_type="upper")

        # Make indication for if row should have a variable:
        indicators = self._get_indicators(filter_column, data)

        # Create names for variables:
        names = self._get_names(index_name_columns, data, var_name_prefix)

        # Make columns in dataframe:
        if is_pandas:
            data = self._build_column_variables_pandas(
                solver=solver,
                data=data,
                destination_column=destination_column,
                variable_dtype=variable_dtype,
                lb=lb,
                ub=ub,
                names=names,
                indicators=indicators,
            )

            return data

        if is_polars:
            data = self._build_column_variables_polars(
                solver=solver,
                data=data,
                destination_column=destination_column,
                variable_dtype=variable_dtype,
                lb=lb,
                ub=ub,
                names=names,
                indicators=indicators,
            )

            return data

        raise ValueError(f"No method for building variables in DataFrame {type(data)} type is defined.")

    def _unpack_column_variables_pandas(
        self,
        data: pd.DataFrame,
        decision_variable_column: str,
        decision_variable_value_column: str,
        solver: AbstractOptimizationSolver,
        indicators: np.array,
        default_unpack_value: float,
        return_dtype: VariableDataType,
    ) -> pd.DataFrame:
        """
        Unpacks decision variables from a pandas DataFrame by applying a solver's get_variable_value method and removes
        the decision variable object column from the DataFrame.

        :param data: The DataFrame containing the decision variables.
        :param decision_variable_column: The name of the column containing the decision variables.
        :param decision_variable_value_column: The name of the column to store the unpacked decision variable values.
        :param solver: The solver object used to retrieve the variable values.
        :param indicators: The indicator for if a variables should be unpacked or not. If not, the value is set
        to default_unpack_value.
        :param default_unpack_value: The default value to use if a variable does not exist.
        :param return_dtype: return VariableDataType to cast decision_variable_value_column
        to (optional, default is float).
        :return: The updated pandas DataFrame with the unpacked decision variable values.
        """

        if data.empty:
            # if dataframe is empty, we should not create any variables
            return data.assign(**{decision_variable_value_column: None}).drop(columns=[decision_variable_column])

        data = data.assign(
            **{
                "indicators": indicators,
            }
        )

        data["var_information"] = data[[decision_variable_column, "indicators"]].values.tolist()

        data = data.assign(
            **{
                decision_variable_value_column: lambda x: x["var_information"].apply(
                    lambda y: solver.get_variable_value(y[0]) if y[1] else default_unpack_value
                )
            }
        ).drop(columns=[decision_variable_column, "indicators", "var_information"])

        if return_dtype == VariableDataType.FLOAT:
            return data

        if return_dtype == VariableDataType.BOOL:
            data[decision_variable_value_column] = data[decision_variable_value_column].round(decimals=0).astype("bool")
            return data

        if return_dtype == VariableDataType.INT:
            data[decision_variable_value_column] = data[decision_variable_value_column].round(decimals=0).astype("int")
            return data

        raise ValueError(f"Unsupported return_dtype {return_dtype}")

    def _unpack_column_variables_polars(
        self,
        data: pl.DataFrame,
        decision_variable_column: str,
        decision_variable_value_column: str,
        solver: AbstractOptimizationSolver,
        indicators: np.array,
        default_unpack_value: float,
        return_dtype: VariableDataType,
    ) -> pl.DataFrame:
        """
        Unpacks decision variables from a polars DataFrame by applying a solver's get_variable_value method and removes
        the decision variable object column from the DataFrame.

        :param data: The DataFrame containing the decision variables.
        :param decision_variable_column: The name of the column containing the decision variables.
        :param decision_variable_value_column: The name of the column to store the unpacked decision variable values.
        :param solver: The solver object used to retrieve the variable values.
        :param indicators: The indicator for if a variables should be unpacked or not. If not, the value is set
        to default_unpack_value.
        :param default_unpack_value: The default value to use if a variable does not exist.
        :param return_dtype: return VariableDataType to cast decision_variable_value_column
        to (optional, default is float).
        :return: The updated polars DataFrame with the unpacked decision variable values.
        """

        if data.is_empty():
            # if dataframe is empty, we should not create any variables
            return data.with_columns(pl.lit(None).alias(decision_variable_value_column)).drop(decision_variable_column)

        data = data.with_columns(
            **{
                "indicators": indicators,
            }
        )

        if data.filter(pl.col("indicators")).is_empty():
            data = data.with_columns(pl.lit(default_unpack_value).alias(decision_variable_value_column))
        else:
            data = data.with_columns(
                pl.when(pl.col("indicators"))
                .then(pl.col(decision_variable_column).map_elements(solver.get_variable_value))
                .otherwise(default_unpack_value)
                .alias(decision_variable_value_column)
            )

        data = data.drop([decision_variable_column, "indicators"])

        if return_dtype == VariableDataType.FLOAT:
            return data

        if return_dtype == VariableDataType.BOOL:
            data = data.with_columns(
                pl.col(decision_variable_value_column).round(decimals=0).cast(pl.datatypes.Boolean)
            )

            return data

        if return_dtype == VariableDataType.INT:
            data = data.with_columns(pl.col(decision_variable_value_column).round(decimals=0).cast(pl.datatypes.Int64))

            return data

        raise ValueError(f"Unsupported return_dtype {return_dtype}")

    def unpack_column_variables(
        self,
        data: pd.DataFrame | pl.DataFrame,
        decision_variable_column: str,
        decision_variable_value_column: str,
        solver: AbstractOptimizationSolver,
        filter_column: str | None = None,
        default_unpack_value: float | None = None,
        return_dtype: VariableDataType | None = VariableDataType.FLOAT,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Unpacks decision variables from a DataFrame by applying a solver's get_variable_value method and removes the
        decision variable object column from the DataFrame.

        :param data: The DataFrame containing the decision variables.
        :param decision_variable_column: The name of the column containing the decision variables.
        :param decision_variable_value_column: The name of the column to store the unpacked decision variable values.
        :param solver: The solver object used to retrieve the variable values.
        :param filter_column: The name of the column with True/False values to filter the rows (optional, default is
        unpacking for all rows).
        :param default_unpack_value: The default value to use if a variable does not exist (optional, default is 0.0).
        :param return_dtype: return VariableDataType to cast decision_variable_value_column
        to (optional, default is float).
        :return: The updated DataFrame with the unpacked decision variable values.
        """
        # Input checks:
        if any(column in data.columns for column in ["indicators", "var_information"]):
            raise ValueError("DataFrame must not contain column names equal to 'indicators'")

        # Check if DataFrame is Pandas or Polars:
        is_pandas = isinstance(data, pd.DataFrame)
        is_polars = isinstance(data, pl.DataFrame)

        # Make indication for if row should have a variable:
        if isinstance(filter_column, str):
            indicators = data[filter_column].to_numpy()
        elif filter_column is None:
            indicators = np.ones(data.shape[0], dtype=bool)
        else:
            raise ValueError(f"Handling filter_column of type {type(filter_column)} not supported.")

        if default_unpack_value is None:
            default_unpack_value = 0.0

        if is_pandas:
            data = self._unpack_column_variables_pandas(
                data=data,
                decision_variable_column=decision_variable_column,
                decision_variable_value_column=decision_variable_value_column,
                solver=solver,
                indicators=indicators,
                default_unpack_value=default_unpack_value,
                return_dtype=return_dtype,
            )

            return data

        if is_polars:
            data = self._unpack_column_variables_polars(
                data=data,
                decision_variable_column=decision_variable_column,
                decision_variable_value_column=decision_variable_value_column,
                solver=solver,
                indicators=indicators,
                default_unpack_value=default_unpack_value,
                return_dtype=return_dtype,
            )

            return data

        raise ValueError(f"No method for unpacking DataFrame {type(data)} type is defined.")

    def _get_var_name_prefix(self, var_name: str | None, destination_column: str) -> str:
        """
        Find the prefix of the variable name. If var_name is None, we simply use the destination column name.

        :param var_name: The name of the variable prompted by the user.
        :param destination_column: The name of the column that should contain the decision variables.
        :return: The prefix of the variable name.
        """
        if isinstance(var_name, str):
            return var_name

        if var_name is None:
            return destination_column

        raise ValueError(f"Handling var_name of type {type(var_name)} not supported.")

    def _get_bound(
        self,
        bound: str | float | None,
        solver: AbstractOptimizationSolver,
        data: pd.DataFrame | pl.DataFrame,
        variable_dtype: VariableDataType,
        bound_type: str,
    ) -> np.array:
        """
        Find the bound of the variable. The bound of the variable is float, the bound will be assigned to all
        variables. If str, the column will be used as bound values (optional, default is -inf / inf).

        :param bound: The bound of the variable prompted by the user.
        :param solver: The solver object used to retrieve the variable values.
        :param data: The DataFrame containing the data.
        :param variable_dtype: The type of variable (int, bool, float).
        :param bound_type: The bound type of variable (lower or upper).
        :return: The lower or upper bound numpy array.
        """
        if isinstance(bound, str):
            return data[bound].to_numpy()

        if isinstance(bound, float):
            return np.full(data.shape[0], fill_value=bound)

        if bound is None:
            if bound_type == "lower":
                bound = 0.0 if variable_dtype == VariableDataType.BOOL else -solver.infinity()
            elif bound_type == "upper":
                bound = 1.0 if variable_dtype == VariableDataType.BOOL else solver.infinity()
            else:
                raise ValueError(f"Handling lower_bound of type {type(bound)} not supported.")

            return np.full(data.shape[0], fill_value=bound)

        raise ValueError(f"Handling bound of type {type(bound)} not supported.")

    def _get_indicators(
        self,
        filter_column: str | None,
        data: pd.DataFrame | pl.DataFrame,
    ) -> np.array:
        """
        Finds the indicator for whether a variable should be made. If filter_column is None, we create
        variables in all rows.

        :param filter_column: The name of the column with True/False values to filter the rows.
        :param data: The DataFrame containing the data.
        :return: The indicator numpy array.
        """
        if isinstance(filter_column, str):
            return data[filter_column].to_numpy()

        if filter_column is None:
            return np.ones(data.shape[0], dtype=bool)

        raise ValueError(f"Handling filter_column of type {type(filter_column)} not supported.")

    def _get_names(
        self, index_name_columns: list[str] | None, data: pd.DataFrame | pl.DataFrame, var_name_prefix: str
    ) -> np.array:
        """
        Finds the name of each variable. If index_name_columns is None, we use number from 0 -> n-1,
        where n is the number of rows.

        :param index_name_columns: List of column names for indexing the variable name.
        :param data: The DataFrame containing the data.
        :param var_name_prefix: The prefix of the variable name.
        :return: The names numpy array.
        """
        if isinstance(index_name_columns, list):
            return np.array(
                [f'{var_name_prefix}[{", ".join(row)}]' for row in data[index_name_columns].to_numpy().astype(str)]
            )

        if index_name_columns is None:
            return np.array([f"{var_name_prefix}[{str(index)}]" for index in range(data.shape[0])])

        raise ValueError(f"Handling index_name_columns of type {type(index_name_columns)} not supported.")
