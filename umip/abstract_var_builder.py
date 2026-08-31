"""Abstract definition of a variable builder."""
from abc import ABC, abstractmethod
from adapta.logs import LoggerInterface
import pandas as pd
import polars as pl
import numpy as np

from umip.enums import (
    VariableDomain,
    BoundType,
    DataFrameArgumentType,
    BoundArgumentType,
    FilterColumnArgumentType,
    IndexColumnsArgumentType,
)
from umip.abstract_solver import AbstractOptimizationSolver
from umip.abstract_dataclasses import AbstractInternalData


class AbstractDecisionVariableBuilder(ABC):
    """A variable builder has the responsibility of building one or more decision variables."""

    lower_bound_column_name = "lb"
    upper_bound_column_name = "ub"
    variable_name_column_name = "var_name"
    indicator_column_name = "indicators"
    variable_information_column_name = "var_information"

    invalid_column_names_build = [
        lower_bound_column_name,
        upper_bound_column_name,
        variable_name_column_name,
        indicator_column_name,
        variable_information_column_name,
    ]
    invalid_column_names_unpack = [indicator_column_name, variable_information_column_name]

    def __init__(self, logger: LoggerInterface):
        """
        Initialize the variable builder.
        :param logger: The logger to use.
        """
        self._logger = logger

    @abstractmethod
    def build(self, solver: AbstractOptimizationSolver, data: AbstractInternalData) -> AbstractInternalData:
        """
        Builds the decision variables on the given model and the given data.

        :param solver: The solver to use to build the variables.
        :param data: The data (e.g. dataframes) providing parameters for the variables.
        :return: The dataframes decorated with the created decision variables.
        """

    @abstractmethod
    def unpack(self, solver: AbstractOptimizationSolver, data: AbstractInternalData) -> AbstractInternalData:
        """
        Unpacks the decision variables after optimization and inserts variable values in the dataframes.

        :param solver: The solver to get the variable values from.
        :param data: The data (e.g. dataframes) containing the variables.
        :return: The dataframes decorated with the values of the decision variables.
        """

    @staticmethod
    def _get_dataframe_argument_type(data: pd.DataFrame | pl.DataFrame) -> DataFrameArgumentType:
        """
        Determines the type of dataframe provided as input.
        """
        if isinstance(data, pd.DataFrame):
            return DataFrameArgumentType.PANDAS

        if isinstance(data, pl.DataFrame):
            return DataFrameArgumentType.POLARS

        raise ValueError(f"Unsupported dataframe type {type(data)}")

    @staticmethod
    def _get_bound_argument_type(bound: float | str | None) -> BoundArgumentType:
        """
        Gets the type of the bound argument.
        """
        if isinstance(bound, float):
            return BoundArgumentType.FLOAT
        if isinstance(bound, str):
            return BoundArgumentType.STRING
        if bound is None:
            return BoundArgumentType.NONE

        raise ValueError(f"Unsupported bound argument type {type(bound)}")

    @staticmethod
    def _get_filter_column_argument_type(filter_column: str | None) -> FilterColumnArgumentType:
        """
        Gets the type of the filter column argument.
        """
        if filter_column is None:
            return FilterColumnArgumentType.NONE
        if isinstance(filter_column, str):
            return FilterColumnArgumentType.STRING

        raise ValueError(f"Unsupported filter column argument type {type(filter_column)}")

    @staticmethod
    def _get_index_columns_argument_type(index_name_columns: list[str] | None) -> IndexColumnsArgumentType:
        """
        Gets the type of the index columns argument.
        """
        if index_name_columns is None:
            return IndexColumnsArgumentType.NONE
        if isinstance(index_name_columns, list) and all(isinstance(item, str) for item in index_name_columns):
            return IndexColumnsArgumentType.LIST_OF_STRINGS

        raise ValueError(f"Unsupported index columns argument type {type(index_name_columns)}")

    def _get_row_count(self, data: pd.DataFrame | pl.DataFrame) -> int:
        """Returns the number of rows in the DataFrame."""
        dataframe_type = self._get_dataframe_argument_type(data=data)
        if dataframe_type == DataFrameArgumentType.PANDAS:
            return len(data)
        if dataframe_type == DataFrameArgumentType.POLARS:
            return data.height
        raise ValueError(f"Cannot get row count for unsupported data type: {type(data)}")

    def _dataframe_has_column(self, data: pd.DataFrame | pl.DataFrame, column_name: str) -> bool:
        """Returns whether the DataFrame contains the specified column."""
        dataframe_type = self._get_dataframe_argument_type(data=data)
        if dataframe_type == DataFrameArgumentType.PANDAS:
            if not column_name in data.columns:
                raise ValueError(f"DataFrame does not contain column {column_name}.")
            return True
        if dataframe_type == DataFrameArgumentType.POLARS:
            if not column_name in data.columns:
                raise ValueError(f"DataFrame does not contain column {column_name}.")
            return True
        raise ValueError(f"Cannot check for column existence for unsupported dataframe type: {dataframe_type}")

    def _dataframe_has_invalid_columns(
        self, data: pd.DataFrame | pl.DataFrame, invalid_column_names: list[str]
    ) -> bool:
        """
        Returns whether the dataframe contains any columns from the list of invalid column names.
        """
        dataframe_type = self._get_dataframe_argument_type(data=data)
        if dataframe_type == DataFrameArgumentType.PANDAS:
            return any(column in data.columns for column in invalid_column_names)
        if dataframe_type == DataFrameArgumentType.POLARS:
            return any(column in data.columns for column in invalid_column_names)
        raise ValueError(f"Cannot check for invalid column existence for unsupported dataframe type: {dataframe_type}")

    def _build_column_variables_pandas(
        self,
        solver: AbstractOptimizationSolver,
        data: pd.DataFrame,
        destination_column: str,
        variable_domain: VariableDomain,
        lower_bound_values: np.ndarray,
        upper_bound_values: np.ndarray,
        names: np.ndarray,
        indicators: np.ndarray,
    ) -> pd.DataFrame:
        """
        Builds decision variables on a pandas DataFrame by applying a solver's add_variable method and add this to a
        new column in the dataframe.

        :param solver: The solver object used to retrieve the variable values.
        :param data: The DataFrame containing the data.
        :param destination_column: The name of the column that should contain the decision variables.
        :param variable_domain: The domain of variable (integer, binary, continuous).
        :param lower_bound_values: The lower bound values.
        :param upper_bound_values: The upper bound values.
        :param names: The names of the variables.
        :param indicators: The indicator for whether variables should be created or not.
        :return: The updated pandas DataFrame with the decision variable values in the destination_column.
        """
        if self._dataframe_has_invalid_columns(data=data, invalid_column_names=self.invalid_column_names_build):
            raise ValueError(
                f"DataFrame must not contain column names from the following list: {self.invalid_column_names_build}."
            )

        if data.empty:
            return data.assign(**{destination_column: None})

        data = data.assign(
            **{
                self.lower_bound_column_name: lower_bound_values,
                self.upper_bound_column_name: upper_bound_values,
                self.variable_name_column_name: names,
                self.indicator_column_name: indicators,
            }
        )

        data[self.variable_information_column_name] = data[
            [
                self.lower_bound_column_name,
                self.upper_bound_column_name,
                self.variable_name_column_name,
                self.indicator_column_name,
            ]
        ].values.tolist()

        # Building variables, where y[0] is lb, y[1] is ub, y[2] is var_name and y[3] is indicators
        data = data.assign(
            **{
                destination_column: lambda x: x[self.variable_information_column_name].apply(
                    lambda y: solver.add_variable(
                        lower_bound=y[0], upper_bound=y[1], name=y[2], variable_domain=variable_domain
                    )
                    if y[3]
                    else None
                ),
            }
        )

        data = data.drop(
            columns=[
                self.lower_bound_column_name,
                self.upper_bound_column_name,
                self.variable_name_column_name,
                self.indicator_column_name,
                self.variable_information_column_name,
            ]
        )

        return data

    def _build_column_variables_polars(
        self,
        solver: AbstractOptimizationSolver,
        data: pl.DataFrame,
        destination_column: str,
        variable_domain: VariableDomain,
        lower_bound_values: np.ndarray,
        upper_bound_values: np.ndarray,
        names: np.ndarray,
        indicators: np.ndarray,
    ) -> pl.DataFrame:
        """
        Builds decision variables on a polars DataFrame by applying a solver's add_variable method and add this to a
        new column in the dataframe.

        :param solver: The solver object used to retrieve the variable values.
        :param data: The DataFrame containing the data.
        :param destination_column: The name of the column that should contain the decision variables.
        :param variable_domain: The domain of variable (integer, binary, continuous).
        :param lower_bound_values: The lower bound values.
        :param upper_bound_values: The upper bound values.
        :param names: The names of the variables.
        :param indicators: The indicator for whether variables should be created or not.
        :return: The updated polars DataFrame with the decision variable values in the destination_column.
        """
        if self._dataframe_has_invalid_columns(data=data, invalid_column_names=self.invalid_column_names_build):
            raise ValueError(
                f"DataFrame must not contain column names from the following list: {self.invalid_column_names_build}."
            )

        if data.is_empty():
            return data.with_columns(pl.lit(None).alias(destination_column))

        def build_variables(row: dict) -> pl.datatypes.Object:
            return solver.add_variable(
                lower_bound=row[self.lower_bound_column_name],
                upper_bound=row[self.upper_bound_column_name],
                name=row[self.variable_name_column_name],
                variable_domain=variable_domain,
            )

        data = data.with_columns(
            **{
                self.lower_bound_column_name: lower_bound_values,
                self.upper_bound_column_name: upper_bound_values,
                self.variable_name_column_name: names,
                self.indicator_column_name: indicators,
            }
        )

        data = data.with_columns(
            pl.when(pl.col(self.indicator_column_name))
            .then(
                pl.struct(
                    [self.lower_bound_column_name, self.upper_bound_column_name, self.variable_name_column_name]
                ).map_elements(build_variables, return_dtype=pl.datatypes.Object)
            )
            .otherwise(None)
            .cast(pl.datatypes.Object)
            .alias(destination_column)
        )

        data = data.drop(
            [
                self.lower_bound_column_name,
                self.upper_bound_column_name,
                self.variable_name_column_name,
                self.indicator_column_name,
            ]
        )

        return data

    def build_column_variables(
        self,
        solver: AbstractOptimizationSolver,
        data: pd.DataFrame | pl.DataFrame,
        destination_column: str,
        variable_domain: VariableDomain,
        index_name_columns: list[str] | None = None,
        lower_bound: float | str | None = None,
        upper_bound: float | str | None = None,
        variable_name: str | None = None,
        filter_column: str | None = None,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Builds decision variables from a DataFrame by applying a solver's add_variable method and add this to a new
        column in the dataframe.

        :param solver: The solver object used to retrieve the variable values.
        :param data: The DataFrame containing the data.
        :param destination_column: The name of the column that should contain the decision variables.
        :param variable_domain: The domain of variable (integer, binary, continuous).
        :param index_name_columns: list of column names for indexing the variable name (optional, default naming will
        be after row number).
        :param lower_bound: The lower bound of the variable. If float, the lower bound will be assigned to all
        variables. If str, the column will be used as lower bound values (optional, default is -inf).
        both lower_bound and lower_bound_column is defined, we use the lower_bound_column values.
        :param upper_bound: The upper bound of the variable. If float, the upper bound will be assigned to all
        variables. If str, the column will be used as upper bound values (optional, default is inf).
        both upper_bound and upper_bound_column is defined, we use the upper_bound_column values.
        :param variable_name: The name of the variable (optional, default is destination_column).
        :param filter_column: The name of the column with True/False values to filter the rows (optional, default is
        creating variables for all rows).
        :return: The updated DataFrame with the decision variable values in the destination_column.
        """
        dataframe_type = self._get_dataframe_argument_type(data=data)

        variable_name = self._get_variable_name(variable_name=variable_name, destination_column=destination_column)

        lower_bound_values = self._get_bounds(
            bound=lower_bound,
            solver=solver,
            data=data,
            variable_domain=variable_domain,
            bound_type=BoundType.LOWER,
        )
        upper_bound_values = self._get_bounds(
            bound=upper_bound,
            solver=solver,
            data=data,
            variable_domain=variable_domain,
            bound_type=BoundType.UPPER,
        )

        indicators = self._get_indicators(filter_column=filter_column, data=data)

        names = self._get_variable_name_with_indices(
            index_column_names=index_name_columns, data=data, variable_name=variable_name
        )

        if dataframe_type == DataFrameArgumentType.PANDAS:
            data = self._build_column_variables_pandas(
                solver=solver,
                data=data,
                destination_column=destination_column,
                variable_domain=variable_domain,
                lower_bound_values=lower_bound_values,
                upper_bound_values=upper_bound_values,
                names=names,
                indicators=indicators,
            )

            return data

        if dataframe_type == DataFrameArgumentType.POLARS:
            data = self._build_column_variables_polars(
                solver=solver,
                data=data,
                destination_column=destination_column,
                variable_domain=variable_domain,
                lower_bound_values=lower_bound_values,
                upper_bound_values=upper_bound_values,
                names=names,
                indicators=indicators,
            )

            return data

        raise ValueError(f"No method for building variables in DataFrame {dataframe_type} type is defined.")

    def _unpack_column_variables_pandas(
        self,
        data: pd.DataFrame,
        decision_variable_column: str,
        decision_variable_value_column: str,
        solver: AbstractOptimizationSolver,
        indicators: np.ndarray,
        default_unpack_value: float,
        variable_domain: VariableDomain,
    ) -> pd.DataFrame:
        """
        Unpacks decision variables from a pandas DataFrame by applying a solver's get_variable_value method and removes
        the decision variable object column from the DataFrame.

        :param data: The DataFrame containing the decision variables.
        :param decision_variable_column: The name of the column containing the decision variables.
        :param decision_variable_value_column: The name of the column to store the unpacked decision variable values.
        :param solver: The solver object used to retrieve the variable values.
        :param indicators: The indicator for if a variable should be unpacked or not. If not, the value is set
        to default_unpack_value.
        :param default_unpack_value: The default value to use if a variable does not exist.
        :param variable_domain: domain of the variable.
        to (optional, default is float).
        :return: The updated pandas DataFrame with the unpacked decision variable values.
        """
        if self._dataframe_has_invalid_columns(data=data, invalid_column_names=self.invalid_column_names_unpack):
            raise ValueError(
                f"DataFrame must not contain column names from the following list: {self.invalid_column_names_unpack}."
            )

        if data.empty:
            data = data.assign(**{decision_variable_value_column: np.array(None, dtype="float")}).drop(
                columns=[decision_variable_column]
            )
        else:
            data = data.assign(
                **{
                    self.indicator_column_name: indicators,
                }
            )

            data[self.variable_information_column_name] = data[
                [decision_variable_column, self.indicator_column_name]
            ].values.tolist()

            data = data.assign(
                **{
                    decision_variable_value_column: lambda x: x[self.variable_information_column_name].apply(
                        lambda y: solver.get_variable_value(y[0]) if y[1] else default_unpack_value
                    )
                }
            ).drop(
                columns=[decision_variable_column, self.indicator_column_name, self.variable_information_column_name]
            )

        if variable_domain == VariableDomain.CONTINUOUS:
            return data

        if variable_domain == VariableDomain.BINARY:
            data[decision_variable_value_column] = data[decision_variable_value_column].round(decimals=0).astype("bool")
            return data

        if variable_domain == VariableDomain.INTEGER:
            data[decision_variable_value_column] = data[decision_variable_value_column].round(decimals=0).astype("int")
            return data

        raise ValueError(f"Unsupported variable data type {variable_domain}")

    def _unpack_column_variables_polars(
        self,
        data: pl.DataFrame,
        decision_variable_column: str,
        decision_variable_value_column: str,
        solver: AbstractOptimizationSolver,
        indicators: np.ndarray,
        default_unpack_value: float,
        variable_domain: VariableDomain,
    ) -> pl.DataFrame:
        """
        Unpacks decision variables from a polars DataFrame by applying a solver's get_variable_value method and removes
        the decision variable object column from the DataFrame.

        :param data: The DataFrame containing the decision variables.
        :param decision_variable_column: The name of the column containing the decision variables.
        :param decision_variable_value_column: The name of the column to store the unpacked decision variable values.
        :param solver: The solver object used to retrieve the variable values.
        :param indicators: The indicator for if a variable should be unpacked or not. If not, the value is set
        to default_unpack_value.
        :param default_unpack_value: The default value to use if a variable does not exist.
        :param variable_domain: VariableDomain to cast decision_variable_value_column
        to (optional, default is float).
        :return: The updated polars DataFrame with the unpacked decision variable values.
        """
        if self._dataframe_has_invalid_columns(data=data, invalid_column_names=self.invalid_column_names_unpack):
            raise ValueError(
                f"DataFrame must not contain column names from the following list: {self.invalid_column_names_unpack}."
            )

        if data.is_empty():
            data = data.with_columns(pl.lit(None).cast(pl.Float64).alias(decision_variable_value_column)).drop(
                decision_variable_column
            )
        else:
            data = data.with_columns(
                **{
                    self.indicator_column_name: indicators,
                }
            )

            data = data.with_columns(
                pl.when(pl.col(self.indicator_column_name))
                .then(pl.col(decision_variable_column).map_elements(solver.get_variable_value, return_dtype=pl.Float64))
                .otherwise(default_unpack_value)
                .alias(decision_variable_value_column)
            ).drop([decision_variable_column, self.indicator_column_name])

        if variable_domain == VariableDomain.CONTINUOUS:
            return data

        if variable_domain == VariableDomain.BINARY:
            data = data.with_columns(
                pl.col(decision_variable_value_column).round(decimals=0).cast(pl.datatypes.Boolean)
            )

            return data

        if variable_domain == VariableDomain.INTEGER:
            data = data.with_columns(pl.col(decision_variable_value_column).round(decimals=0).cast(pl.datatypes.Int64))

            return data

        raise ValueError(f"Unsupported variable data type {variable_domain}")

    def unpack_column_variables(
        self,
        data: pd.DataFrame | pl.DataFrame,
        decision_variable_column: str,
        decision_variable_value_column: str,
        solver: AbstractOptimizationSolver,
        filter_column: str | None = None,
        default_unpack_value: float = 0.0,
        variable_domain: VariableDomain | None = VariableDomain.CONTINUOUS,
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
        :param variable_domain: VariableDomain to cast decision_variable_value_column
        to (optional, default is float).
        :return: The updated DataFrame with the unpacked decision variable values.
        """
        dataframe_type = self._get_dataframe_argument_type(data=data)

        indicators = self._get_indicators(filter_column, data)

        if dataframe_type == DataFrameArgumentType.PANDAS:
            data = self._unpack_column_variables_pandas(
                data=data,
                decision_variable_column=decision_variable_column,
                decision_variable_value_column=decision_variable_value_column,
                solver=solver,
                indicators=indicators,
                default_unpack_value=default_unpack_value,
                variable_domain=variable_domain,
            )

            return data

        if dataframe_type == DataFrameArgumentType.POLARS:
            data = self._unpack_column_variables_polars(
                data=data,
                decision_variable_column=decision_variable_column,
                decision_variable_value_column=decision_variable_value_column,
                solver=solver,
                indicators=indicators,
                default_unpack_value=default_unpack_value,
                variable_domain=variable_domain,
            )

            return data

        raise ValueError(f"No method for unpacking variables in DataFrame of type {dataframe_type} is defined.")

    @staticmethod
    def _get_variable_name(destination_column: str, variable_name: str | None = None) -> str:
        """
        Find the variable name. If variable_name is None, we simply use the destination column name.

        :param variable_name: The name of the variable prompted by the user.
        :param destination_column: The name of the column that should contain the decision variables.
        :return: The prefix of the variable name.
        """
        if variable_name is not None and not isinstance(variable_name, str):
            raise ValueError(f"Handling variable_name of type {type(variable_name)} not supported.")

        return variable_name or destination_column

    def _get_bounds(
        self,
        bound: str | float | None,
        solver: AbstractOptimizationSolver,
        data: pd.DataFrame | pl.DataFrame,
        variable_domain: VariableDomain,
        bound_type: BoundType,
    ) -> np.ndarray:
        """
        Find the bound of the variable. The bound of the variable is float, the bound will be assigned to all
        variables. If str, the column will be used as bound values (optional, default is -inf / inf).

        :param bound: The bound of the variable prompted by the user.
        :param solver: The solver object used to retrieve the variable values.
        :param data: The DataFrame containing the data.
        :param variable_domain: The domain of the variable (integer, binary, continuous).
        :param bound_type: The bound type of variable (lower or upper).
        :return: The lower or upper bound numpy array.
        """
        bound_argument_type = self._get_bound_argument_type(bound=bound)
        dataframe_type = self._get_dataframe_argument_type(data=data)
        number_of_variables = self._get_row_count(data=data)

        if bound_argument_type == BoundArgumentType.STRING:
            if dataframe_type == DataFrameArgumentType.PANDAS and self._dataframe_has_column(
                data=data, column_name=bound
            ):
                return data[bound].fillna(value=solver.infinity()).to_numpy()
            if dataframe_type == DataFrameArgumentType.POLARS and self._dataframe_has_column(
                data=data, column_name=bound
            ):
                return data.get_column(name=bound).fill_null(value=solver.infinity()).to_numpy()
            raise ValueError(f"Cannot find a bound column in DataFrame of unsupported type {dataframe_type}.")

        if bound_argument_type == BoundArgumentType.FLOAT:
            return np.full(number_of_variables, fill_value=bound)

        if bound_argument_type == BoundArgumentType.NONE:
            if bound_type == BoundType.LOWER:
                bound = 0.0 if variable_domain == VariableDomain.BINARY else -solver.infinity()
            elif bound_type == BoundType.UPPER:
                bound = 1.0 if variable_domain == VariableDomain.BINARY else solver.infinity()
            else:
                raise ValueError(f"Handling of bound_type={bound_type} not supported.")

            return np.full(number_of_variables, fill_value=bound)

        raise ValueError(f"Handling bound of type {bound_argument_type} not supported.")

    def _get_indicators(
        self,
        filter_column: str | None,
        data: pd.DataFrame | pl.DataFrame,
    ) -> np.ndarray:
        """
        Finds the indicator for whether a variable should exist. If filter_column is None, the variable
        exists for all rows.

        :param filter_column: The name of the column with True/False values to filter the rows.
        :param data: The DataFrame containing the data.
        :return: The indicator numpy array.
        """
        filter_column_type = self._get_filter_column_argument_type(filter_column=filter_column)
        dataframe_type = self._get_dataframe_argument_type(data=data)
        number_of_variables = self._get_row_count(data=data)

        if filter_column_type == FilterColumnArgumentType.STRING:
            if dataframe_type == DataFrameArgumentType.POLARS and self._dataframe_has_column(
                data=data, column_name=filter_column
            ):
                return data[filter_column].to_numpy()
            if dataframe_type == DataFrameArgumentType.PANDAS and self._dataframe_has_column(
                data=data, column_name=filter_column
            ):
                return data[filter_column].to_numpy()
            raise ValueError(f"Cannot find a filter column in DataFrame of unsupported type {dataframe_type}.")

        if filter_column_type == FilterColumnArgumentType.NONE:
            return np.ones(number_of_variables, dtype=bool)

        raise ValueError(f"Handling filter_column of type {filter_column_type} not supported.")

    def _get_variable_name_with_indices(
        self, index_column_names: list[str] | None, data: pd.DataFrame | pl.DataFrame, variable_name: str
    ) -> np.ndarray:
        """
        Finds the name of each variable. If index_name_columns is None, we use numbers from 0 -> n-1,
        where n is the number of rows.

        :param index_column_names: List of column names for indexing the variable name.
        :param data: The DataFrame containing the data.
        :param variable_name: The prefix of the variable name.
        :return: The names numpy array.
        """
        index_name_columns_type = self._get_index_columns_argument_type(index_name_columns=index_column_names)
        dataframe_type = self._get_dataframe_argument_type(data=data)

        if index_name_columns_type == IndexColumnsArgumentType.LIST_OF_STRINGS:
            if dataframe_type == DataFrameArgumentType.PANDAS:
                return np.array(
                    [
                        f'{variable_name}[{", ".join(index)}]'
                        for index in data[index_column_names].to_numpy().astype(str)
                    ]
                )
            if dataframe_type == DataFrameArgumentType.POLARS:
                return np.array(
                    [
                        f'{variable_name}[{", ".join(index)}]'
                        for index in data[index_column_names].to_numpy().astype(str)
                    ]
                )
            raise ValueError(f"Cannot add variable indices from a DataFrame of unsupported type {dataframe_type}.")

        if index_name_columns_type == IndexColumnsArgumentType.NONE:
            return np.array([f"{variable_name}[{str(index)}]" for index in range(self._get_row_count(data=data))])

        raise ValueError(f"Handling index_name_columns of type {index_name_columns_type} not supported.")
