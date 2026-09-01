from enum import Enum


class CplexStatus(Enum):
    """Cplex status values.

    This `Enum` is used to convert cplex status string values into an
    enumeration
    """

    OPTIMAL = "optimal"
    INTEGER_OPTIMAL = "integer optimal solution"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INTEGER_INFEASIBLE = "integer infeasible"
    UNBOUNDED = "unbounded"
    NUMERICAL_DIFFICULTIES = "numerical_difficulties"
    ABORTED = "aborted"
    UNKNOWN = "unknown"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    INTEGER_LIMIT = "integer_limit"
