"""Solver types."""

from enum import Enum


class SolverType(Enum):
    """Enum for the different solver engines."""

    CPLEX = "CPLEX"
    GUROBI = "GUROBI"
    HIGHS = "HIGHS"
    SCIP = "SCIP"
    ORTOOLS_SCIP = "ORTOOLS_SCIP"
    ORTOOLS_CBC = "ORTOOLS_CBC"
    ORTOOLS_CPLEX = "ORTOOLS_CPLEX_ORTOOLS"
    ORTOOLS_XPRESS = "ORTOOLS_XPRESS"
    ORTOOLS_GLPK = "ORTOOLS_GLPK"
    ORTOOLS_GUROBI = "ORTOOLS_GUROBI"
