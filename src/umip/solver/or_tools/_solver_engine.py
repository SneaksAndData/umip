"""
Solver engine for OR-Tools.
"""

from enum import Enum


class OrToolsSolverEngine(Enum):
    """OR-Tools compatible MIP solver engines."""

    SCIP = "SCIP"
    GUROBI = "GUROBI"
    CBC = "CBC"
    CPLEX = "CPLEX"
    XPRESS = "XPRESS"
    GLPK = "GLPK"
