import numpy as np
import sys
from adapta.logs import SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel
from generic_mip import VariableDataType
from generic_mip.solver.cplex import CplexSolver
from generic_mip.solver.or_tools import OrToolsSolverEngine, OrToolsSolver
from generic_mip.solver.gurobi import GurobiSolver
from generic_mip.solver.highs import HighsSolver


"""
Model to implement:

maximize x + 4y

subject to:
x + y <= 100
y <= 20
100 >= x, y >= 0
x, y are integers

This model is equivalent to the example in full_example.py.

You can access the solver API directly for local experiments, prototyping, debugging and project initialization.
For production grade models, use the generic_mip framework (full_example.py).
"""

SOLVER = "cplex"

logger = SemanticLogger().add_log_source(
    log_source_name="MyModel",
    min_log_level=LogLevel.DEBUG,
    log_handlers=[SafeStreamHandler(sys.stdout)],
    is_default=True,
)

if SOLVER == "ortools":
    solver = OrToolsSolver(solver_engine=OrToolsSolverEngine.SCIP, logger=logger)
elif SOLVER == "gurobi":
    solver = GurobiSolver(logger=logger)
elif SOLVER == "highs":
    solver = HighsSolver(logger=logger)
elif SOLVER == "cplex":
    solver = CplexSolver(logger=logger)
else:
    raise ValueError(f"Invalid solver: {SOLVER}")

x = solver.add_variable(lb=0, ub=100, name="x", dtype=VariableDataType.INT)
y = solver.add_variable(lb=0, ub=100, name="y", dtype=VariableDataType.INT)
solver.add_multiple_objective_terms(
    coeffs=np.array([1.0, 4.0]),
    vars_=np.array([x, y]),
)
solver.add_constraint(lb=None, ub=100, coeffs=np.array([1.0, 1.0]), vars_=np.array([x, y]), name="my_constraint")
solver.add_constraint(lb=None, ub=20, coeffs=np.array([1.0]), vars_=np.array([y]), name="my_constraint2")
solver.set_optimization_direction(True)
solver.set_verbose(True)
solver.solve()
print(solver.get_variable_value(x))
print(solver.get_variable_value(y))
