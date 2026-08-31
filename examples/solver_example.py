import numpy as np
import sys
from adapta.logs import SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel
from umip import VariableDomain
from umip.enums import SolverType
from umip.solver_factory import SolverFactory

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

logger = SemanticLogger().add_log_source(
    log_source_name="MyModel",
    min_log_level=LogLevel.DEBUG,
    log_handlers=[SafeStreamHandler(sys.stdout)],
    is_default=True,
)

solver = SolverFactory(logger=logger).construct(solver_type=SolverType.ORTOOLS_SCIP)

x = solver.add_variable(
    lower_bound=0, upper_bound=100, name="x", variable_domain=VariableDomain.INTEGER
)
y = solver.add_variable(
    lower_bound=0, upper_bound=100, name="y", variable_domain=VariableDomain.INTEGER
)
solver.add_multiple_objective_terms(
    coefficients=np.array([1.0, 4.0]),
    variables=np.array([x, y]),
)
solver.add_constraint(
    lower_bound=None,
    upper_bound=100,
    coefficients=np.array([1.0, 1.0]),
    variables=np.array([x, y]),
    name="my_constraint",
)
solver.add_constraint(
    lower_bound=None,
    upper_bound=20,
    coefficients=np.array([1.0]),
    variables=np.array([y]),
    name="my_constraint2",
)
solver.set_optimization_direction(True)
solver.set_verbose(True)
solver.solve()
print(solver.get_variable_value(x))
print(solver.get_variable_value(y))
