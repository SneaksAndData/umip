import pandas as pd
import sys
from adapta.logs import SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel

from examples.example_model import (
    MyMipModel,
    MyConstraintBuilder,
    MyOtherConstraintBuilder,
    MyVariableBuilder,
    MyObjectiveBuilder,
    MyDataPreparator,
)
from generic_mip.enums import SolverType
from generic_mip.solver_factory import SolverFactory

"""
Model to implement:

maximize x + 4y

subject to:
x + y <= 100
y <= 20
100 >= x, y >= 0
x, y are integers

This model is equivalent to the example in solver_example.py.

You can access the solver API directly (as in solver_example.py) for local experiments, prototyping, debugging and project initialization.
For production grade models, use the generic_mip framework.
"""
logger = SemanticLogger().add_log_source(
    log_source_name="MyModel",
    min_log_level=LogLevel.DEBUG,
    log_handlers=[SafeStreamHandler(sys.stdout)],
    is_default=True,
)


print("--- Running in SYNC mode ---")
my_model = MyMipModel(
    solver=SolverFactory(logger=logger).construct(solver_type=SolverType.ORTOOLS_SCIP),
    constraint_builders=[MyConstraintBuilder(logger=logger), MyOtherConstraintBuilder(logger=logger)],
    variable_builders=[MyVariableBuilder(logger=logger)],
    objective_builders=[MyObjectiveBuilder(logger=logger)],
    data_preparator=MyDataPreparator(logger=logger),
    logger=logger,
)

my_model.build(
    my_df=pd.DataFrame(
        data={
            "row_number": [1, 2],
        }
    )
)

result = my_model.solve()

print(result.iloc[0]["value"])
print(result.iloc[1]["value"])
