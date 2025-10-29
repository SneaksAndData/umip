import asyncio

import pandas as pd
import sys
from adapta.logs import create_async_logger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel

from examples.example_model import (
    MyMipModel,
    MyConstraintBuilder,
    MyVariableBuilder,
    MyObjectiveBuilder,
    MyDataPreparator,
    MyOtherConstraintBuilder,
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


# asyncio example
async def run_async():
    with create_async_logger(
        MyMipModel, log_handlers=[SafeStreamHandler(sys.stdout)], min_log_level=LogLevel.INFO
    ) as async_logger:
        my_async_model = MyMipModel(
            solver=SolverFactory(logger=async_logger).construct(solver_type=SolverType.ORTOOLS_SCIP),
            constraint_builders=[
                MyConstraintBuilder(logger=async_logger),
                MyOtherConstraintBuilder(logger=async_logger),
            ],
            variable_builders=[MyVariableBuilder(logger=async_logger)],
            objective_builders=[MyObjectiveBuilder(logger=async_logger)],
            data_preparator=MyDataPreparator(logger=async_logger),
            logger=async_logger,
        )
        await my_async_model.build_async(
            my_df=pd.DataFrame(
                data={
                    "row_number": [1, 2],
                }
            )
        )

        async_result = await my_async_model.solve_async()

    # allow to gracefully shut down and flush the logger
    await asyncio.sleep(1)

    print(async_result.iloc[0]["value"])
    print(async_result.iloc[1]["value"])


print("--- Running in ASYNC mode ---")
asyncio.run(run_async())
