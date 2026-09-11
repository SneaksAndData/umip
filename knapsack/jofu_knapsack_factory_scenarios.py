"""Run and plot capacity and maximum-volume-gap knapsack scenarios."""

import sys
from itertools import product
from pathlib import Path

import plotly.express as px
import polars as pl
from adapta.logs import SemanticLogger
from adapta.logs.handlers.safe_stream_handler import SafeStreamHandler
from adapta.logs.models import LogLevel
from jofu_knapsack import (
    ITEM_IS_SELECTED_VALUE,
    ITEM_NUMBER,
    PROFIT_COLUMN,
    VOLUME_COLUMN,
    KnapsackInputData,
    KnapsackModelFactory,
    KnapsackSettings,
)

from umip.enums import SolverType

CAPACITIES = range(10, 71, 5)
MAX_VOLUME_GAPS = range(1, 7)
DATA_PATH = Path(__file__).parent / "data" / "knapsack.parquet"


def load_knapsack_data() -> pl.DataFrame:
    return (
        pl.read_parquet(DATA_PATH)
        .with_row_index("item_id")
        .with_columns(
            pl.col("item_id").cast(pl.String).alias(ITEM_NUMBER),
        )
    )


def run_scenarios(
    knapsack_data: pl.DataFrame,
    capacities: range = CAPACITIES,
    max_volume_gaps: range = MAX_VOLUME_GAPS,
) -> pl.DataFrame:
    logger = SemanticLogger().add_log_source(
        log_source_name="KnapsackScenarios",
        min_log_level=LogLevel.INFO,
        log_handlers=[SafeStreamHandler(sys.stdout)],
        is_default=True,
    )
    settings = KnapsackSettings(
        add_large_small_gap_constraint=True,
        add_same_volume_pairs_constraint=True,
    )
    results: list[dict[str, int | float]] = []

    for capacity, max_gap in product(capacities, max_volume_gaps):
        model = KnapsackModelFactory(
            logger=logger,
            solver_type=SolverType.ORTOOLS_SCIP,
        ).construct(settings=settings)
        model.build(
            input_data=KnapsackInputData(
                knapsack_data=knapsack_data.clone(),
                knapsack_capacity=capacity,
                knapsack_volume_max_gap=max_gap,
            ),
            redirect_solver_log=False,
        )
        model.solve()

        output = model.get_output_data()
        selected = output.knapsack_data.filter(pl.col(ITEM_IS_SELECTED_VALUE))
        result = {
            "capacity": capacity,
            "max_gap": max_gap,
            "profit": selected.get_column(PROFIT_COLUMN).sum(),
            "volume": selected.get_column(VOLUME_COLUMN).sum(),
            "selected_items": selected.height,
        }
        results.append(result)

        print(
            f"capacity={capacity}, max_gap={max_gap}, "
            f"profit={result['profit']}, volume={result['volume']}, "
            f"selected_items={result['selected_items']}"
        )

    return pl.DataFrame(results)


def show_plots(results: pl.DataFrame) -> None:
    profit_by_capacity = (
        results.group_by("capacity")
        .agg(pl.col("profit").max())
        .sort("capacity")
    )
    px.line(
        x=profit_by_capacity["capacity"].to_list(),
        y=profit_by_capacity["profit"].to_list(),
        markers=True,
        labels={"x": "Capacity (liters)", "y": "Profit (€)"},
        title="Profit vs. Capacity",
    ).show()

    profit_by_gap = (
        results.group_by("max_gap")
        .agg(pl.col("profit").max())
        .sort("max_gap")
    )
    px.line(
        x=profit_by_gap["max_gap"].to_list(),
        y=profit_by_gap["profit"].to_list(),
        markers=True,
        labels={"x": "Maximum volume gap (L)", "y": "Profit (€)"},
        title="Profit vs. L",
    ).show()

    px.density_heatmap(
        x=results["capacity"].to_list(),
        y=results["max_gap"].to_list(),
        z=results["profit"].to_list(),
        histfunc="max",
        text_auto=True,
        labels={
            "x": "Capacity (liters)",
            "y": "Maximum volume gap (L)",
            "color": "Profit (€)",
        },
        title="Profit by Capacity and L",
    ).show()


def main() -> None:
    results = run_scenarios(load_knapsack_data())
    print("\nSummary:")
    print(results)
    show_plots(results)


if __name__ == "__main__":
    main()
