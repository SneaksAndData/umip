from typing import Any

from adapta.logs import LoggerInterface

from generic_mip import AbstractObjectiveBuilder
from umip.abstract_solver import AbstractOptimizationSolver


def test__abstract_objective_builder__get_analytics__general(logger):
    """
    Tests whether the get_analytics method correctly calls the analytics calculators.
    """

    ### Arrange
    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.name = "whatever_i_want"
            self.add_analytics_granularity("aggregated", self._aggregated_analytics)
            self.add_analytics_granularity("location", self._location_analytics)
            self.add_analytics_granularity("sku_location", self._sku_location_analytics)
            self.number_of_granularity_calls = {
                "aggregated": 0,
                "location": 0,
                "sku_location": 0,
            }

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

        def _aggregated_analytics(self, analytics_data) -> Any:
            self.number_of_granularity_calls["aggregated"] += 1
            return sum(self.get_analytics("location", analytics_data))

        def _location_analytics(self, analytics_data) -> Any:
            self.number_of_granularity_calls["location"] += 1
            return [1, 2, 3]

        def _sku_location_analytics(self, analytics_data) -> Any:
            self.number_of_granularity_calls["sku_location"] += 1
            return "something_else"

    builder = ObjectiveBuilder1(logger=logger)

    ### Act & Assert
    # First, check that the supported granularities are as expected
    assert builder.get_analytics("location", analytics_data="whatever") == [1, 2, 3]
    assert builder.get_analytics("aggregated", analytics_data="whatever") == 6
    assert builder.get_analytics("sku_location", analytics_data="whatever") == "something_else"

    # Check that the number of calls to each granularity function is as expected
    assert set(builder.number_of_granularity_calls.values()) == {1}

    # Call location granularity with new analytics data and check that it works and increments call count
    assert builder.get_analytics("location", analytics_data="whatever_new") == [1, 2, 3]
    assert builder.number_of_granularity_calls["location"] == 2
