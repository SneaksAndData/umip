from typing import Any
from unittest.mock import MagicMock

from adapta.logs import LoggerInterface

from generic_mip import AbstractObjectiveBuilder
from generic_mip.abstract_solver import AbstractOptimizationSolver


def test__abstract_objective_builder__add_analytics(logger):
    # arrange
    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.name = "whatever_i_want"
            self.add_analytics_granularity("location", self._location_analytics)
            self.add_analytics_granularity("sku_location", self._sku_location_analytics)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

        def _location_analytics(self, analytics_data) -> Any:
            return [1, 2, 3]

        def _sku_location_analytics(self, analytics_data) -> Any:
            return "something_else"

    # setup
    builder = ObjectiveBuilder1(logger=logger)

    # act & assert
    assert builder.get_analytics("location", analytics_data="whatever") == [1, 2, 3]
    assert builder.get_analytics("sku_location", analytics_data="whatever") == "something_else"
