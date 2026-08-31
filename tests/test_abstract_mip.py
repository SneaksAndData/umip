"""
Tests of the AbstractMipModel class. Tests do not cover the solver classes and simple wrappers.
"""

from unittest.mock import MagicMock, call

import pytest
from adapta.logs import LoggerInterface

from tests.mock_classes_and_data import *
from umip.enums import SolverType
from umip.solver_factory import SolverFactory


def _construct_solver(
    logger: LoggerInterface, solver_type: SolverType = SolverType.ORTOOLS_SCIP
) -> AbstractOptimizationSolver:
    return SolverFactory(logger=logger).construct(solver_type=solver_type)


def test__build__general():
    """
    Tests that all provided builders are called.
    """
    # Arrange
    mock_data_preparator = MagicMock(spec=AbstractDataPreparator, prepare=MagicMock())
    mock_variable_builder = MagicMock(spec=AbstractDecisionVariableBuilder, build=MagicMock(), unpack=MagicMock())
    mock_constraint_builder = MagicMock(spec=AbstractConstraintBuilder, build=MagicMock())
    mock_objective_builder = MagicMock(
        spec=AbstractObjectiveBuilder,
        build=MagicMock(),
        unpack=MagicMock(),
        objective_name="test_name",
    )
    logger = MagicMock(spec=LoggerInterface)

    model = MockMipModel(
        solver=_construct_solver(logger=logger),
        constraint_builders=[mock_constraint_builder],
        variable_builders=[mock_variable_builder],
        objective_builders=[mock_objective_builder],
        data_preparator=mock_data_preparator,
        logger=logger,
    )

    # Act
    model.build(input_data=MagicMock())

    # Assert
    mock_data_preparator.prepare.assert_called_once()
    mock_variable_builder.build.assert_called_once()
    mock_objective_builder.build.assert_called_once()
    mock_constraint_builder.build.assert_called_once()
    mock_variable_builder.unpack.assert_not_called()


def test__solve__model_built__methods_are_called():
    """
    Testing that the right method calls are made when solve is called on a model that has been built.
    """
    # Arrange
    mock_objective_builder = MagicMock(spec=AbstractObjectiveBuilder)
    mock_variable_builder = MagicMock(spec=AbstractDecisionVariableBuilder, unpack=MagicMock())
    mock_constraint_builder = MagicMock(spec=AbstractConstraintBuilder)
    mock_data_preparator = MagicMock(spec=AbstractDataPreparator)
    logger = MagicMock(spec=LoggerInterface)

    model = MockMipModel(
        solver=_construct_solver(logger=logger),
        constraint_builders=[mock_constraint_builder],
        variable_builders=[mock_variable_builder],
        objective_builders=[mock_objective_builder],
        data_preparator=mock_data_preparator,
        logger=logger,
    )
    model._built = True
    model._internal_data = MockInternalData(data=MagicMock())

    # Act
    model.solve()

    # Assert
    assert model._solved
    mock_variable_builder.unpack.assert_called_once()


def test__solve__model_not_built__raises_value_error():
    """
    Testing that the solve method raises an error if the model is not built.
    """
    # Arrange
    mock_objective_builder = MagicMock(spec=AbstractObjectiveBuilder)
    mock_variable_builder = MagicMock(spec=AbstractDecisionVariableBuilder)
    mock_constraint_builder = MagicMock(spec=AbstractConstraintBuilder)
    mock_data_preparator = MagicMock(spec=AbstractDataPreparator)
    logger = MagicMock(spec=LoggerInterface)

    model = MockMipModel(
        solver=_construct_solver(logger=logger),
        constraint_builders=[mock_constraint_builder],
        variable_builders=[mock_variable_builder],
        objective_builders=[mock_objective_builder],
        data_preparator=mock_data_preparator,
        logger=logger,
    )

    # Act & Assert
    with pytest.raises(ValueError):
        model.solve()


@pytest.mark.parametrize("solver_type", [SolverType.ORTOOLS_SCIP])
def test__abstract_mip__get_analytics__analytics_data_not_provided(solver_type: SolverType, logger):
    """
    Tests the get_analytics method for the case when analytics_data is not provided as an argument for
    the method (model._output_data must be used)
    """

    @dataclass
    class TestInternalData(AbstractInternalData):
        location: list[int]
        sku: list[int]

    class TestOutputData(TestInternalData, AbstractOutputData):
        pass

    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.objective_name = "builder_1"
            self.add_analytics_granularity("sku", self._sku_analytics)
            self.add_analytics_granularity("sku_location", self._sku_location_analytics)
            self.add_analytics_granularity("aggregated", self._aggregated_analytics)

        def build(self, solver: AbstractOptimizationSolver, data: TestInternalData) -> None:
            pass

        def _sku_analytics(self, analytics_data: TestInternalData) -> Any:
            return analytics_data.sku

        def _sku_location_analytics(self, analytics_data: TestOutputData) -> Any:
            return "something_else"

        def _aggregated_analytics(self, analytics_data: TestOutputData) -> Any:
            return sum(analytics_data.sku)

    class ObjectiveBuilder2(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.add_analytics_granularity("aggregated", self.aggregated_analytics)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

        def aggregated_analytics(self, analytics_data: TestOutputData) -> Any:
            return sum(analytics_data.location)

    model = MockMipModel(
        solver=_construct_solver(logger=logger, solver_type=solver_type),
        constraint_builders=[MockConstraintBuilder(logger)],
        variable_builders=[MockDecisionVariableBuilder(logger)],
        objective_builders=[ObjectiveBuilder1(logger), ObjectiveBuilder2(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger,
    )

    # we don't want to test optimization here, so we are simply overriding the _solved attribute
    model._solved = True
    # we need to set the _data attribute manually, as we are not calling the build method
    model._output_data = TestOutputData(
        location=[10, 5, 20],
        sku=[1000, 500, 100],
    )
    assert model.get_analytics(granularity="sku") == {"builder_1": [1000, 500, 100]}
    assert model.get_analytics(granularity="sku_location") == {"builder_1": "something_else"}
    assert model.get_analytics(granularity="aggregated") == {
        "builder_1": 1600,
        "ObjectiveBuilder2": 35,
    }
    assert model.get_analytics(granularity="unknown") == {}


@pytest.mark.parametrize("solver_type", [SolverType.ORTOOLS_SCIP])
def test__abstract_mip__get_analytics__analytics_data_provided(solver_type: SolverType, logger):
    """
    Tests the get_analytics method for the case when analytics_data provided as an argument for
    the method. Model is not built and therefore it does not have any data attributes.
    """

    @dataclass
    class TestModelInternalData(AbstractInternalData):
        location: list[int]
        sku: list[int]

    class TestModelOutputData(TestModelInternalData, AbstractOutputData):
        pass

    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.objective_name = "builder_1"
            self.add_analytics_granularity("sku", self._sku_analytics)
            self.add_analytics_granularity("sku_location", self._sku_location_analytics)
            self.add_analytics_granularity("aggregated", self._aggregated_analytics)

        def build(self, solver: AbstractOptimizationSolver, data: TestModelInternalData) -> None:
            pass

        def _sku_analytics(self, analytics_data: TestModelOutputData) -> Any:
            return analytics_data.sku

        def _sku_location_analytics(self, analytics_data: TestModelOutputData) -> Any:
            return "something_else"

        def _aggregated_analytics(self, analytics_data: TestModelOutputData) -> Any:
            return sum(analytics_data.sku)

    class ObjectiveBuilder2(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.add_analytics_granularity("aggregated", self.aggregated_analytics)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

        def aggregated_analytics(self, analytics_data: TestModelOutputData) -> Any:
            return sum(analytics_data.location)

    model = MockMipModel(
        solver=_construct_solver(logger=logger, solver_type=solver_type),
        constraint_builders=[],
        variable_builders=[],
        objective_builders=[ObjectiveBuilder1(logger), ObjectiveBuilder2(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger,
    )
    # we need to set the _data attribute manually, as we are not calling the build method
    analytics_data = TestModelOutputData(
        location=[10, 5, 20],
        sku=[1000, 500, 100],
    )

    assert model.get_analytics(granularity="sku", analytics_data=analytics_data) == {"builder_1": [1000, 500, 100]}
    assert model.get_analytics(granularity="sku_location", analytics_data=analytics_data) == {
        "builder_1": "something_else"
    }
    assert model.get_analytics(granularity="aggregated", analytics_data=analytics_data) == {
        "builder_1": 1600,
        "ObjectiveBuilder2": 35,
    }
    assert model.get_analytics(granularity="unknown", analytics_data=analytics_data) == {}


@pytest.mark.parametrize("solver_type", [SolverType.ORTOOLS_SCIP])
def test__abstract_mip__get_analytics__logs_warning(solver_type: SolverType):
    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

    class ObjectiveBuilder2(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

    logger = MagicMock()
    model = MockMipModel(
        solver=_construct_solver(logger=logger, solver_type=solver_type),
        constraint_builders=[],
        variable_builders=[],
        objective_builders=[ObjectiveBuilder1(logger), ObjectiveBuilder2(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger,
    )
    # we don't want to test optimization here, so we are simply overriding the _solved attribute
    model._solved = True

    model.get_analytics(granularity="aggregated")

    expected_warning_call = call(
        template="No analytics found for granularity '{granularity}' in objective builders: {objective_builders}",
        granularity="aggregated",
        objective_builders=["ObjectiveBuilder1", "ObjectiveBuilder2"],
    )

    assert model._logger.warning.call_args_list[0] == expected_warning_call


@pytest.mark.parametrize("solver_type", [SolverType.ORTOOLS_SCIP])
def test__abstract_mip__duplicate_objective_builder_names(solver_type: SolverType, logger):
    class ObjectiveBuilder1(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.objective_name = "same_as_other"

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

    class ObjectiveBuilder2(AbstractObjectiveBuilder):
        def __init__(self, logger: LoggerInterface):
            super().__init__(logger)
            self.objective_name = "same_as_other"

        def build(self, solver: AbstractOptimizationSolver, data) -> None:
            pass

    model = MockMipModel(
        solver=_construct_solver(logger=logger, solver_type=solver_type),
        constraint_builders=[],
        variable_builders=[],
        objective_builders=[ObjectiveBuilder1(logger), ObjectiveBuilder2(logger)],
        data_preparator=MockDataPreparator(logger),
        logger=logger,
    )

    with pytest.raises(ValueError, match="Duplicate objective builder name found: same_as_other"):
        model.build(input_data=MagicMock())


var1_object = object()
var2_object = object()


@dataclass
class InputTestKeepVariablesData:
    keep_variables_data: bool


@dataclass
class OutputTestKeepVariablesData:
    internal_data: dict
    internal_unpacked_data: dict


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(
            InputTestKeepVariablesData(keep_variables_data=False),
            OutputTestKeepVariablesData(
                internal_data={"var1_unpacked": 1.0, "var2_unpacked": 2.0},
                internal_unpacked_data={"var1_unpacked": 1.0, "var2_unpacked": 2.0},
            ),
            id="1) When keep_variables_data=False, internal_data and internal_unpacked_data are the same object",
        ),
        pytest.param(
            InputTestKeepVariablesData(keep_variables_data=True),
            OutputTestKeepVariablesData(
                internal_data={"var1": var1_object, "var2": var2_object},
                internal_unpacked_data={"var1_unpacked": 1.0, "var2_unpacked": 2.0},
            ),
            id="2) When keep_variables_data=True, internal_data and internal_unpacked_data are different objects",
        ),
    ],
)
def test__solve__keep_variables_data__general(
    inputs: InputTestKeepVariablesData, expected: OutputTestKeepVariablesData
):
    """
    Test method solve logic for keep_variables_data parameter:

    * 1) When keep_variables_data=False, internal_data and internal_unpacked_data reference the same object,
         and variable data is removed after unpacking.
    * 2) When keep_variables_data=True, internal_data and internal_unpacked_data are different objects,
         with variable data preserved in internal_data but removed from internal_unpacked_data after unpacking.
    """

    # Arrange
    @dataclass
    class TestInputData(AbstractInternalData):
        """
        Test input data class.
        """

        variables: dict = None

    @dataclass
    class TestInternalData(TestInputData, AbstractInternalData):
        """
        Test internal data class.
        """

    class TestVariableBuilder(AbstractDecisionVariableBuilder):
        def build(self, solver: AbstractOptimizationSolver, data: TestInternalData) -> TestInternalData:
            data.variables = {"var1": var1_object, "var2": var2_object}
            return data

        def unpack(self, solver: AbstractOptimizationSolver, data: TestInternalData) -> TestInternalData:
            data.variables = {"var1_unpacked": 1.0, "var2_unpacked": 2.0}
            return data

    logger = MagicMock(spec=LoggerInterface)
    model = MockMipModel(
        solver=_construct_solver(logger=logger),
        constraint_builders=[],
        variable_builders=[TestVariableBuilder(logger=logger)],
        objective_builders=[],
        data_preparator=MagicMock(spec=AbstractDataPreparator),
        logger=logger,
    )

    # Act
    model.build(
        input_data=TestInputData(),
    )
    model.solve(keep_variables_data=inputs.keep_variables_data)

    # Assert
    assert model._internal_data.variables == expected.internal_data
    assert model._internal_unpacked_data.variables == expected.internal_unpacked_data
