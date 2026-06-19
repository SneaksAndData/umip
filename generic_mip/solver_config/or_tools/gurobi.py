"""OR-Tools Gurobi solver config types."""

from dataclasses import dataclass

from generic_mip.solver_config.or_tools.base import OrToolsSolverConfig


@dataclass(frozen=True)
class OrToolsGurobiSolverConfig(OrToolsSolverConfig):
    """Configuration for OR-Tools Gurobi solver."""
