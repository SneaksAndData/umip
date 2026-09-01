"""OR-Tools CPLEX solver config types."""

from dataclasses import dataclass

from umip.solver_config.or_tools.base import OrToolsSolverConfig


@dataclass(frozen=True)
class OrToolsCplexSolverConfig(OrToolsSolverConfig):
    """Configuration for OR-Tools CPLEX solver."""
