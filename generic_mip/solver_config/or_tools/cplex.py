"""OR-Tools CPLEX solver config types."""

from dataclasses import dataclass

from generic_mip.solver_config.or_tools.base import OrToolsSolverConfig


@dataclass(frozen=True)
class OrToolsCplexSolverConfig(OrToolsSolverConfig):
    """Configuration for OR-Tools CPLEX solver."""
