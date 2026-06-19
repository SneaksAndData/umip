"""LocalSolver config types."""

from dataclasses import dataclass

from generic_mip.solver_config.base import SolverConfig


@dataclass(frozen=True)
class LocalSolverConfig(SolverConfig):
    """Configuration for LocalSolver solver."""
