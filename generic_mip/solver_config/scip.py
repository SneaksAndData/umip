"""SCIP solver config types."""

from dataclasses import dataclass

from generic_mip.solver_config.base import SolverConfig


@dataclass(frozen=True)
class ScipSolverConfig(SolverConfig):
    """Configuration for SCIP solver."""
