"""OR-Tools SCIP solver config types."""

from dataclasses import dataclass

from umip.solver_config.or_tools.base import OrToolsSolverConfig


@dataclass(frozen=True)
class OrToolsScipSolverConfig(OrToolsSolverConfig):
    """
    Configuration for the OR-Tools SCIP solver engine.

    See https://www.scipopt.org/doc/html/PARAMETERS.php for full docs.

    :attr numerics_epsilon: numerics/epsilon — values below this threshold are treated as zero.
        Default: 1e-09.
    :attr numerics_feastol: numerics/feastol — primal feasibility tolerance for LP and MIP constraints.
        Default: 1e-06.
    :attr numerics_dual_feastol: numerics/dualfeastol — LP dual feasibility tolerance. Default: 1e-07.
    :attr presolving_max_rounds: presolving/maxrounds — maximum presolving rounds (0: disabled,
        -1: unlimited). Default: -1.
    :attr lp_threads: lp/threads — threads used for LP solving (0: solver default). Default: 0.
    :attr parallel_max_threads: parallel/maxnthreads — maximum threads for the overall solve.
        Default: 8.
    :attr limits_gap: limits/gap — relative MIP gap stopping criterion. Default: Scip claims default is 0, but in
    practice it seems to be 1e-5.
    :attr branching_prefer_binary: branching/preferbinary — prefer branching on binary variables
        over general integers. Default: False.
    """

    numerics_epsilon: float | None = None
    numerics_feastol: float | None = None
    numerics_dual_feastol: float | None = None
    presolving_max_rounds: int | None = None
    lp_threads: int | None = None
    parallel_max_threads: int | None = None
    limits_gap: float | None = None
    branching_prefer_binary: bool | None = None

    def to_scip_parameters_string(self) -> str:
        """
        Build an OR-Tools SCIP parameter string from non-null fields.

        Examples:

        - OrToolsScipSolverConfig(numerics_epsilon=0.01)
          -> "numerics/epsilon = 0.01"
        - OrToolsScipSolverConfig(numerics_epsilon=1e-10, numerics_feastol=1e-9)
          -> "numerics/epsilon = 1e-10\\nnumerics/feastol = 1e-09"
        - OrToolsScipSolverConfig(branching_prefer_binary=True)
          -> "branching/preferbinary = TRUE"

        :return: Newline-separated SCIP parameter string ready for SetSolverSpecificParametersAsString.
        """
        setting_lines: list[str] = []
        if self.numerics_epsilon is not None:
            setting_lines.append(f"numerics/epsilon = {self.numerics_epsilon}")
        if self.numerics_feastol is not None:
            setting_lines.append(f"numerics/feastol = {self.numerics_feastol}")
        if self.numerics_dual_feastol is not None:
            setting_lines.append(f"numerics/dualfeastol = {self.numerics_dual_feastol}")
        if self.presolving_max_rounds is not None:
            setting_lines.append(f"presolving/maxrounds = {self.presolving_max_rounds}")
        if self.lp_threads is not None:
            setting_lines.append(f"lp/threads = {self.lp_threads}")
        if self.parallel_max_threads is not None:
            setting_lines.append(f"parallel/maxnthreads = {self.parallel_max_threads}")
        if self.limits_gap is not None:
            setting_lines.append(f"limits/gap = {self.limits_gap}")
        if self.branching_prefer_binary is not None:
            setting_lines.append(f"branching/preferbinary = {'TRUE' if self.branching_prefer_binary else 'FALSE'}")
        return "\n".join(setting_lines)
