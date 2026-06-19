"""HiGHS solver config types."""

from dataclasses import dataclass
from enum import Enum

from generic_mip.solver_config.base import SolverConfig


class HighsPresolveOption(Enum):
    """Allowed values for HiGHS presolve option."""

    OFF = "off"
    ON = "on"
    CHOOSE = "choose"


class HighsSolverOption(Enum):
    """Allowed values for HiGHS solver option."""

    SIMPLEX = "simplex"
    IPM = "ipm"
    IPX = "ipx"
    PDLP = "pdlp"
    CHOOSE = "choose"


class HighsParallelOption(Enum):
    """Allowed values for HiGHS parallel option."""

    OFF = "off"
    ON = "on"
    CHOOSE = "choose"


@dataclass(frozen=True)
class HighsSolverConfig(SolverConfig):
    """
    Configuration for HiGHS solver.

    See https://ergo-code.github.io/HiGHS/dev/options/definitions/ for full docs.

    :attr threads: maximum number of threads. Default: 0 (automatic).
    :attr random_seed: random seed. Default: 0.
    :attr presolve: presolve mode (off/on/choose). If None, HiGHS default is used (choose).
    :attr solver: LP solver strategy (simplex/ipm/ipx/pdlp/choose). If None, HiGHS default is used (choose).
    :attr parallel: parallel strategy (off/on/choose). If None, HiGHS default is used (choose).
    """

    threads: int | None = None
    random_seed: int | None = None
    presolve: HighsPresolveOption | None = None
    solver: HighsSolverOption | None = None
    parallel: HighsParallelOption | None = None

    def to_highs_options(self) -> dict[str, bool | int | float | str]:
        """
        Build a HiGHS option dictionary from non-null fields.

        :return: Mapping of HiGHS option names to values.
        """
        option_by_name: dict[str, bool | int | float | str] = {}
        if self.threads is not None:
            option_by_name["threads"] = self.threads
        if self.random_seed is not None:
            option_by_name["random_seed"] = self.random_seed
        if self.presolve is not None:
            option_by_name["presolve"] = self.presolve.value
        if self.solver is not None:
            option_by_name["solver"] = self.solver.value
        if self.parallel is not None:
            option_by_name["parallel"] = self.parallel.value
        return option_by_name
