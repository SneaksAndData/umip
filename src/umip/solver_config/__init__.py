"""Typed solver configuration objects."""

#  Copyright (c) 2026. ECCO Data & AI and other project contributors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from umip.solver_config.base import SolverConfig as SolverConfig
from umip.solver_config.cplex import CplexSolverConfig as CplexSolverConfig
from umip.solver_config.gurobi import GurobiSolverConfig as GurobiSolverConfig
from umip.solver_config.highs import (
    HighsParallelOption as HighsParallelOption,
)
from umip.solver_config.highs import (
    HighsPresolveOption as HighsPresolveOption,
)
from umip.solver_config.highs import (
    HighsSolverConfig as HighsSolverConfig,
)
from umip.solver_config.highs import (
    HighsSolverOption as HighsSolverOption,
)
from umip.solver_config.local_solver import LocalSolverConfig as LocalSolverConfig
from umip.solver_config.or_tools.base import OrToolsSolverConfig as OrToolsSolverConfig
from umip.solver_config.or_tools.cbc import OrToolsCbcSolverConfig as OrToolsCbcSolverConfig
from umip.solver_config.or_tools.cplex import OrToolsCplexSolverConfig as OrToolsCplexSolverConfig
from umip.solver_config.or_tools.glpk import OrToolsGlpkSolverConfig as OrToolsGlpkSolverConfig
from umip.solver_config.or_tools.gurobi import OrToolsGurobiSolverConfig as OrToolsGurobiSolverConfig
from umip.solver_config.or_tools.scip import OrToolsScipSolverConfig as OrToolsScipSolverConfig
from umip.solver_config.or_tools.xpress import OrToolsXpressSolverConfig as OrToolsXpressSolverConfig
from umip.solver_config.scip import ScipSolverConfig as ScipSolverConfig
