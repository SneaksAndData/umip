"""Optimization specific exceptions."""


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


class OptimizationException(Exception):
    """A general optimization exception"""


class InfeasibleException(OptimizationException):
    """An exception for when the model is infeasible"""


class AbnormalException(OptimizationException):
    """An exception for when the model is abnormal"""


class UnboundedException(OptimizationException):
    """An exception for when the model is unbounded"""
