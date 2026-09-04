"""Module for argument type enums."""

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

from enum import Enum


class DataFrameArgumentType(Enum):
    """Possible dataframe types."""

    POLARS = "polars"
    PANDAS = "pandas"


class BoundArgumentType(Enum):
    """Possible types for bound arguments."""

    FLOAT = "float"
    STRING = "string"
    NONE = "none"


class FilterColumnArgumentType(Enum):
    """Possible types for filter column arguments."""

    STRING = "string"
    NONE = "none"


class IndexColumnsArgumentType(Enum):
    """Possible types for index column arguments."""

    LIST_OF_STRINGS = "list_of_strings"
    NONE = "none"
