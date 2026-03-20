# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Specialized AI Agents for Visualization Generation

Each agent has a distinct role within the CoDA four-phase pipeline.
"""

from .base import BaseAgent
from .query_analyzer import QueryAnalyzer
from .data_processor import DataProcessor
from .viz_mapper import VisualizationMappingAgent as VizMapper
from .search_agent import SearchAgent
from .design_explorer import DesignExplorer
from .code_generator import CodeGenerator
from .debug_agent import DebugAgent
from .visual_evaluator import VisualEvaluator

__all__ = [
    "BaseAgent",
    "QueryAnalyzer",
    "DataProcessor",
    "VizMapper",
    "SearchAgent",
    "DesignExplorer",
    "CodeGenerator",
    "DebugAgent",
    "VisualEvaluator",
]
