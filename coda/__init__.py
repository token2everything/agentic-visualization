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
CoDA: Collaborative Data-visualization Agents

A multi-agent framework for generating publication-quality visualizations
from natural language queries. Accepted to ICLR 2026.
"""

__version__ = "1.0.0"

from .workflow import Workflow
from .agents import (
    QueryAnalyzer,
    DataProcessor,
    VizMapper,
    SearchAgent,
    DesignExplorer,
    CodeGenerator,
    DebugAgent,
    VisualEvaluator
)

# Simple API for beginners
def plot(query: str, data: str, output: str = "result.png", **kwargs):
    """
    Generate visualization from natural language query.
    
    Args:
        query: Natural language description of desired visualization
        data: Path to data file (CSV, JSON, Excel)
        output: Output file path (default: "result.png")
        **kwargs: Additional workflow parameters
        
    Returns:
        Result object with quality_score and output_file
        
    Example:
        >>> import coda
        >>> result = ap.plot(
        ...     query="Create a scatter plot of x vs y",
        ...     data="data.csv"
        ... )
        >>> print(f"Quality: {result.quality_score}")
    """
    workflow = Workflow(**kwargs)
    result = workflow.generate(query=query, data=data)
    if result.success and result.output_file:
        import shutil
        shutil.copy(result.output_file, output)
    return result

__all__ = [
    "Workflow",
    "plot",
    "QueryAnalyzer",
    "DataProcessor",
    "VizMapper",
    "SearchAgent",
    "DesignExplorer",
    "CodeGenerator",
    "DebugAgent",
    "VisualEvaluator",
]
