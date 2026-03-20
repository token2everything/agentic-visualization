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
Visualization Mapping Agent - Maps Query Intent to Data Columns

This agent solves the core problem: Given a query and data, 
determine exactly how to map data columns to visualization elements.
"""

import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .base import BaseAgent

@dataclass
class VisualizationMapping:
    """Complete mapping from query intent to data visualization"""
    mapping_id: str
    chart_type: str                     # bar, line, scatter, pie, etc.
    data_mappings: Dict[str, str]       # role -> column_name
    aggregations: List[Dict[str, Any]]  # required aggregations
    filters: List[Dict[str, Any]]       # data filters needed
    styling_hints: Dict[str, Any]       # colors, labels, etc.
    data_transformations: List[str]     # pandas operations needed
    visualization_goal: str             # what story to tell
    token_usage: Dict[str, int]         # LLM token usage
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'mapping_id': self.mapping_id,
            'chart_type': self.chart_type,
            'data_mappings': self.data_mappings,
            'aggregations': self.aggregations,
            'filters': self.filters,
            'styling_hints': self.styling_hints,
            'data_transformations': self.data_transformations,
            'visualization_goal': self.visualization_goal,
            'token_usage': self.token_usage,
            'processing_time': self.processing_time
        }

class VisualizationMappingAgent(BaseAgent):
    """
    Maps user query intent to specific data column assignments.
    
    Solves: "Show sales by region" + data['region', 'sales', 'date'] 
    → x_axis='region', y_axis='sales', chart_type='bar'
    """
    
    def __init__(self, model_name: str, agent_id: str = "viz_mapping_v2", config=None):
        super().__init__(agent_id, config)
        self.model_name = model_name
        self.persona = "Dr. Sarah Kim - Data Visualization Expert & UX Designer"
        self.specialization = "Query-to-visualization mapping, data storytelling"
    
    def map_query_to_visualization(self, 
                                 query: str, 
                                 data: pd.DataFrame,
                                 query_context: Optional[Dict[str, Any]] = None) -> VisualizationMapping:
        """
        Map user query to specific visualization instructions.
        
        Args:
            query: User's visualization request
            data: The actual data DataFrame
            query_context: Additional context from query analyzer
            
        Returns:
            Complete visualization mapping instructions
        """
        start_time = datetime.now()
        
        # Generate data structure summary
        data_summary = self._generate_data_summary(data)
        
        # Create mapping prompt
        mapping_prompt = self._build_mapping_prompt(query, data_summary, query_context)
        
        try:
            # Get LLM mapping decision
            response = self._generate_with_usage(
                model=self.model_name,
                content=mapping_prompt, 
                generation_config={
                    "max_output_tokens": 8000,
                    "temperature": 0.3
                }
            )
            result_text = response.text
            
            # Track token usage
            token_usage = response.usage_metadata if hasattr(response, 'usage_metadata') else {}
            
            # Parse mapping result
            mapping_result = self._parse_mapping_result(result_text)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create final mapping
            visualization_mapping = VisualizationMapping(
                mapping_id=f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                chart_type=mapping_result['chart_type'],
                data_mappings=mapping_result['data_mappings'],
                aggregations=mapping_result.get('aggregations', []),
                filters=mapping_result.get('filters', []),
                styling_hints=mapping_result.get('styling_hints', {}),
                data_transformations=mapping_result.get('transformations', []),
                visualization_goal=mapping_result.get('goal', ''),
                token_usage=token_usage,
                processing_time=processing_time
            )
            
            # Track reasoning quality
            self._track_reasoning_quality(
                {"confidence": 0.9, "reasoning_type": "visualization_mapping"},
                processing_time,
                token_usage
            )
            
            self.logger.info(f"Visualization mapping completed in {processing_time:.2f}s")
            self.logger.info(f"Mapped to {mapping_result['chart_type']} with {len(mapping_result['data_mappings'])} column mappings")
            
            return visualization_mapping
            
        except Exception as e:
            self.logger.error(f"Visualization mapping failed: {str(e)}")
            return self._create_fallback_mapping(query, data)
    
    def _generate_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate concise data structure summary"""
        
        # Basic structure
        summary = {
            "shape": list(data.shape),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
        
        # Sample data (first 3 rows)
        summary["sample_data"] = data.head(3).to_dict('records')
        
        # Column analysis
        summary["column_analysis"] = {}
        for col in data.columns:
            col_info = {
                "type": str(data[col].dtype),
                "unique_count": data[col].nunique(),
                "null_count": data[col].isnull().sum()
            }
            
            # Add value examples
            if data[col].dtype in ['object', 'string']:
                col_info["sample_values"] = data[col].dropna().unique()[:5].tolist()
            elif data[col].dtype in ['int64', 'float64']:
                col_info["range"] = [float(data[col].min()), float(data[col].max())]
            
            summary["column_analysis"][col] = col_info
            
        return summary
    
    def _build_mapping_prompt(self, query: str, data_summary: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Build the LLM prompt for query-to-data mapping"""
        
        # Format data structure clearly
        columns_info = []
        for col, info in data_summary["column_analysis"].items():
            if info["type"] in ["object", "string"]:
                sample_vals = info.get("sample_values", [])
                columns_info.append(f"- {col}: {info['type']}, {info['unique_count']} unique values, examples: {sample_vals}")
            else:
                range_info = info.get("range", ["N/A", "N/A"])
                columns_info.append(f"- {col}: {info['type']}, range: {range_info[0]} to {range_info[1]}")
        
        data_structure = "\\n".join(columns_info)
        
        prompt = f"""
You are a data visualization expert. Map this user query to specific data columns and chart configuration.

USER QUERY: "{query}"

AVAILABLE DATA:
Shape: {data_summary['shape'][0]} rows × {data_summary['shape'][1]} columns
Columns:
{data_structure}

Sample data:
{json.dumps(data_summary['sample_data'][:2], indent=2)}

TASK: Determine the optimal visualization mapping.

Respond with JSON:
{{
    "chart_type": "bar|line|scatter|pie|histogram|box|heatmap",
    "data_mappings": {{
        "x_axis": "column_name_for_x",
        "y_axis": "column_name_for_y", 
        "color": "column_for_grouping_colors",
        "size": "column_for_sizes",
        "category": "column_for_categories"
    }},
    "aggregations": [
        {{"operation": "sum|mean|count|max|min", "column": "column_name", "group_by": "grouping_column"}}
    ],
    "filters": [
        {{"column": "column_name", "condition": "filter_condition"}}
    ],
    "styling_hints": {{
        "title": "Chart title based on query",
        "xlabel": "X-axis label", 
        "ylabel": "Y-axis label",
        "color_palette": "suggested_palette"
    }},
    "transformations": [
        "pandas operation if needed, e.g., 'df.groupby(x).sum()'"
    ],
    "goal": "Brief description of what this visualization shows"
}}

IMPORTANT:
- Only include data_mappings keys that are actually needed for this chart type
- Choose the chart type that best answers the user's question
- If aggregation is needed, specify exactly how
- Be precise with column names - they must match the available columns exactly
"""
        return prompt
    
    def _parse_mapping_result(self, result_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured mapping"""
        
        try:
            # Extract JSON from response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = result_text[json_start:json_end]
                mapping_result = json.loads(json_text)
                
                # Validate required fields
                required_fields = ['chart_type', 'data_mappings']
                for field in required_fields:
                    if field not in mapping_result:
                        raise ValueError(f"Missing required field: {field}")
                
                return mapping_result
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse mapping result: {str(e)}")
            self.logger.error(f"Response text: {result_text[:500]}...")
            
            # Return minimal fallback
            return {
                "chart_type": "bar",
                "data_mappings": {},
                "aggregations": [],
                "filters": [],
                "styling_hints": {},
                "transformations": [],
                "goal": "Fallback visualization"
            }
    
    def _create_fallback_mapping(self, query: str, data: pd.DataFrame) -> VisualizationMapping:
        """Create basic fallback mapping when LLM fails"""
        
        # Simple heuristics for fallback
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Basic mapping logic
        data_mappings = {}
        chart_type = "bar"
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            data_mappings['x_axis'] = categorical_cols[0]
            data_mappings['y_axis'] = numeric_cols[0]
            chart_type = "bar"
        elif len(numeric_cols) >= 2:
            data_mappings['x_axis'] = numeric_cols[0]  
            data_mappings['y_axis'] = numeric_cols[1]
            chart_type = "scatter"
        
        return VisualizationMapping(
            mapping_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            chart_type=chart_type,
            data_mappings=data_mappings,
            aggregations=[],
            filters=[],
            styling_hints={"title": "Fallback Visualization"},
            data_transformations=[],
            visualization_goal="Basic fallback visualization",
            token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            processing_time=0.0
        )
