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
Code Generator Agent

Generates clean matplotlib visualization code from design specifications,
guided by search examples and the global TODO list.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import ast
import re
import numpy as np

def safe_json_dumps(obj, **kwargs):
    """Safely serialize objects to JSON, converting numpy types to Python types"""
    def convert_numpy_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    return json.dumps(convert_numpy_types(obj), **kwargs)

from .base import BaseAgent, AgentMessage
from .design_explorer import DesignExplorationResult, DesignSpecification
from .data_processor import DataProcessingResult
from .viz_mapper import VisualizationMapping

@dataclass
class CodeGenerationResult:
    """Result structure for code generation"""
    generation_id: str
    generated_code: str
    code_quality_score: float
    code_complexity: Dict[str, Any]
    dependencies: List[str]
    code_documentation: str
    error_handling: List[str]
    performance_optimizations: List[str]
    testing_suggestions: List[str]
    maintenance_notes: List[str]
    code_metrics: Dict[str, Any]
    generation_time: float

class CodeGenerator(BaseAgent):
    """
    Code Generator that produces clean matplotlib visualization code.

    Translates design specifications into maintainable code,
    guided by retrieved matplotlib examples and the TODO list.
    """

    def __init__(self, model_name: str, search_model_name: str, agent_id: str = "code_generator", config=None):
        super().__init__(agent_id, config)
        self.model_name = model_name
        self.search_model_name = search_model_name
        self.persona = "Code generation expert"
        self.specialization = "Matplotlib code generation and software architecture"
        
    def generate_code(self, design_result: DesignExplorationResult,
                     data_result: DataProcessingResult,
                     query_result: Optional['QueryAnalysisResult'] = None,
                     search_result: Optional[Dict[str, Any]] = None,
                     code_requirements: Optional[Dict[str, Any]] = None,
                     output_dir: Optional[str] = None,
                     data_file_path: Optional[str] = None,
                     workflow_context: Optional[Dict[str, Any]] = None,
                     viz_mapping: Optional['VisualizationMapping'] = None) -> CodeGenerationResult:
        """
        Generate matplotlib code based on design specifications and data.
        
        Args:
            design_result: Results from Design Explorer
            data_result: Results from Data Processor
            query_result: Results from Query Analyzer
            viz_mapping: Query-to-data visualization mapping (NEW)
            search_result: Optional matplotlib examples from search agent
            code_requirements: Optional code generation requirements
            output_dir: Optional output directory
            data_file_path: Optional path to data file for type detection
            workflow_context: Optional workflow context with file type info
            
        Returns:
            CodeGenerationResult with generated code and analysis
        """
        start_time = datetime.now()
        
        try:
            # Check TODO items for code generation
            code_todos = []
            if query_result and hasattr(query_result, 'global_todo_list'):
                code_todos = [todo for todo in query_result.global_todo_list 
                            if todo.get('agent') == 'code_generator']
            
            # Analyze code generation requirements including TODOs
            code_analysis = self._analyze_code_requirements(
                design_result, data_result, search_result, code_requirements, code_todos
            )
            
            # Generate the main visualization code
            generated_code = self._generate_main_code(
                design_result.primary_design, data_result, code_analysis, search_result, query_result, data_file_path, workflow_context, viz_mapping
            )
            
            # Validate and analyze the generated code
            code_quality_score = self._assess_code_quality(generated_code)
            code_complexity = self._analyze_code_complexity(generated_code)
            dependencies = self._extract_dependencies(generated_code)
            
            # Generate comprehensive documentation
            documentation = self._generate_code_documentation(
                generated_code, design_result, data_result
            )
            
            # Identify error handling
            error_handling = self._identify_error_handling(generated_code)
            
            # Suggest performance optimizations
            performance_opts = self._suggest_performance_optimizations(
                generated_code, data_result
            )
            
            # Generate testing suggestions
            testing_suggestions = self._generate_testing_suggestions(
                generated_code, design_result
            )
            
            # Create maintenance notes
            maintenance_notes = self._create_maintenance_notes(
                generated_code, design_result
            )
            
            # Calculate code metrics
            code_metrics = self._calculate_code_metrics(generated_code)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Track token usage (assuming we have token_usage from main generation)
            total_token_usage = getattr(self, '_latest_token_usage', {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            self._track_reasoning_quality(
                {"confidence": code_quality_score, "reasoning_type": "code_generation"},
                processing_time,
                total_token_usage
            )
            
            # Save generated code to file if output directory is provided
            code_file_path = None
            if output_dir:
                code_file_path = self._save_generated_code(generated_code, output_dir)
            
            result = CodeGenerationResult(
                generation_id=f"code_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_code=generated_code,
                code_quality_score=code_quality_score,
                code_complexity=code_complexity,
                dependencies=dependencies,
                code_documentation=documentation,
                error_handling=error_handling,
                performance_optimizations=performance_opts,
                testing_suggestions=testing_suggestions,
                maintenance_notes=maintenance_notes,
                code_metrics=code_metrics,
                generation_time=processing_time
            )
            
            # Store the file path in the result if available
            if code_file_path:
                result.code_file_path = code_file_path
            
            self.logger.info(f"Code generation completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            # NO FALLBACK - All code must be generated by API
            raise Exception(f"Code generation must use API only: {str(e)}")
    
    def _analyze_code_requirements(self, design_result: DesignExplorationResult,
                                 data_result: DataProcessingResult,
                                 search_result: Optional[Dict[str, Any]] = None,
                                 requirements: Optional[Dict[str, Any]] = None,
                                 code_todos: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze code generation requirements using LLM"""
        
        # Prepare analysis context - using FULL design specifications
        primary_design = design_result.primary_design
        context = {
            "visualization_type": primary_design.visualization_type,
            "data_shape": data_result.processed_data.shape,
            "data_columns": list(data_result.processed_data.columns),
            "design_rationale": primary_design.design_rationale,
            "color_scheme": primary_design.color_scheme,
            "layout_specifications": primary_design.layout_specifications,
            "typography": primary_design.typography,
            "implementation_guidelines": design_result.implementation_guidelines,
            "quality_metrics": design_result.quality_metrics
        }
        
        requirements_str = ""
        enhanced_fixes_str = ""
        
        if requirements:
            # Check for enhanced feedback with specific fixes
            if 'priority_fixes' in requirements:
                # Extract specific fixes from enhanced feedback
                fixes = []
                for fix in requirements.get('priority_fixes', []):
                    if hasattr(fix, '__dict__'):  # If it's an EnhancedFeedback object
                        fixes.append({
                            'issue': fix.current_problem,
                            'fix': fix.specific_fix,
                            'code': fix.code_snippet
                        })
                
                enhanced_fixes_str = f"""
CRITICAL FIXES REQUIRED (Apply these EXACTLY):

Priority Fixes:
{safe_json_dumps(fixes, indent=2)}

Code Modifications to Apply:
{safe_json_dumps(requirements.get('code_modifications', {}), indent=2)}

Figure Size Optimization:
{safe_json_dumps(requirements.get('figure_size_optimization', {}), indent=2)}

Subplot Recommendations:
{safe_json_dumps(requirements.get('subplot_recommendations', {}), indent=2)}
"""
                # Still include other requirements
                requirements_str = f"\n\nAdditional Requirements:\n{safe_json_dumps({k:v for k,v in requirements.items() if k not in ['priority_fixes', 'code_modifications', 'figure_size_optimization', 'subplot_recommendations']}, indent=2)}"
            else:
                # Standard requirements
                requirements_str = f"\n\nCode Requirements:\n{safe_json_dumps(requirements, indent=2)}"
        
        todos_str = ""
        if code_todos:
            todos_str = f"\n\nTODO Items to Complete:\n{safe_json_dumps(code_todos, indent=2)}"
        
        analysis_prompt = f"""
You are Alex Thompson, a CMU CS MS and Microsoft Engineer specializing in high-quality code generation.

Analyze the following requirements to create a CONCISE code generation plan:

Context:
{safe_json_dumps(context, indent=2)}

Design Specifications:
{safe_json_dumps(design_result.primary_design.__dict__, indent=2)}

Data Characteristics:
- Shape: {data_result.processed_data.shape}
- Columns: {list(data_result.processed_data.columns)}
- Quality Score: {data_result.data_quality_score}

{enhanced_fixes_str}{requirements_str}{todos_str}

Please provide a detailed code generation analysis in JSON format:

{{
    "code_architecture": {{
        "main_functions": ["function names and purposes"],
        "helper_functions": ["utility functions needed"],
        "class_structure": "needed classes if any",
        "modular_design": "how to structure the code"
    }},
    "matplotlib_approach": {{
        "plotting_method": "plt.subplots|plt.figure|object_oriented",
        "style_management": "rcParams|style_sheets|manual",
        "color_implementation": "colormap|manual_colors|cycler",
        "layout_strategy": "tight_layout|gridspec|constrained_layout"
    }},
    "data_handling": {{
        "data_preparation": ["preprocessing steps"],
        "data_validation": ["validation checks"],
        "error_handling": ["error scenarios to handle"],
        "performance_considerations": ["optimization strategies"]
    }},
    "code_structure": {{
        "imports": ["required imports"],
        "configuration": "setup and configuration code",
        "main_plotting": "core plotting logic",
        "customization": "styling and customization",
        "output_handling": "save and display logic"
    }},
    "quality_requirements": {{
        "code_style": "PEP8|Google|specific_style",
        "documentation_level": "minimal|standard|comprehensive",
        "error_handling_level": "basic|robust|comprehensive",
        "performance_priority": "readability|balanced|speed"
    }},
    "testing_strategy": {{
        "unit_tests": ["what to test"],
        "integration_tests": ["integration scenarios"],
        "visual_tests": ["visual validation approaches"],
        "edge_cases": ["edge cases to handle"]
    }},
    "maintenance_considerations": {{
        "code_flexibility": "how to make code adaptable",
        "documentation_needs": "what to document",
        "future_enhancements": "potential improvements",
        "dependency_management": "how to handle dependencies"
    }}
}}

Focus on creating clean, maintainable, and efficient code that accurately implements the design specifications.
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=analysis_prompt,
                generation_config={"temperature": 0.7}
            )
            result_text = response.text
            
            # Parse JSON response
            json_match = result_text.find('{')
            if json_match != -1:
                json_str = result_text[json_match:]
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > 0:
                    json_str = json_str[:end_pos]
                    return json.loads(json_str)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze code requirements: {str(e)}")
        
        # Fallback analysis
        return {
            "code_architecture": {"main_functions": ["create_visualization"]},
            "matplotlib_approach": {"plotting_method": "plt.subplots"},
            "data_handling": {"data_preparation": ["basic_validation"]},
            "code_structure": {"imports": ["matplotlib.pyplot", "pandas", "numpy"]},
            "quality_requirements": {"code_style": "PEP8"},
            "testing_strategy": {"unit_tests": ["basic_tests"]},
            "maintenance_considerations": {"code_flexibility": "moderate"}
        }
    
    def _generate_main_code(self, design: DesignSpecification,
                           data_result: DataProcessingResult,
                           code_analysis: Dict[str, Any],
                           search_result: Optional[Dict[str, Any]] = None,
                           query_result: Optional['QueryAnalysisResult'] = None,
                           data_file_path: Optional[str] = None,
                           workflow_context: Optional[Dict[str, Any]] = None,
                           viz_mapping: Optional['VisualizationMapping'] = None) -> str:
        """Generate the main matplotlib code"""
        
        # Check if this is a Plotly Sankey JSON first
        is_plotly_sankey = False
        if hasattr(data_result, 'metadata') and data_result.metadata.get('is_plotly_sankey'):
            is_plotly_sankey = True
            self.logger.info("Detected Plotly Sankey JSON data structure")
        
        # Prepare comprehensive data information
        # Prepare data info with proper serialization, handling None processed_data
        import json
        
        # Check if we have actual processed data or need to use metadata approach
        if hasattr(data_result, 'processed_data') and data_result.processed_data is not None:
            # We have actual DataFrame - use it
            data_info = {
                "columns": list(data_result.processed_data.columns),
                "shape": list(data_result.processed_data.shape),
                "dtypes": {str(k): str(v) for k, v in data_result.processed_data.dtypes.to_dict().items()},
                "sample_data": json.loads(data_result.processed_data.head(5).to_json()),
                "column_stats": self._get_column_statistics(data_result.processed_data),
                "null_counts": {str(k): int(v) for k, v in data_result.processed_data.isnull().sum().to_dict().items()},
                "is_plotly_sankey": is_plotly_sankey
            }
        else:
            # Use metadata approach when no processed_data available
            metadata = getattr(data_result, 'metadata', {})
            data_info = {
                "columns": metadata.get('columns', []),
                "shape": metadata.get('shape', [0, 0]),
                "dtypes": metadata.get('dtypes', {}),
                "sample_data": metadata.get('sample_data', []),
                "column_stats": {},
                "null_counts": metadata.get('null_counts', {}),
                "is_plotly_sankey": is_plotly_sankey,
                "note": "Using metadata approach - actual data not loaded",
                "metadata_source": True
            }
        
        # Check if we have expert instruction
        expert_instruction = ""
        if query_result and hasattr(query_result, 'implementation_guidelines'):
            self.logger.info(f"Query result implementation_guidelines type: {type(query_result.implementation_guidelines)}")
            if query_result.implementation_guidelines is not None:
                expert_instruction = query_result.implementation_guidelines.get("expert_instruction", "")
            else:
                self.logger.warning("implementation_guidelines is None")
                expert_instruction = ""
        
        # Extract data features for semantic analysis
        data_features = getattr(query_result, 'data_features', {}) if query_result else {}
        
        # Check if this is a Sankey diagram request
        is_sankey_diagram = self._detect_sankey_diagram(expert_instruction, query_result)
        # Design parameter is DesignSpecification, not DesignExplorationResult
        # So no implementation_guidelines available here
        
        # Get actual filename from metadata or workflow context - for multi-file data use real filenames
        actual_filenames = self._get_actual_filenames(data_result, workflow_context, data_file_path)
        data_loading_function, data_path_instruction = self._build_data_loading_instructions(actual_filenames)

        # Prepare matplotlib examples information (optional)
        examples_str = ""
        if search_result and isinstance(search_result, dict) and 'examples' in search_result and search_result['examples']:
            examples_dict = search_result['examples']
            if isinstance(examples_dict, dict) and examples_dict:
                examples_str = "\n\nMatplotlib Examples for Reference:\n"
                for plot_type, example in examples_dict.items():
                    if example and isinstance(example, dict):
                        examples_str += f"\n{plot_type.upper()} Example:\n"
                        examples_str += f"Source: {example.get('url', 'N/A')}\n"
                        examples_str += f"Code Reference:\n{example.get('code', 'N/A')[:500]}...\n"
                examples_str += "\nUse these examples as reference for matplotlib syntax and patterns.\n"
            else:
                examples_str = "\n\nNote: No matplotlib examples provided. Generate code using standard matplotlib practices.\n"
        else:
            examples_str = "\n\nNote: No matplotlib examples provided. Generate code using standard matplotlib practices.\n"
        
        if expert_instruction:
            # Check if this is a Sankey diagram request
            is_sankey = self._detect_sankey_diagram(expert_instruction, query_result, data_result)
            
            # All Sankey diagrams use matplotlib professional engine
            self.logger.info(f"Checking Sankey condition: is_sankey={is_sankey}, data_info type={type(data_info)}")
            # Note: Plotly Sankey handling removed - matplotlib only
            
            # Build base prompt for expert instruction
            if is_sankey:
                # Special handling for Sankey diagrams using matplotlib professional engine
                base_prompt = f"""
Convert this expert instruction into executable Python matplotlib code for creating a Sankey diagram:

EXPERT INSTRUCTION:
{expert_instruction}

DESIGN SPECIFICATIONS FROM DESIGN EXPLORER:
- Visualization Type: {design.visualization_type}
- Design Rationale: {design.design_rationale}
- Color Scheme: {safe_json_dumps(design.color_scheme, indent=2)}
- Layout Specifications: {safe_json_dumps(design.layout_specifications, indent=2)}
- Typography: {safe_json_dumps(design.typography, indent=2)}
- Canvas Size: {design.layout_specifications.get('canvas_size', {'width': 800, 'height': 600})}

{data_path_instruction}

COMPREHENSIVE DATA INFORMATION:
- Available columns: {data_info['columns']}
- Data shape: {data_info['shape']}
- Data types: {data_info['dtypes']}
- Column Statistics: {safe_json_dumps(data_info['column_stats'], indent=2)}
- Sample Data (first 5 rows):
{safe_json_dumps(data_info['sample_data'], indent=2)}
- Null counts: {data_info['null_counts']}

Additional Understanding:
- DataFrame.info summary (non-CSV context):\n{(data_result.data_summary.get('info_summary', '') if hasattr(data_result, 'data_summary') and isinstance(data_result.data_summary, dict) else '')[:1200]}
- Data attributes (YAML/SQLite/JSON hints): {safe_json_dumps((data_result.data_summary.get('attributes', {}) if hasattr(data_result, 'data_summary') and isinstance(data_result.data_summary, dict) else {}), indent=2)}

Additional Understanding:
- DataFrame.info summary (non-CSV context):\n{(data_result.data_summary.get('info_summary', '') if hasattr(data_result, 'data_summary') and isinstance(data_result.data_summary, dict) else '')[:1200]}
- Data attributes (YAML/SQLite/JSON hints): {safe_json_dumps((data_result.data_summary.get('attributes', {}) if hasattr(data_result, 'data_summary') and isinstance(data_result.data_summary, dict) else {}), indent=2)}

CRITICAL REQUIREMENTS FOR SANKEY DIAGRAM:
1. You MUST use the EXACT column names from the data above
2. You MUST handle data types correctly based on the dtypes information
3. You MUST use ONLY MATPLOTLIB for Sankey diagrams - NO PLOTLY ALLOWED
4. Ground your code in the actual data structure - do not assume column names or types
5. For source-target data, calculate counts/weights using pandas groupby
6. Create proper left-to-right layout with sources on left, targets on right
7. YOU MUST USE THE EXACT COLORS from Design Explorer color scheme above
8. INTELLIGENT SIZING: Adjust figure size based on content complexity to prevent overlaps:
   - For simple plots: Use default (8, 6)
   - For complex/multi-element plots: Scale up proportionally (e.g., 12, 8 or 14, 10)
   - For subplots: Calculate based on number of subplots (n_cols * 4, n_rows * 3)
   - Always ensure text/labels don't overlap by adjusting size or using plt.tight_layout()
9. YOU MUST FOLLOW THE TYPOGRAPHY specifications for title and labels
10. Save the plot as static image to 'result.png' using plt.savefig()

MATPLOTLIB PROFESSIONAL SANKEY TEMPLATE:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Professional Sankey Engine Template - integrated above in the prompt string

            # End of template section
```"""
            else:
                # Standard matplotlib prompt for non-Sankey diagrams
                base_prompt = f"""
Convert this expert instruction into executable Python matplotlib code:

EXPERT INSTRUCTION:
{expert_instruction}

DESIGN SPECIFICATIONS FROM DESIGN EXPLORER:
- Visualization Type: {design.visualization_type}
- Design Rationale: {design.design_rationale}
- Color Scheme: {safe_json_dumps(design.color_scheme, indent=2)}
- Layout Specifications: {safe_json_dumps(design.layout_specifications, indent=2)}
- Typography: {safe_json_dumps(design.typography, indent=2)}
- Canvas Size: {design.layout_specifications.get('canvas_size', {'width': 800, 'height': 600})}

{data_path_instruction}

COMPREHENSIVE DATA INFORMATION:
- Available columns: {data_info['columns']}
- Data shape: {data_info['shape']}
- Data types: {data_info['dtypes']}
- Column Statistics: {safe_json_dumps(data_info['column_stats'], indent=2)}
- Sample Data (first 5 rows):
{safe_json_dumps(data_info['sample_data'], indent=2)}
- Null counts: {data_info['null_counts']}

VISUALIZATION MAPPING (CRITICAL):
{self._format_viz_mapping(viz_mapping)}

CRITICAL REQUIREMENTS:
1. You MUST use the EXACT column names from the data above
2. You MUST handle data types correctly based on the dtypes information
3. You MUST use ONLY Matplotlib for plotting (no Seaborn, no Plotly, no other packages)
4. Ground your code in the actual data structure - do not assume column names or types
5. YOU MUST USE THE EXACT COLORS from Design Explorer color scheme above
6. INTELLIGENT SIZING & OVERLAP CONTROL:
   - Base size on data density and visual elements count
   - Ensure adequate space for all labels, legends, and annotations
   - Create figures with constrained_layout=True and also call fig.tight_layout() after plotting
   - Use a dynamic formula when multiple subplots or dense labels exist:
     figsize = (max(12, 3.5 * num_cols * scale_factor), max(9, 3.0 * num_rows * scale_factor))
     where scale_factor increases with long labels (len>10), many ticks (>8), or large legends
   - For crowded x-axis labels: rotate 45° and right-align
   - Place legends outside main plot: ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
   - Save tightly to avoid cut-offs: plt.savefig('result.png', bbox_inches='tight', dpi=150)
7. YOU MUST FOLLOW THE TYPOGRAPHY specifications for title and labels (use pad/labelpad for spacing)
8. Apply the Design Explorer layout specifications for margins, positioning; tune hspace/wspace when using subplots
9. MULTI-PLOT COMPOSITION:
   - If the query asks for multiple plots/panels in one figure, you MUST use subplots (no multiple figures)
   - Prefer near-square layout (e.g., 2x2, 2x3) and set hspace>=0.35, wspace>=0.3

Generate complete Python code that:
1. Follows the expert instruction exactly
2. Uses ONLY Matplotlib for all plotting (import matplotlib.pyplot as plt)
3. Uses the correct column names and handles data types properly
4. Uses the correct file path for data loading 
5. Saves the plot to 'result.png'
6. Is executable without additional input

Respond with only Python code:
```python
# Code based on expert instruction with actual data columns
```"""
            
            # Enhance prompt with semantic guidance if data features are available
            if data_features:
                code_prompt = self.integrate_semantic_guidance_into_prompt(base_prompt, query_result, data_features)
            else:
                code_prompt = base_prompt
        else:
            # Use general prompt with query details
            # Extract query details if available
            query_details = ""
            implementation_steps = ""
            
            if query_result:
                if hasattr(query_result, 'expanded_query') and query_result.expanded_query:
                    query_details = f"\nEXPANDED QUERY REQUIREMENT:\n{query_result.expanded_query}\n"
                
                if hasattr(query_result, 'implementation_plan') and query_result.implementation_plan:
                    implementation_steps = "\nIMPLEMENTATION PLAN:\n"
                    for step in query_result.implementation_plan:
                        self.logger.info(f"Processing implementation step: {type(step)}")
                        if step and isinstance(step, dict):
                            implementation_steps += f"\nStep {step.get('step', '?')}: {step.get('action', '')}\n"
                            implementation_steps += f"Details: {step.get('details', '')}\n"
                            if step.get('functions'):
                                implementation_steps += f"Functions to use: {', '.join(step.get('functions', []))}\n"
                        else:
                            self.logger.warning(f"Invalid step in implementation_plan: {step}")
                
                if hasattr(query_result, 'plotting_key_points') and query_result.plotting_key_points:
                    query_details += "\nKEY VISUALIZATION REQUIREMENTS:\n"
                    for i, point in enumerate(query_result.plotting_key_points, 1):
                        query_details += f"{i}. {point}\n"
            
            code_prompt = f"""
{query_details}{implementation_steps}

DESIGN SPECIFICATIONS FROM DESIGN EXPLORER:
- Visualization Type: {design.visualization_type}
- Design Rationale: {design.design_rationale}
- Color Scheme: {safe_json_dumps(design.color_scheme, indent=2)}
- Layout Specifications: {safe_json_dumps(design.layout_specifications, indent=2)}
- Typography: {safe_json_dumps(design.typography, indent=2)}
- Canvas Size: {design.layout_specifications.get('canvas_size', {'width': 800, 'height': 600})}

AVAILABLE DATA:
- Columns: {data_info['columns']}
- Shape: {data_info['shape']}
- Types: {data_info['dtypes']}
- Column Statistics: {safe_json_dumps(data_info['column_stats'], indent=2)}
- Sample (first 5 rows):
{safe_json_dumps(data_info['sample_data'], indent=2)}

Additional Understanding:
- DataFrame.info summary (non-CSV context):\n{(data_result.data_summary.get('info_summary', '') if hasattr(data_result, 'data_summary') and isinstance(data_result.data_summary, dict) else '')[:1200]}
- Data attributes (YAML/SQLite/JSON hints): {safe_json_dumps((data_result.data_summary.get('attributes', {}) if hasattr(data_result, 'data_summary') and isinstance(data_result.data_summary, dict) else {}), indent=2)}

DATA LOADING:
- Use: {data_loading_function}

CRITICAL REQUIREMENTS:
1. YOU MUST USE THE EXACT COLORS from Design Explorer color scheme above
2. SMART FIGURE SIZING: Intelligently set figure size based on query requirements:
   - Analyze the query for size hints (e.g., "large", "detailed", "compact")
   - Consider data volume and complexity
   - Prevent overlapping elements by increasing size when needed
   - Use a dynamic formula when multiple subplots or dense labels exist:
     figsize = (max(12, 3.5 * num_cols * scale_factor), max(9, 3.0 * num_rows * scale_factor))
     where scale_factor increases with long labels (len>10), many ticks (>8), or large legends
   - If only one plot and simple labels, default to (10, 8)
3. YOU MUST FOLLOW THE TYPOGRAPHY specifications for title and labels
4. Apply the Design Explorer layout specifications for margins, positioning
5. Use ONLY Matplotlib for all plotting (no Seaborn, no Plotly)
6. OVERLAP CONTROL (MANDATORY):
   - Create figures with constrained_layout=True (e.g., plt.subplots(..., constrained_layout=True))
   - After plotting, call fig.tight_layout()
   - Set text padding: ax.set_title(..., pad=15); ax.set_xlabel(..., labelpad=10); ax.set_ylabel(..., labelpad=10)
   - For crowded x-axis labels: rotate 45° and right-align (for lbl in ax.get_xticklabels(): lbl.set_rotation(45); lbl.set_ha('right'))
   - Place legends outside main plot: ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
   - Save tightly: plt.savefig('result.png', bbox_inches='tight', dpi=150)
7. MULTI-PLOT COMPOSITION:
   - If the query asks for multiple plots/panels in one figure, you MUST use subplots (no multiple figures)
   - Prefer near-square layout (e.g., 2x2, 2x3) and set hspace>=0.35, wspace>=0.3

Generate complete Python code that implements the visualization exactly as described above.
Save the result to 'result.png'.

```python
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=code_prompt,
                generation_config={"temperature": 0.7}
            )
            code_text = response.text
            
            # Debug: log the prompt and response
            self.logger.info(f"Code generation prompt length: {len(code_prompt)}")
            self.logger.info(f"Expert instruction mode: {bool(expert_instruction)}")
            self.logger.info(f"Response length: {len(code_text)}")
            
            # Extract code from markdown blocks
            code_blocks = re.findall(r'```python\n(.*?)\n```', code_text, re.DOTALL)
            if code_blocks:
                generated_code = code_blocks[0].strip()
                self.logger.info(f"Extracted code block, length: {len(generated_code)}")
                return generated_code
            
            # If no code blocks found, return the whole response
            self.logger.info("No code blocks found, returning whole response")
            return code_text.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate main code: {str(e)}")
            # NO FALLBACK - Force retry with API
            raise Exception(f"Code generation must use API only: {str(e)}")
    
# Fallback code generation removed - all code must be generated by API
    
    def _assess_code_quality(self, code: str) -> float:
        """Assess code quality using LLM analysis"""
        
        quality_prompt = f"""
As Alex Thompson, assess the quality of this matplotlib code:

```python
{code}
```

Evaluate the code on these criteria (0.0-1.0):
1. Code structure and organization
2. Adherence to best practices
3. Error handling
4. Documentation quality
5. Maintainability
6. Performance considerations
7. Readability
8. Matplotlib usage efficiency

Provide overall quality score as a single float value (0.0-1.0).
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=quality_prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": 100}
            )
            score_text = response.text.strip()
            
            # Extract numerical score
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Failed to assess code quality: {str(e)}")
        
        return 0.8  # Default score
    
    def _analyze_code_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity metrics"""
        
        try:
            # Basic complexity analysis
            lines = code.split('\n')
            total_lines = len(lines)
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            
            # Function count
            function_count = len(re.findall(r'def\s+\w+', code))
            
            # Import count
            import_count = len(re.findall(r'import\s+\w+|from\s+\w+\s+import', code))
            
            # Basic cyclomatic complexity estimation
            control_structures = len(re.findall(r'\b(if|for|while|try|except|with)\b', code))
            
            return {
                "total_lines": total_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "comment_ratio": comment_lines / total_lines if total_lines > 0 else 0,
                "function_count": function_count,
                "import_count": import_count,
                "estimated_cyclomatic_complexity": control_structures + 1,
                "lines_per_function": code_lines / function_count if function_count > 0 else code_lines
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze code complexity: {str(e)}")
            return {"error": str(e)}
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from the code"""
        
        dependencies = []
        
        # Find import statements
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
            r'import\s+(\w+\.\w+)',
            # Plotly removed - matplotlib only
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            dependencies.extend(matches)
        
        # Note: All plots use matplotlib only - no plotly dependencies needed
        
        # Remove duplicates and sort
        return sorted(list(set(dependencies)))
    
    def _get_column_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get detailed statistics for each column in the dataframe with semantic analysis"""
        stats = {}
        
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "semantic_type": self._infer_semantic_type(df[col], col),
                "data_pattern": self._analyze_data_pattern(df[col])
            }
            
            # For numeric columns, add statistical measures
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()) if df[col].std() != 0 else 0.0,
                    "median": float(df[col].median())
                })
                
                # Analyze distribution pattern
                col_stats["distribution_pattern"] = self._analyze_numeric_distribution(df[col])
            
            # For categorical/object columns, add category info
            elif pd.api.types.is_object_dtype(df[col]):
                value_counts = df[col].value_counts()
                col_stats.update({
                    "top_values": {str(k): int(v) for k, v in value_counts.head(5).to_dict().items()},
                    "category_count": len(value_counts)
                })
            
            stats[col] = col_stats
        
        return stats
    
    def _infer_semantic_type(self, column: pd.Series, col_name: str) -> str:
        """Infer semantic type of a column based on name and data patterns"""
        col_name_lower = col_name.lower()
        
        # Common semantic patterns
        if any(pattern in col_name_lower for pattern in ['date', 'time', 'timestamp', 'year', 'month', 'day']):
            return 'temporal'
        elif any(pattern in col_name_lower for pattern in ['price', 'cost', 'amount', 'value', 'revenue', 'salary']):
            return 'monetary' 
        elif any(pattern in col_name_lower for pattern in ['id', 'key', 'index', 'number']):
            return 'identifier'
        elif any(pattern in col_name_lower for pattern in ['name', 'title', 'label', 'category', 'type']):
            return 'categorical'
        elif any(pattern in col_name_lower for pattern in ['lat', 'lng', 'longitude', 'latitude', 'address', 'location']):
            return 'geographic'
        elif any(pattern in col_name_lower for pattern in ['count', 'quantity', 'size', 'volume', 'weight']):
            return 'measurement'
        elif col_name_lower in ['x', 'y', 'z', 'w']:
            return 'coordinate'
        else:
            # Fallback based on data type
            if pd.api.types.is_numeric_dtype(column):
                return 'numeric'
            elif pd.api.types.is_object_dtype(column):
                return 'textual'
            else:
                return 'unknown'
    
    def _analyze_data_pattern(self, column: pd.Series) -> str:
        """Analyze the data pattern in a column"""
        if pd.api.types.is_numeric_dtype(column) and len(column) > 1:
            # Check for linear, exponential, or other patterns
            sorted_vals = column.dropna().sort_values()
            if len(sorted_vals) >= 3:
                # Simple pattern detection
                diffs = sorted_vals.diff().dropna()
                if diffs.std() / (diffs.mean() + 1e-8) < 0.1:  # Low variation in differences
                    return 'linear_progression'
                elif sorted_vals.max() / (sorted_vals.min() + 1e-8) > 100:  # Large range
                    return 'exponential_like'
                else:
                    return 'irregular_numeric'
            else:
                return 'limited_numeric'
        elif pd.api.types.is_object_dtype(column):
            column_length = len(column)
            if column_length == 0:
                return 'empty_categorical'
            unique_ratio = column.nunique() / column_length
            if unique_ratio < 0.1:
                return 'low_cardinality_categorical'
            elif unique_ratio > 0.9:
                return 'high_cardinality_categorical'
            else:
                return 'medium_cardinality_categorical'
        else:
            return 'unknown_pattern'
    
    def _analyze_numeric_distribution(self, column: pd.Series) -> str:
        """Analyze the distribution pattern of numeric data"""
        if column.std() == 0:
            return 'constant'
        
        # Check skewness
        try:
            import scipy.stats as stats
            skewness = stats.skew(column.dropna())
            if abs(skewness) < 0.5:
                return 'normal_like'
            elif skewness > 1:
                return 'right_skewed'
            elif skewness < -1:
                return 'left_skewed'
            else:
                return 'moderate_skew'
        except:
            # Fallback without scipy
            median = column.median()
            mean = column.mean()
            std_val = column.std()
            if std_val == 0:
                return 'constant_values'
            if abs(mean - median) / std_val < 0.3:
                return 'approximately_normal'
            elif mean > median:
                return 'right_skewed_approx'
            else:
                return 'left_skewed_approx'
    
    def search_semantic_data_patterns(self, query_analysis: Any, data_features: Dict[str, Any]) -> str:
        """Use LLM with Google Search to find optimal data generation patterns for the query"""
        try:
            from google import genai
            from google.genai import types
            
            # Build search query based on extracted features
            search_prompt = self._build_semantic_search_prompt(query_analysis, data_features)
            
            self.logger.info("Searching for optimal data generation patterns with Google Search")
            
            response = self._generate_with_usage(
                content=search_prompt,
                model=self.search_model_name,
                generation_config={
                    "tools": [{"google_search": {}}],
                    "max_tokens": 800,
                    "temperature": 0.3
                }
            )
            
            return response.text
            
        except Exception as e:
            self.logger.warning(f"Search failed: {str(e)}, using simple guidance")
            # Simple fallback without search - just basic guidance
            return "Use linear base data with appropriate transformations for mathematical relationships. Avoid trigonometric functions when linear relationships are specified. Generate reproducible data with clear variable patterns."
    
    def _build_semantic_search_prompt(self, query_analysis: Any, data_features: Dict[str, Any]) -> str:
        """Build a targeted search prompt for data generation patterns"""
        
        query_text = getattr(query_analysis, 'original_query', 'matplotlib plotting')
        data_patterns = data_features.get('data_patterns', [])
        math_relationships = data_features.get('mathematical_relationships', [])
        variables = data_features.get('variables', [])
        
        prompt_parts = [
            f"Query analysis: '{query_text[:200]}...' if len > 200 else query_text",
            "",
            "I need to find the optimal data generation approach for matplotlib plotting. Search for:",
            ""
        ]
        
        # Build specific search targets
        if math_relationships:
            relationships_text = ', '.join([f"{y} vs {x}" for y, x in math_relationships])
            prompt_parts.append(f"1. Best practices for generating data with mathematical relationships: {relationships_text}")
            prompt_parts.append("   - Should I use linear base data for transformations?")
            prompt_parts.append("   - How to avoid trigonometric functions when linear relationships are required?")
        
        if 'linear' in data_patterns or 'linear_base' in data_patterns:
            prompt_parts.append("2. matplotlib linear data generation best practices")
            prompt_parts.append("   - Linear vs trigonometric base data for mathematical transformations")
        
        if variables:
            vars_text = ', '.join(variables)
            prompt_parts.append(f"3. Data generation patterns for variables: {vars_text}")
            prompt_parts.append("   - Standard conventions for these variable names in plotting")
        
        if 'random' in data_patterns:
            prompt_parts.append("4. Random data generation for reproducible matplotlib plots")
            prompt_parts.append("   - Best practices for seeding and distribution selection")
        
        # Add specific context for common problematic cases
        if any('z against w' in str(rel) for rel in math_relationships):
            prompt_parts.append("")
            prompt_parts.append("SPECIFIC ISSUE: Previous attempts generated trigonometric data (sin/cos) when linear")
            prompt_parts.append("relationships were required (z against w, z**3 against w, etc.).")
            prompt_parts.append("Find best practices for generating LINEAR base data for these transformations.")
        
        prompt_parts.append("")
        prompt_parts.append("Focus on:")
        prompt_parts.append("- Correct data patterns that match the mathematical relationships")
        prompt_parts.append("- Code examples showing proper numpy data generation")
        prompt_parts.append("- Common pitfalls to avoid (like using sin/cos for linear relations)")
        prompt_parts.append("- Reproducible data generation techniques")
        
        return "\n".join(prompt_parts)
    
    
    def integrate_semantic_guidance_into_prompt(self, base_prompt: str, query_analysis: Any, data_features: Dict[str, Any]) -> str:
        """Integrate semantic guidance from LLM search into the code generation prompt"""
        
        # Simple semantic guidance - direct approach, no search to avoid MAX_TOKENS
        semantic_guidance = "Use linear base data with mathematical transformations matching the query requirements. Focus on reproducible patterns with clear variable relationships."
        
        # Insert semantic guidance into the prompt
        enhanced_prompt_parts = [
            base_prompt,
            "",
            "=== SEMANTIC DATA GENERATION GUIDANCE ===",
            semantic_guidance,
            "",
            "CRITICAL: Use the above semantic guidance to generate appropriate data.",
            "The data generation must match the query semantics, not use generic templates.",
            "=== END SEMANTIC GUIDANCE ===",
            ""
        ]
        
        return "\n".join(enhanced_prompt_parts)
    
    def _detect_sankey_diagram(self, expert_instruction: str, query_result: Optional[Any] = None, 
                               data_result: Optional[DataProcessingResult] = None) -> bool:
        """Detect if the request is for a Sankey diagram"""
        
        # Check data result metadata first (most reliable)
        if data_result and hasattr(data_result, 'metadata'):
            if data_result.metadata.get('is_plotly_sankey'):
                return True
        
        # Check expert instruction for Sankey keywords
        if expert_instruction:
            sankey_keywords = ['sankey', 'flow', 'source', 'target', 'flow diagram']
            if any(keyword in expert_instruction.lower() for keyword in sankey_keywords):
                return True
        
        # Check query text if available
        if query_result and hasattr(query_result, 'original_query'):
            query_text = query_result.original_query.lower()
            sankey_keywords = ['sankey', 'flow', 'source', 'target', 'flow diagram']
            if any(keyword in query_text for keyword in sankey_keywords):
                return True
        
        return False
    
    # DEPRECATED: Plotly Sankey functions removed - matplotlib only
    # def _generate_plotly_sankey_from_json(self, data_result: DataProcessingResult, 
    #                                      expert_instruction: str = None,
    #                                      workflow_context: Dict[str, Any] = None) -> str:
    #     """DEPRECATED: Generate code based on understanding of Plotly Sankey JSON structure"""
        
        # Extract data structure information from metadata
        sankey_info = data_result.metadata.get('sankey_info', {})
        data_loading = data_result.metadata.get('data_loading_instruction', '')
        
        # Build a prompt that explains the data structure to LLM
        generation_prompt = f"""
Based on the data analysis, this is a Plotly Sankey JSON file with the following structure:
{safe_json_dumps(sankey_info, indent=2)}

Data Loading Instructions:
{data_loading}

The data has:
- {data_result.metadata.get('sankey_info', {}).get('structure', {}).get('sankey_object', {}).get('node', {}).get('label', 'unknown nodes')}
- {data_result.metadata.get('sankey_info', {}).get('structure', {}).get('sankey_object', {}).get('link', {}).get('source', 'unknown links')}

Expert Instruction (if provided):
{expert_instruction if expert_instruction else 'Create a Sankey diagram from this data'}

Generate Python code that:
1. Loads the JSON file using json.load()
2. Extracts the Sankey data from the nested structure (data[0])
3. Creates a Plotly Sankey figure using the extracted data
4. Uses the node labels, link sources/targets/values from the JSON
5. Adds appropriate colors and styling
6. Saves as 'result.png'

Return ONLY executable Python code, no explanations.
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=generation_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_tokens": 30000
                }
            )
            generated_code = response.text
            
            # Clean up the response
            if "```python" in generated_code:
                import re
                match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
                if match:
                    generated_code = match.group(1)
            
            return generated_code
            
        except Exception as e:
            self.logger.error(f"Sankey generation failed: {str(e)}")
            # No fallback - let agent handle complex Sankey requirements
            raise Exception(f"Sankey generation requires LLM analysis: {str(e)}")
    
    
    def _generate_sankey_diagram_code(self, data_info: Dict[str, Any], expert_instruction: str, 
                                    data_path_instruction: str) -> str:
        """Generate Plotly-based Sankey diagram code"""
        
        code_prompt = f"""
Generate a Sankey diagram using Plotly based on this expert instruction:

EXPERT INSTRUCTION:
{expert_instruction}

{data_path_instruction}

COMPREHENSIVE DATA INFORMATION:
- Available columns: {data_info['columns']}
- Data shape: {data_info['shape']}
- Data types: {data_info['dtypes']}
- Sample Data (first 5 rows):
{safe_json_dumps(data_info['sample_data'], indent=2)}

CRITICAL REQUIREMENTS FOR SANKEY DIAGRAM:
1. You MUST use Plotly for Sankey diagrams (not matplotlib)
2. You MUST process the data to create source-target-value relationships
3. You MUST create node indices for all unique labels
4. You MUST save the figure as 'result.png' using fig.write_image()
5. You MUST handle the data columns correctly based on the actual data structure

SANKEY DIAGRAM CODE TEMPLATE:
```python
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Load data
df = pd.read_csv('DATA_FILE_PATH')

# Process data for Sankey diagram
# If data has source-target pairs, group and count them
# If data already has weights, use them directly

# Create source-target-value structure
# Example approach:
if len(df.columns) == 2:
    # Assume first column is source, second is target
    df.columns = ['Source', 'Target']
    df_grouped = df.groupby(['Source', 'Target']).size().reset_index(name='Weight')
else:
    # Data might already have weights
    df_grouped = df

# Create unique labels and indices
all_labels = list(pd.unique(df_grouped[['Source', 'Target']].values.ravel('K')))
label_to_index = {{label: idx for idx, label in enumerate(all_labels)}}

# Map labels to indices
source_indices = [label_to_index[source] for source in df_grouped['Source']]
target_indices = [label_to_index[target] for target in df_grouped['Target']]
values = df_grouped['Weight'].tolist()

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = all_labels,
        color = "blue"
    ),
    link = dict(
        source = source_indices,
        target = target_indices,
        value = values
    )
)])

fig.update_layout(title_text="Sankey Diagram", font_size=10)

# Save as PNG
fig.write_image("result.png")
```

Generate complete Python code that:
1. Loads the data using the correct file path
2. Processes the data for Sankey diagram format
3. Creates proper source-target-value relationships
4. Generates the Sankey diagram with Plotly
5. Saves the result as 'result.png'
6. Handles the specific data structure shown above

Respond with only Python code that will execute successfully.
"""
        return code_prompt
    
    def _generate_code_documentation(self, code: str,
                                   design_result: DesignExplorationResult,
                                   data_result: DataProcessingResult) -> str:
        """Generate comprehensive code documentation"""
        
        doc_prompt = f"""
As Alex Thompson, create comprehensive documentation for this matplotlib code:

```python
{code}
```

Design Context:
- Visualization Type: {design_result.primary_design.visualization_type}
- Data Shape: {data_result.processed_data.shape}

Create documentation that includes:
1. Overview of what the code does
2. Function descriptions
3. Parameter explanations
4. Usage examples
5. Important notes and considerations

Format as markdown documentation.
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=doc_prompt,
                generation_config={
                    "temperature": 0.7
                }
            )
            return response.text
            
        except Exception as e:
            self.logger.error(f"Failed to generate documentation: {str(e)}")
            return f"Documentation generation failed: {str(e)}"
    
    def _identify_error_handling(self, code: str) -> List[str]:
        """Identify error handling mechanisms in the code"""
        
        error_handling = []
        
        # Check for try-except blocks
        if 'try:' in code and 'except' in code:
            error_handling.append("Exception handling with try-except blocks")
        
        # Check for data validation
        if 'isinstance' in code or 'len(' in code:
            error_handling.append("Data validation checks")
        
        # Check for None checks
        if 'is None' in code or 'is not None' in code:
            error_handling.append("None value checks")
        
        # Check for empty data checks
        if 'empty' in code or 'shape[0]' in code:
            error_handling.append("Empty data checks")
        
        if not error_handling:
            error_handling.append("Limited error handling detected")
        
        return error_handling
    
    def _suggest_performance_optimizations(self, code: str,
                                         data_result: DataProcessingResult) -> List[str]:
        """Suggest performance optimizations"""
        
        optimizations = []
        
        # Check data size
        if data_result.processed_data.shape[0] > 10000:
            optimizations.append("Consider data sampling for large datasets")
            optimizations.append("Use efficient plotting methods for large data")
        
        # Check for potential inefficiencies
        if 'for' in code and 'range' in code:
            optimizations.append("Review loops for vectorization opportunities")
        
        if 'append' in code:
            optimizations.append("Consider using list comprehensions instead of append")
        
        # Memory optimization
        if data_result.processed_data.memory_usage().sum() > 100000000:  # 100MB
            optimizations.append("Consider memory-efficient data types")
        
        return optimizations
    
    def _generate_testing_suggestions(self, code: str,
                                    design_result: DesignExplorationResult) -> List[str]:
        """Generate testing suggestions"""
        
        test_prompt = f"""
Suggest simple ways to verify this matplotlib code works:

```python
{code}
```

Visualization Type: {design_result.primary_design.visualization_type}

Keep it SIMPLE:
1. Just basic checks that the code runs
2. Visual check that plot looks reasonable  
3. Basic validation of outputs

Return as a JSON array of simple verification suggestion strings.
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=test_prompt,
                generation_config={
                    "temperature": 0.7
                }
            )
            result_text = response.text
            
            if '[' in result_text:
                start = result_text.find('[')
                end = result_text.rfind(']') + 1
                json_str = result_text[start:end]
                return json.loads(json_str)
            
        except Exception as e:
            self.logger.error(f"Failed to generate testing suggestions: {str(e)}")
        
        return [
            "Test with various data sizes",
            "Validate output figure properties",
            "Test error handling with invalid inputs",
            "Verify visual elements are correctly positioned",
            "Test performance with large datasets"
        ]
    
    def _create_maintenance_notes(self, code: str,
                                design_result: DesignExplorationResult) -> List[str]:
        """Create maintenance notes"""
        
        notes = []
        
        # Code structure notes
        if len(code.split('\n')) > 100:
            notes.append("Consider breaking large functions into smaller modules")
        
        # Design dependency notes
        if design_result.primary_design.design_confidence < 0.8:
            notes.append("Design confidence is low - may need future refinements")
        
        # Complexity notes
        if len(design_result.primary_design.interactive_elements) > 0:
            notes.append("Interactive elements may require additional maintenance")
        
        # General maintenance
        notes.extend([
            "Regular testing with different data formats",
            "Monitor performance with varying data sizes",
            "Update documentation as code evolves",
            "Consider user feedback for future improvements"
        ])
        
        return notes
    
    def _calculate_code_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate detailed code metrics"""
        
        lines = code.split('\n')
        
        return {
            "total_lines": len(lines),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            "function_count": len(re.findall(r'def\s+\w+', code)),
            "class_count": len(re.findall(r'class\s+\w+', code)),
            "import_count": len(re.findall(r'import|from.*import', code)),
            "complexity_indicators": {
                "if_statements": len(re.findall(r'\bif\b', code)),
                "for_loops": len(re.findall(r'\bfor\b', code)),
                "while_loops": len(re.findall(r'\bwhile\b', code)),
                "try_blocks": len(re.findall(r'\btry\b', code))
            }
        }
    
# Fallback result creation removed - all code must be generated by LLM API
    
    def _save_generated_code(self, code: str, output_dir: str) -> str:
        """Save generated code to generated_code.py file"""
        
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save code to generated_code.py
        code_file_path = output_path / "generated_code.py"
        
        with open(code_file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        self.logger.info(f"Generated code saved to: {code_file_path}")
        return str(code_file_path)
    
    def create_agent_message(self, result: CodeGenerationResult, target_agent: str) -> AgentMessage:
        """Create message for the next agent in the pipeline"""
        
        return AgentMessage(
            id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender=self.agent_id,
            receiver=target_agent,
            message_type="response",
            payload={
                "code_generation_result": result.__dict__,
                "generated_code": result.generated_code,
                "code_quality_score": result.code_quality_score
            },
            timestamp=datetime.now(),
            correlation_id=result.generation_id
        )
    
    def _get_actual_filenames(self, data_result: DataProcessingResult, 
                             workflow_context: Optional[Dict[str, Any]], 
                             data_file_path: Optional[str]) -> Dict[str, Any]:
        """Get actual filenames from metadata or workflow context for multi-file data"""
        
        # Check if we have metadata with filename information
        if hasattr(data_result, 'metadata') and data_result.metadata:
            metadata = data_result.metadata
            
            # Check for multi-file metadata
            if isinstance(metadata, dict) and metadata.get('multiple_files', False):
                files_metadata = metadata.get('files_metadata', {})
                return {
                    'multiple_files': True,
                    'files': files_metadata,
                    'primary_file': self._get_primary_data_file(files_metadata),
                    'file_list': list(files_metadata.keys())
                }

        # If workflow_context explicitly declares multi-file info, honor it
        if workflow_context and workflow_context.get('multi_files', False):
            file_list = workflow_context.get('multi_files_files', [])
            # Build simple files_metadata from extensions if available
            files_metadata = {fn: {'file_type': ('.' + fn.split('.')[-1].lower()) if '.' in fn else ''}
                              for fn in file_list}
            return {
                'multiple_files': True,
                'files': files_metadata,
                'primary_file': self._get_primary_data_file(files_metadata),
                'file_list': file_list
            }
            
        # Check workflow context for original data path
        if workflow_context:
            original_data_path = workflow_context.get('original_data_path', '')
            if original_data_path:
                import os
                filename = os.path.basename(original_data_path)
                file_ext = os.path.splitext(filename)[1].lower()
                return {
                    'multiple_files': False,
                    'primary_file': filename,
                    'file_extension': file_ext
                }
        
        # Check data_file_path as fallback
        if data_file_path:
            import os
            if os.path.isfile(data_file_path):
                filename = os.path.basename(data_file_path)
                file_ext = os.path.splitext(filename)[1].lower()
                return {
                    'multiple_files': False,
                    'primary_file': filename,
                    'file_extension': file_ext
                }
        
        # Default fallback
        return {
            'multiple_files': False,
            'primary_file': 'data.csv',
            'file_extension': '.csv'
        }
    
    def _get_primary_data_file(self, files_metadata: Dict[str, Any]) -> str:
        """Get the primary data file from multiple files"""
        
        # Prioritize data files over documentation
        data_extensions = ['.csv', '.json', '.xlsx', '.xls', '.sqlite', '.db']
        
        for filename, metadata in files_metadata.items():
            file_type = metadata.get('file_type', '').lower()
            if file_type in data_extensions:
                return filename
        
        # If no data files found, return first file
        return list(files_metadata.keys())[0] if files_metadata else 'data.csv'
    
    def _build_data_loading_instructions(self, actual_filenames: Dict[str, Any]) -> Tuple[str, str]:
        """Build data loading function and instructions using actual filenames"""
        
        if actual_filenames.get('multiple_files', False):
            # Multiple files case
            primary_file = actual_filenames.get('primary_file', 'data.csv')
            file_list = actual_filenames.get('file_list', [])
            
            # Build loading function for primary file
            file_ext = primary_file.split('.')[-1].lower()
            if file_ext == 'csv':
                data_loading_function = f"pd.read_csv('{primary_file}')"
            elif file_ext == 'json':
                data_loading_function = f"pd.read_json('{primary_file}')"
            elif file_ext in ['xlsx', 'xls']:
                data_loading_function = f"pd.read_excel('{primary_file}')"
            elif file_ext in ['sqlite', 'db']:
                data_loading_function = f"pd.read_sql('SELECT * FROM table_name', sqlite3.connect('{primary_file}'))"
            else:
                data_loading_function = f"pd.read_csv('{primary_file}')"
            
            # Build comprehensive instructions for multi-file datasets
            data_path_instruction = f"""
IMPORTANT: Multi-file data loading
This task includes multiple data files in the repository:
Available files: {', '.join(file_list)}

PRIMARY DATA FILE: {primary_file}
- Load primary data using: {data_loading_function}

ADDITIONAL FILES AVAILABLE:
"""
            for filename in file_list:
                if filename != primary_file:
                    file_ext = filename.split('.')[-1].lower()
                    if file_ext == 'csv':
                        load_func = f"pd.read_csv('{filename}')"
                    elif file_ext == 'json':
                        load_func = f"pd.read_json('{filename}')"
                    elif file_ext in ['xlsx', 'xls']:
                        load_func = f"pd.read_excel('{filename}')"
                    elif file_ext in ['sqlite', 'db']:
                        load_func = f"pd.read_sql('SELECT * FROM table_name', sqlite3.connect('{filename}'))"
                    elif file_ext in ['md', 'txt']:
                        load_func = f"# Documentation file: {filename}"
                    elif file_ext in ['yaml', 'yml']:
                        load_func = f"# Configuration file: {filename}"
                    else:
                        load_func = f"# Additional file: {filename}"
                    data_path_instruction += f"- {filename}: {load_func}\n"
            
            data_path_instruction += """
CRITICAL: Use the EXACT filenames shown above (not generic names like 'data.csv')
All files have been copied to the execution directory.
"""
            
        else:
            # Single file case
            primary_file = actual_filenames.get('primary_file', 'data.csv')
            file_ext = actual_filenames.get('file_extension', '.csv')
            
            if file_ext == '.json':
                data_loading_function = f"pd.read_json('{primary_file}')"
            elif file_ext in ['.xlsx', '.xls']:
                data_loading_function = f"pd.read_excel('{primary_file}')"
            elif file_ext in ['.sqlite', '.db']:
                data_loading_function = f"pd.read_sql('SELECT * FROM table_name', sqlite3.connect('{primary_file}'))"
            else:
                data_loading_function = f"pd.read_csv('{primary_file}')"
            
            data_path_instruction = f"""
IMPORTANT: Load data using the actual filename:
- Load data using: {data_loading_function}
- Use exact filename: '{primary_file}' (not generic 'data.csv')
- The data file has been copied to the execution directory
"""
        
        return data_loading_function, data_path_instruction
    
    def _format_viz_mapping(self, viz_mapping) -> str:
        """Format visualization mapping for LLM prompt"""
        
        if not viz_mapping:
            return "No specific visualization mapping provided - infer from query and data."
        
        mapping_text = f"""
Chart Type: {viz_mapping.chart_type}
Data Column Mappings:
{safe_json_dumps(viz_mapping.data_mappings, indent=2)}

Visualization Goal: {viz_mapping.visualization_goal}
"""
        
        if viz_mapping.aggregations:
            mapping_text += f"\nRequired Aggregations:\n{safe_json_dumps(viz_mapping.aggregations, indent=2)}"
        
        if viz_mapping.data_transformations:
            mapping_text += f"\nData Transformations:\n{safe_json_dumps(viz_mapping.data_transformations, indent=2)}"
        
        if viz_mapping.styling_hints:
            mapping_text += f"\nStyling Hints:\n{safe_json_dumps(viz_mapping.styling_hints, indent=2)}"
        
        return mapping_text
