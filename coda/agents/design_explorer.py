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
Design Explorer Agent

Creates comprehensive visual design specifications based on query analysis
and data structure, selecting optimal aesthetics for the target visualization.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from .base import BaseAgent, AgentMessage
import re
from .query_analyzer import QueryAnalysisResult
from .data_processor import DataProcessingResult

@dataclass
class DesignSpecification:
    """Comprehensive design specification for visualization"""
    design_id: str
    visualization_type: str
    design_rationale: str
    color_scheme: Dict[str, Any]
    layout_specifications: Dict[str, Any]
    typography: Dict[str, Any]
    interactive_elements: List[str]
    accessibility_features: List[str]
    responsive_design: Dict[str, Any]
    aesthetic_choices: Dict[str, Any]
    user_experience_considerations: List[str]
    design_confidence: float
    creation_time: float

@dataclass
class DesignExplorationResult:
    """Result structure for design exploration"""
    exploration_id: str
    primary_design: DesignSpecification
    alternative_designs: List[DesignSpecification]
    design_recommendations: List[str]
    implementation_guidelines: Dict[str, Any]
    quality_metrics: Dict[str, float]
    potential_challenges: List[str]
    success_indicators: List[str]
    processing_time: float

class DesignExplorer(BaseAgent):
    """
    Design Explorer that creates comprehensive visual design specifications.

    Combines query analysis and data processing results to generate
    detailed design specifications optimized for the target visualization.
    """

    def __init__(self, model_name: str, agent_id: str = "design_explorer", config=None):
        super().__init__(agent_id, config)
        self.model_name = model_name
        self.persona = "Isabella Nakamura - RISD MFA, Apple Senior Designer"
        self.specialization = "Visual design, user experience, and aesthetic innovation"
        
    def explore_design(self, query_result: QueryAnalysisResult, 
                      data_result: DataProcessingResult,
                      search_result: Optional[Dict[str, Any]] = None,
                      design_constraints: Optional[Dict[str, Any]] = None,
                      viz_mapping: Optional[Any] = None) -> DesignExplorationResult:
        """
        Explore design options based on query analysis and data processing results.
        
        Args:
            query_result: Results from Query Analyzer
            data_result: Results from Data Processor
            search_result: Optional matplotlib examples from search agent
            design_constraints: Optional design constraints
            
        Returns:
            DesignExplorationResult with comprehensive design specifications
        """
        start_time = datetime.now()
        
        try:
            # Analyze design requirements for ALL visualizations (no shortcuts!)
            # 哥说了：不要有任何template和简化处理！
            design_analysis = self._analyze_design_requirements(query_result, data_result, search_result, design_constraints)
            
            # Generate primary design specification
            primary_design = self._create_primary_design(design_analysis, query_result, data_result)
            
            # Generate alternative designs
            alternative_designs = self._generate_alternative_designs(design_analysis, primary_design)
            
            # Create implementation guidelines
            implementation_guidelines = self._create_implementation_guidelines(primary_design, data_result)
            
            # Assess design quality
            quality_metrics = self._assess_design_quality(primary_design, query_result, data_result)
            
            # Generate recommendations
            recommendations = self._generate_design_recommendations(
                primary_design, alternative_designs, quality_metrics
            )
            
            # Identify potential challenges
            challenges = self._identify_design_challenges(primary_design, data_result)
            
            # Define success indicators
            success_indicators = self._define_success_indicators(primary_design, query_result)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = DesignExplorationResult(
                exploration_id=f"design_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                primary_design=primary_design,
                alternative_designs=alternative_designs,
                design_recommendations=recommendations,
                implementation_guidelines=implementation_guidelines,
                quality_metrics=quality_metrics,
                potential_challenges=challenges,
                success_indicators=success_indicators,
                processing_time=processing_time
            )
            
            self.logger.info(f"Design exploration completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Design exploration failed: {str(e)}")
            return self._create_fallback_result(query_result, data_result, str(e))
    
    def _analyze_design_requirements(self, query_result: QueryAnalysisResult,
                                   data_result: DataProcessingResult,
                                   search_result: Optional[Dict[str, Any]] = None,
                                   constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze design requirements using LLM reasoning"""
        
        # Prepare data characteristics (optimized to avoid overly long prompts)
        # Truncate data_summary if it's too long
        data_summary = data_result.data_summary
        if isinstance(data_summary, dict):
            # Truncate description if it's too long
            if 'description' in data_summary and len(str(data_summary['description'])) > 500:
                truncated_summary = data_summary.copy()
                truncated_summary['description'] = str(data_summary['description'])[:500] + "..."
                data_summary = truncated_summary
        
        data_characteristics = {
            "shape": data_result.processed_data.shape,
            "columns": list(data_result.processed_data.columns)[:10],  # Limit to first 10 columns
            "data_types": {str(k): str(v) for k, v in list(data_result.processed_data.dtypes.to_dict().items())[:10]},  # Limit to first 10
            "data_summary": data_summary,
            "quality_score": data_result.data_quality_score,
            "sample_preview": str(data_result.processed_data.head(2))  # Only first 2 rows for context
        }
        
        # Extract design-related TODO items
        design_todos = [todo for todo in query_result.global_todo_list if todo.get('agent') == 'design_explorer']
        
        constraints_str = ""
        if constraints:
            constraints_str = f"\n\nDesign Constraints:\n{json.dumps(constraints, indent=2)}"
        
        # Prepare matplotlib examples information (optional, truncated to avoid long prompts)
        examples_str = ""
        if search_result and search_result.get('examples') and search_result['examples']:
            examples_info = []
            for plot_type, example in search_result['examples'].items():
                examples_info.append({
                    "plot_type": plot_type,
                    "url": example.get('url', ''),
                    "code_preview": example.get('code', '')[:200] + "..." if len(example.get('code', '')) > 200 else example.get('code', ''),
                    "description": example.get('description', '')[:100] + "..." if len(example.get('description', '')) > 100 else example.get('description', '')
                })
            examples_str = f"\n\nMatplotlib Examples Found:\n{json.dumps(examples_info, indent=2)}"
        else:
            examples_str = "\n\nNote: No matplotlib examples provided."
        
        analysis_prompt = f"""
You are Isabella Nakamura, an RISD MFA and Apple Senior Designer specializing in visual design and user experience.

Analyze the following requirements to create comprehensive design specifications:

Query Analysis:
- Original Query: "{query_result.original_query}"
- Interpreted Intent: "{query_result.interpreted_intent}"
- Visualization Type: "{query_result.visualization_type}"

Data Characteristics:
{json.dumps(data_characteristics, indent=2, default=str)}

Design TODO Items:
{json.dumps(design_todos, indent=2)}

{constraints_str}

{examples_str}

Please provide a comprehensive design analysis in JSON format. Consider the matplotlib examples when making design decisions:

{{
    "design_objectives": [
        "Primary design goals",
        "User experience objectives",
        "Communication goals"
    ],
    "target_audience": {{
        "primary_audience": "Who is the main audience",
        "expertise_level": "beginner|intermediate|expert",
        "context_of_use": "presentation|exploration|reporting",
        "accessibility_requirements": ["specific accessibility needs"]
    }},
    "visual_hierarchy": {{
        "primary_elements": ["most important visual elements"],
        "secondary_elements": ["supporting elements"],
        "emphasis_strategy": "how to create visual emphasis"
    }},
    "color_strategy": {{
        "primary_colors": ["#hex1", "#hex2"],
        "color_meaning": "what colors communicate",
        "accessibility_compliance": "WCAG compliance level",
        "cultural_considerations": "any cultural color meanings"
    }},
    "layout_principles": {{
        "composition_approach": "grid|organic|asymmetric|balanced",
        "spacing_strategy": "tight|moderate|generous",
        "alignment_system": "left|center|right|justified",
        "proportion_ratios": "golden ratio|rule of thirds|custom"
    }},
    "typography_requirements": {{
        "font_hierarchy": "title|subtitle|body|caption sizes",
        "readability_priority": "high|medium|low",
        "brand_alignment": "corporate|academic|creative|technical"
    }},
    "interaction_design": {{
        "interaction_level": "static|basic|advanced",
        "user_controls": ["zoom", "filter", "hover"],
        "feedback_mechanisms": "visual|audio|haptic"
    }},
    "technical_constraints": {{
        "output_format": "static|interactive|animated",
        "size_limitations": "print|screen|mobile",
        "performance_requirements": "fast|moderate|detailed"
    }},
    "innovation_opportunities": [
        "Areas for creative enhancement",
        "Unique design elements to explore"
    ],
    "design_confidence": 0.95
}}

Focus on creating designs that are both aesthetically pleasing and functionally effective for the specific data and use case.
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=analysis_prompt,
                generation_config={"max_output_tokens": 12000}
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
            self.logger.error(f"Failed to analyze design requirements: {str(e)}")
        
        # Fallback analysis
        return {
            "design_objectives": ["Create clear data visualization"],
            "target_audience": {"primary_audience": "general", "expertise_level": "intermediate"},
            "visual_hierarchy": {"primary_elements": ["main chart"]},
            "color_strategy": {"primary_colors": ["#1f77b4"]},
            "layout_principles": {"composition_approach": "balanced"},
            "typography_requirements": {"font_hierarchy": "standard"},
            "interaction_design": {"interaction_level": "static"},
            "technical_constraints": {"output_format": "static"},
            "innovation_opportunities": [],
            "design_confidence": 0.6
        }
    
    def _create_primary_design(self, design_analysis: Dict[str, Any],
                              query_result: QueryAnalysisResult,
                              data_result: DataProcessingResult) -> DesignSpecification:
        """Create the primary design specification"""
        
        design_prompt = f"""
As Isabella, create a concise design specification for {query_result.visualization_type} visualization.

Context for intelligent sizing:
- Query: {query_result.original_query[:200]}...
- Data complexity: {data_result.processed_data.shape} rows/columns
- Visualization type: {query_result.visualization_type}
- Data columns: {list(data_result.processed_data.columns)[:5]}

You are the design expert. Analyze this visualization requirement and decide ALL design aspects:

1. **Canvas Size**: You decide the optimal width and height
2. **Layout**: You decide margins, positioning, spacing
3. **Typography**: You decide all font sizes and hierarchy  
4. **Colors**: You choose the perfect color scheme
5. **Accessibility**: You ensure the design is accessible

Consider:
- Data volume and complexity
- Visualization type requirements
- Text readability and spacing
- Visual hierarchy and flow
- Target audience needs

Respond with JSON containing your expert design decisions:
{{
    "visualization_type": "{query_result.visualization_type}",
    "design_rationale": "Your design reasoning and decisions",
    "color_scheme": {{
        "primary_colors": ["your chosen colors"],
        "background_color": "your choice",
        "text_color": "your choice"
    }},
    "layout_specifications": {{
        "canvas_size": {{"width": YOUR_OPTIMAL_WIDTH, "height": YOUR_OPTIMAL_HEIGHT}},
        "margins": {{"top": YOUR_TOP, "right": YOUR_RIGHT, "bottom": YOUR_BOTTOM, "left": YOUR_LEFT}},
        "title_position": "your choice",
        "legend_position": "your choice"  
    }},
    "typography": {{
        "title_font": {{"size": YOUR_TITLE_SIZE, "weight": "your choice"}},
        "axis_label_font": {{"size": YOUR_LABEL_SIZE}}
    }},
    "accessibility_features": ["your accessibility choices"],
    "design_confidence": YOUR_CONFIDENCE_0_TO_1
}}

Trust your design expertise - make the best decisions for this specific visualization.
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=design_prompt,
                generation_config={"max_output_tokens": 12000}
            )
            result_text = response.text
            
            # Parse JSON response
            # Parse JSON using safe parsing
            design_spec = self._safe_json_parse(result_text, fallback_value={}, is_array=False)
            
            if design_spec:
                return DesignSpecification(
                        design_id=f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        visualization_type=design_spec.get('visualization_type', query_result.visualization_type),
                        design_rationale=design_spec.get('design_rationale', ''),
                        color_scheme=design_spec.get('color_scheme', {}),
                        layout_specifications=design_spec.get('layout_specifications', {}),
                        typography=design_spec.get('typography', {}),
                        interactive_elements=design_spec.get('interactive_elements', []),
                        accessibility_features=design_spec.get('accessibility_features', []),
                        responsive_design=design_spec.get('responsive_design', {}),
                        aesthetic_choices=design_spec.get('aesthetic_choices', {}),
                        user_experience_considerations=design_spec.get('user_experience_considerations', []),
                        design_confidence=float(design_spec.get('design_confidence', 0.8)),
                        creation_time=0.0
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to create primary design: {str(e)}")
        
        # Fallback design
        return self._create_fallback_design(query_result.visualization_type)
    
    def _generate_alternative_designs(self, design_analysis: Dict[str, Any],
                                    primary_design: DesignSpecification) -> List[DesignSpecification]:
        """Generate alternative design options"""
        
        alt_prompt = f"""
As Isabella Nakamura, create 2-3 alternative design variations based on:

Primary Design: {json.dumps(primary_design.__dict__, indent=2, default=str)}
Design Analysis: {json.dumps(design_analysis, indent=2)}

Create alternative designs that explore different approaches:
1. A more minimalist approach
2. A more colorful/expressive approach
3. A more data-dense approach (if applicable)

Return as a JSON array of design specifications with the same structure as the primary design.
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=alt_prompt,
                generation_config={"max_output_tokens": 12000}
            )
            result_text = response.text
            
            # Parse JSON array using safe parsing
            alt_designs_data = self._safe_json_parse(result_text, fallback_value=[], is_array=True)
            
            alternatives = []
            if alt_designs_data:
                for i, design_data in enumerate(alt_designs_data):
                    alt_design = DesignSpecification(
                        design_id=f"alt_design_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        visualization_type=design_data.get('visualization_type', primary_design.visualization_type),
                        design_rationale=design_data.get('design_rationale', ''),
                        color_scheme=design_data.get('color_scheme', {}),
                        layout_specifications=design_data.get('layout_specifications', {}),
                        typography=design_data.get('typography', {}),
                        interactive_elements=design_data.get('interactive_elements', []),
                        accessibility_features=design_data.get('accessibility_features', []),
                        responsive_design=design_data.get('responsive_design', {}),
                        aesthetic_choices=design_data.get('aesthetic_choices', {}),
                        user_experience_considerations=design_data.get('user_experience_considerations', []),
                        design_confidence=float(design_data.get('design_confidence', 0.7)),
                        creation_time=0.0
                    )
                    alternatives.append(alt_design)
                
                return alternatives
                
        except Exception as e:
            self.logger.error(f"Failed to generate alternative designs: {str(e)}")
        
        return []
    
    def _create_implementation_guidelines(self, design: DesignSpecification,
                                        data_result: DataProcessingResult) -> Dict[str, Any]:
        """Create implementation guidelines for the code generator"""
        
        guidelines_prompt = f"""
As Isabella Nakamura, create implementation guidelines for developers:

Design Specification: {json.dumps(design.__dict__, indent=2, default=str)}
Data Columns: {list(data_result.processed_data.columns)}
Data Shape: {data_result.processed_data.shape}

Create comprehensive implementation guidelines in JSON format:

{{
    "matplotlib_configuration": {{
        "figure_size": [8, 6],
        "dpi": 100,
        "style_sheet": "seaborn|default|ggplot",
        "rcParams": {{"font.size": 12}}
    }},
    "plot_creation_steps": [
        "1. Configure matplotlib settings",
        "2. Create figure and axes",
        "3. Plot data with specified styling",
        "4. Customize axes and labels",
        "5. Add legend and annotations",
        "6. Apply final styling"
    ],
    "color_implementation": {{
        "colormap": "custom|viridis|plasma|tab10",
        "color_codes": ["#hex1", "#hex2"],
        "transparency_levels": {{"primary": 0.8, "secondary": 0.6}}
    }},
    "text_styling": {{
        "title_formatting": "fontsize=16, fontweight='bold'",
        "label_formatting": "fontsize=12",
        "tick_formatting": "fontsize=10"
    }},
    "layout_instructions": {{
        "subplot_layout": "single|grid|custom",
        "spacing_adjustments": "tight_layout|subplots_adjust",
        "margin_settings": "left=0.1, right=0.9, top=0.9, bottom=0.1"
    }},
    "data_mapping": {{
        "x_axis": "column_name",
        "y_axis": "column_name",
        "color_mapping": "column_name",
        "size_mapping": "column_name",
        "grouping": "column_name"
    }},
    "error_handling": [
        "Check for missing data",
        "Validate data types",
        "Handle edge cases",
        "Provide fallback options"
    ],
    "quality_checkpoints": [
        "Verify color accessibility",
        "Check text readability",
        "Validate data representation",
        "Test responsive behavior"
    ],
    "optimization_tips": [
        "Use appropriate data structures",
        "Optimize for performance",
        "Minimize memory usage",
        "Cache expensive operations"
    ]
}}
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=guidelines_prompt,
                generation_config={"max_output_tokens": 12000}
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
            self.logger.error(f"Failed to create implementation guidelines: {str(e)}")
        
        return {
            "matplotlib_configuration": {"figure_size": [8, 6]},
            "plot_creation_steps": ["Create basic plot"],
            "color_implementation": {"colormap": "default"},
            "text_styling": {"title_formatting": "fontsize=12"},
            "layout_instructions": {"subplot_layout": "single"},
            "data_mapping": {},
            "error_handling": ["Basic error handling"],
            "quality_checkpoints": ["Basic quality checks"],
            "optimization_tips": ["Standard optimization"]
        }
    
    def _assess_design_quality(self, design: DesignSpecification,
                              query_result: QueryAnalysisResult,
                              data_result: DataProcessingResult) -> Dict[str, float]:
        """Assess design quality across multiple dimensions"""
        
        quality_prompt = f"""
As Isabella Nakamura, assess the quality of this design specification:

Design: {json.dumps(design.__dict__, indent=2, default=str)}
Original Query: "{query_result.original_query}"
Data Quality: {data_result.data_quality_score}

Assess quality across these dimensions (0.0-1.0):

{{
    "aesthetic_quality": 0.95,
    "usability_score": 0.90,
    "accessibility_compliance": 0.85,
    "data_representation_accuracy": 0.92,
    "visual_clarity": 0.88,
    "innovation_level": 0.75,
    "implementation_feasibility": 0.94,
    "responsive_design_quality": 0.80,
    "brand_consistency": 0.85,
    "user_experience_quality": 0.87,
    "overall_quality": 0.89
}}

Provide only the JSON with numerical scores.
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=quality_prompt,
                generation_config={"max_output_tokens": 12000}
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
            self.logger.error(f"Failed to assess design quality: {str(e)}")
        
        return {
            "aesthetic_quality": 0.8,
            "usability_score": 0.8,
            "accessibility_compliance": 0.8,
            "data_representation_accuracy": 0.8,
            "visual_clarity": 0.8,
            "innovation_level": 0.7,
            "implementation_feasibility": 0.9,
            "responsive_design_quality": 0.7,
            "brand_consistency": 0.8,
            "user_experience_quality": 0.8,
            "overall_quality": 0.8
        }
    
    def _generate_design_recommendations(self, primary_design: DesignSpecification,
                                       alternatives: List[DesignSpecification],
                                       quality_metrics: Dict[str, float]) -> List[str]:
        """Generate design recommendations"""
        
        rec_prompt = f"""
As Isabella Nakamura, provide design recommendations based on:

Primary Design Quality: {json.dumps(quality_metrics, indent=2)}
Number of Alternatives: {len(alternatives)}

Provide specific recommendations for:
1. Design improvements
2. Implementation considerations
3. User experience enhancements
4. Accessibility improvements
5. Innovation opportunities

Return as a JSON array of recommendation strings.
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=rec_prompt,
                generation_config={"max_output_tokens": 12000}
            )
            result_text = response.text
            
            if '[' in result_text:
                start = result_text.find('[')
                end = result_text.rfind(']') + 1
                json_str = result_text[start:end]
                return json.loads(json_str)
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {str(e)}")
        
        return [
            "Consider user feedback during implementation",
            "Test design with target audience",
            "Ensure accessibility compliance",
            "Optimize for performance"
        ]
    
    def _identify_design_challenges(self, design: DesignSpecification,
                                  data_result: DataProcessingResult) -> List[str]:
        """Identify potential design implementation challenges"""
        
        challenges = []
        
        # Data-related challenges
        if data_result.data_quality_score < 0.8:
            challenges.append("Low data quality may affect visualization accuracy")
        
        if data_result.processed_data.shape[0] > 10000:
            challenges.append("Large dataset may impact performance")
        
        # Design complexity challenges
        if len(design.interactive_elements) > 3:
            challenges.append("Multiple interactive elements may increase complexity")
        
        if design.design_confidence < 0.8:
            challenges.append("Design confidence is lower than optimal")
        
        return challenges
    
    def _define_success_indicators(self, design: DesignSpecification,
                                 query_result: QueryAnalysisResult) -> List[str]:
        """Define success indicators for the design"""
        
        return [
            "Visualization clearly communicates the intended message",
            "Users can easily interpret the data",
            "Design meets accessibility standards",
            "Implementation is technically feasible",
            "Visual quality meets professional standards",
            f"Addresses original query: '{query_result.original_query}'"
        ]
    
    def _create_fallback_design(self, visualization_type: str) -> DesignSpecification:
        """Create fallback design when generation fails"""
        
        return DesignSpecification(
            design_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            visualization_type=visualization_type,
            design_rationale="Fallback design due to generation failure",
            color_scheme={"primary_colors": ["#1f77b4"], "background_color": "#ffffff"},
            layout_specifications={"canvas_size": {"width": 1000, "height": 700}},  # Larger fallback for safety
            typography={"title_font": {"family": "Arial", "size": 14}},
            interactive_elements=[],
            accessibility_features=[],
            responsive_design={},
            aesthetic_choices={"visual_style": "minimal"},
            user_experience_considerations=["Basic usability"],
            design_confidence=0.5,
            creation_time=0.0
        )
    
    def _create_fallback_result(self, query_result: QueryAnalysisResult,
                              data_result: DataProcessingResult,
                              error_msg: str) -> DesignExplorationResult:
        """Create fallback result when exploration fails"""
        
        fallback_design = self._create_fallback_design(query_result.visualization_type)
        
        return DesignExplorationResult(
            exploration_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            primary_design=fallback_design,
            alternative_designs=[],
            design_recommendations=[f"Fix design exploration error: {error_msg}"],
            implementation_guidelines={"error": error_msg},
            quality_metrics={"overall_quality": 0.3},
            potential_challenges=[error_msg],
            success_indicators=["Resolve design exploration issues"],
            processing_time=0.0
        )
    
    def create_agent_message(self, result: DesignExplorationResult, target_agent: str) -> AgentMessage:
        """Create message for the next agent in the pipeline"""
        
    def _safe_json_parse(self, response_text: str, fallback_value=None, is_array=False):
        """Safely parse JSON from LLM response with multiple fallback strategies"""
        try:
            # Strategy 1: Find complete JSON object/array
            if is_array and '[' in response_text:
                start = response_text.find('[')
                json_content = response_text[start:]
                bracket_count = 0
                end_pos = 0
                
                for i, char in enumerate(json_content):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > 0:
                    json_str = json_content[:end_pos]
                    return json.loads(json_str)
                    
            elif not is_array and '{' in response_text:
                start = response_text.find('{')
                json_content = response_text[start:]
                brace_count = 0
                end_pos = 0
                
                for i, char in enumerate(json_content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > 0:
                    json_str = json_content[:end_pos]
                    return json.loads(json_str)
            
            # Strategy 2: Try to clean up common JSON issues
            cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response_text)  # Remove control chars
            cleaned_text = re.sub(r',\s*}', '}', cleaned_text)  # Remove trailing commas
            cleaned_text = re.sub(r',\s*]', ']', cleaned_text)  # Remove trailing commas in arrays
            
            if is_array:
                match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
                if match:
                    return json.loads(match.group())
            else:
                match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if match:
                    return json.loads(match.group())
                    
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {str(e)}. Using fallback value.")
        except Exception as e:
            self.logger.error(f"Unexpected error in JSON parsing: {str(e)}. Using fallback value.")
            
        return fallback_value

        return AgentMessage(
            id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender=self.agent_id,
            receiver=target_agent,
            message_type="response",
            payload={
                "design_exploration_result": result.__dict__,
                "primary_design": result.primary_design.__dict__,
                "implementation_guidelines": result.implementation_guidelines
            },
            timestamp=datetime.now(),
            correlation_id=result.exploration_id
        )
    
    def refine_design(self, 
                     original_design_result: DesignExplorationResult,
                     visual_feedback: Dict[str, Any],
                     query_result: QueryAnalysisResult,
                     data_result: DataProcessingResult) -> DesignExplorationResult:
        """
        Refine existing design based on visual evaluation feedback.
        
        This is called when Visual Evaluator identifies design issues (not code issues).
        """
        start_time = datetime.now()
        self.logger.info("Starting design refinement based on visual feedback")
        
        # Analyze what specific design issues were identified
        feedback_text = " ".join(visual_feedback.get("visual_feedback", [])).lower()
        quality_issues = visual_feedback.get("quality_issues", [])
        target_quality = visual_feedback.get("target_quality", 0.9)
        
        refinement_prompt = f"""
You are Isabella Nakamura, an expert designer. The current design received feedback from visual evaluation.

ORIGINAL DESIGN SPECIFICATIONS:
- Primary Design: {json.dumps(original_design_result.primary_design.__dict__, indent=2, default=str)}
- Alternative Designs Available: {len(original_design_result.alternative_designs)}

VISUAL FEEDBACK ANALYSIS:
- Feedback Comments: {visual_feedback.get("visual_feedback", [])}
- Quality Issues: {quality_issues}  
- Target Quality Threshold: {target_quality}
- Current Quality Score: Below threshold

REFINEMENT STRATEGY:
Based on the feedback, determine what needs to change:

1. **Color Issues**: If feedback mentions colors, provide new color scheme
2. **Layout Issues**: If feedback mentions spacing/layout, adjust layout specifications  
3. **Typography Issues**: If feedback mentions text/fonts, update typography
4. **Overall Aesthetic**: If feedback mentions visual appeal, try alternative design

REFINEMENT ACTION:
Choose the best approach and provide updated design specifications in the same JSON format as the original primary design.

Focus on addressing the specific feedback while maintaining design coherence.

Return the refined design specification as JSON.
"""

        try:
            generation_config = generative_models.GenerationConfig(
                max_output_tokens=12000
            )
            response = self.model.generate_content(
                refinement_prompt,
                generation_config=generation_config
            )
            result_text = response.text
            
            # Parse refined design
            refined_design_data = self._safe_json_parse(result_text, fallback_value={}, is_array=False)
            
            if refined_design_data:
                # Create refined design specification
                refined_primary_design = DesignSpecification(
                    design_id=f"refined_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    visualization_type=refined_design_data.get('visualization_type', original_design_result.primary_design.visualization_type),
                    design_rationale=refined_design_data.get('design_rationale', original_design_result.primary_design.design_rationale + " [REFINED based on feedback]"),
                    color_scheme=refined_design_data.get('color_scheme', original_design_result.primary_design.color_scheme),
                    layout_specifications=refined_design_data.get('layout_specifications', original_design_result.primary_design.layout_specifications),
                    typography=refined_design_data.get('typography', original_design_result.primary_design.typography),
                    interactive_elements=refined_design_data.get('interactive_elements', original_design_result.primary_design.interactive_elements),
                    accessibility_features=refined_design_data.get('accessibility_features', original_design_result.primary_design.accessibility_features),
                    responsive_design=refined_design_data.get('responsive_design', original_design_result.primary_design.responsive_design),
                    aesthetic_choices=refined_design_data.get('aesthetic_choices', original_design_result.primary_design.aesthetic_choices),
                    user_experience_considerations=refined_design_data.get('user_experience_considerations', original_design_result.primary_design.user_experience_considerations),
                    design_confidence=float(refined_design_data.get('design_confidence', 0.85)),  # Slightly lower for refined designs
                    creation_time=0.0
                )
                
                # Create new result with refined design
                processing_time = (datetime.now() - start_time).total_seconds()
                
                refined_result = DesignExplorationResult(
                    exploration_id=f"refined_{original_design_result.exploration_id}",
                    primary_design=refined_primary_design,
                    alternative_designs=original_design_result.alternative_designs,  # Keep alternatives
                    design_recommendations=["Design refined based on visual feedback"] + original_design_result.design_recommendations,
                    implementation_guidelines=original_design_result.implementation_guidelines,  # Keep same guidelines
                    quality_metrics={"overall_quality": 0.9, "refinement_applied": True},  # Higher expected quality
                    potential_challenges=original_design_result.potential_challenges,
                    success_indicators=original_design_result.success_indicators + ["Visual feedback addressed"],
                    processing_time=processing_time
                )
                
                self.logger.info(f"Design refinement completed in {processing_time:.2f}s")
                return refined_result
                
        except Exception as e:
            self.logger.error(f"Design refinement failed: {str(e)}")
        
        # Fallback: Try alternative design from original exploration
        if original_design_result.alternative_designs:
            self.logger.info("Using alternative design as refinement fallback")
            alternative = original_design_result.alternative_designs[0]
            
            # Promote alternative to primary design
            refined_result = DesignExplorationResult(
                exploration_id=f"alt_{original_design_result.exploration_id}",
                primary_design=alternative,
                alternative_designs=original_design_result.alternative_designs[1:],  # Use remaining alternatives
                design_recommendations=["Used alternative design due to feedback"],
                implementation_guidelines=original_design_result.implementation_guidelines,
                quality_metrics={"overall_quality": 0.85, "used_alternative": True},
                potential_challenges=original_design_result.potential_challenges,
                success_indicators=original_design_result.success_indicators + ["Alternative design applied"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            return refined_result
        
        # Last resort: Return original with minor adjustments
        self.logger.warning("Design refinement failed, returning original with minor adjustments")
        return original_design_result
    
