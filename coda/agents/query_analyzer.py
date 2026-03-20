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
Query Analyzer Agent

Processes user queries and generates a structured TODO list that guides
all subsequent agents in the CoDA visualization pipeline.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .base import BaseAgent, AgentMessage

@dataclass
class QueryAnalysisResult:
    """Result structure for query analysis with global TODO list"""
    query_id: str
    original_query: str
    interpreted_intent: str
    visualization_type: str
    plotting_key_points: List[str]
    global_todo_list: List[Dict[str, Any]]  # The master TODO list for all agents
    success_criteria: List[str]
    confidence_score: float
    processing_time: float
    translated_query: str = None  # English translation for Chinese queries
    implementation_guidelines: Dict[str, Any] = None  # For expert instructions
    query_needs_expansion: bool = False  # Whether query was too simple
    expanded_query: str = None  # Expanded version of simple query
    implementation_plan: List[Dict[str, Any]] = None  # Step-by-step implementation plan
    complexity_assessment: Dict[str, Any] = None  # Complexity assessment for error handling

class QueryAnalyzer(BaseAgent):
    """
    Query Analyzer that generates structured TODO lists for the pipeline.

    Serves as the entry point for the CoDA pipeline, analyzing user queries
    and decomposing them into actionable plans for downstream agents.
    """

    def __init__(self, model_name: str, agent_id: str = "query_analyzer", config=None):
        super().__init__(agent_id, config)
        self.model_name = model_name
        self.persona = "Dr. Sarah Chen - MIT PhD in Computational Linguistics"
        self.specialization = "Query understanding, intent analysis, and task decomposition"
        
    def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryAnalysisResult:
        """
        Analyze user query and generate comprehensive TODO list for the pipeline.
        
        Args:
            query: User's visualization request
            context: Optional context information
            
        Returns:
            QueryAnalysisResult containing TODO list and analysis
        """
        start_time = datetime.now()
        
        # Check if we have expert instruction
        expert_instruction = None
        if context and context.get('expert_instruction'):
            expert_instruction = context['expert_instruction']
            self.logger.info("Using expert instruction mode")
        
        # Construct the analysis prompt
        analysis_prompt = self._build_analysis_prompt(query, context)
        
        try:
            # Use LLM reasoning for query analysis
            response = self._generate_with_usage(
                model=self.model_name,
                content=analysis_prompt,
                generation_config={"max_output_tokens": 12000}
            )
            result_text = response.text
            
            # Parse the structured response
            analysis_result = self._parse_analysis_result(result_text, query)
            
            # Track token usage
            token_usage = self._extract_token_usage(response)
            
            # Add expert instruction to implementation guidelines if provided
            if expert_instruction:
                analysis_result.implementation_guidelines = {
                    "expert_instruction": expert_instruction,
                    "use_expert_mode": True
                }
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            analysis_result.processing_time = processing_time
            
            # Track reasoning quality with token usage
            self._track_reasoning_quality(
                {"confidence": analysis_result.confidence_score, "reasoning_type": "query_analysis"},
                processing_time,
                token_usage
            )
            
            # Record behavior
            self.record_behavior(
                behavior_type="query_analysis",
                input_data={"query": query, "context": context},
                output_data=analysis_result,
                processing_time=processing_time,
                additional_metadata={"llm_response_length": len(result_text), "token_usage": token_usage}
            )
            
            self.logger.info(f"Query analysis completed in {processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {str(e)}")
            fallback_result = self._create_fallback_result(query, str(e))
            
            # Record failed behavior
            processing_time = (datetime.now() - start_time).total_seconds()
            self.record_behavior(
                behavior_type="query_analysis_failed",
                input_data={"query": query, "context": context},
                output_data=fallback_result,
                processing_time=processing_time,
                additional_metadata={"error": str(e)}
            )
            
            return fallback_result
    
    def _build_analysis_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the analysis prompt for LLM reasoning"""
        
        context_str = ""
        if context:
            context_str = f"\n\nContext Information:\n{json.dumps(context, indent=2)}"
        
        translation_section = ""
        if self._is_chinese_text(query):
            translation_section = """
STEP 1: TRANSLATION
First, translate the Chinese query to professional English for matplotlib visualization:
- Translate technical terms accurately (3D体素图 = 3D voxel plot, 箭头指示 = arrow annotation, etc.)
- Maintain the precise technical meaning
- Use standard matplotlib/scientific terminology

"""
        
        # Query expansion section - always available
        query_expansion_section = """
STEP 1: QUERY EXPANSION CHECK
First, evaluate if this query needs expansion:
- Is the query too vague or generic? 
- Does it lack specific details about data, styling, or requirements?
- Would expanding it lead to better visualization results?

If yes, expand the query with reasonable assumptions about what the user wants.

STEP 2: IMPLEMENTATION PLANNING
After query expansion, create a detailed step-by-step implementation plan:
- Break down the visualization into specific implementation steps
- Include data processing, chart creation, styling, and formatting steps
- Specify matplotlib functions and parameters to use
- Consider the logical order of operations
- Include specific code patterns and techniques needed
"""
        
        prompt = f"""
You are Dr. Sarah Chen, visualization query expert. Analyze this query and create a master TODO list.

USER QUERY: "{query}"{context_str}

{translation_section}{query_expansion_section}

Respond with concise JSON:
{{
    "query_needs_expansion": true/false,
    "expanded_query": "If needs expansion, provide detailed expanded version; otherwise same as original",
    "translated_query": "English translation if Chinese, otherwise same as expanded_query",
    "interpreted_intent": "what user wants to visualize", 
    "visualization_type": "plot type (scatter/bar/line/histogram/boxplot/heatmap etc)",
    "plotting_key_points": [
        "key point 1: specific visualization requirement",
        "key point 2: data processing requirement", 
        "key point 3: styling/design requirement",
        "key point 4: additional features/constraints"
    ],
    "implementation_plan": [
        {{"step": 1, "action": "Load and prepare data", "details": "specific data loading/processing steps", "functions": ["pd.read_csv", "etc"]}},
        {{"step": 2, "action": "Create base plot", "details": "basic chart creation", "functions": ["plt.figure", "plt.plot", "etc"]}},
        {{"step": 3, "action": "Apply formatting", "details": "styling and formatting", "functions": ["plt.xlabel", "ax.tick_params", "etc"]}},
        {{"step": 4, "action": "Finalize and save", "details": "final touches and save", "functions": ["plt.tight_layout", "plt.savefig", "etc"]}}
    ],
    "global_todo_list": [
        {{"id": "todo_1", "task": "specific task description", "agent": "data_processor|design_explorer|code_generator|debug_agent|visual_evaluator", "status": "pending", "priority": "high|medium|low"}},
        {{"id": "todo_2", "task": "specific task description", "agent": "agent_name", "status": "pending", "priority": "priority_level"}}
    ],
    "success_criteria": ["criteria for completion"],
}}

IMPORTANT: The "plotting_key_points" should be a comprehensive breakdown of ALL key visualization requirements from the query, including:
- Chart type and specific visualization style
- Data columns/variables to use
- Color schemes, styling requirements  
- Interactive elements or special features
- Layout, axis, legend requirements
- Any domain-specific requirements (scientific, business, etc.)

Create 3-5 specific TODO items covering data processing, design, code generation, debugging, and evaluation.
"""
        
        return prompt
    
    def _is_chinese_text(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        chinese_char_count = 0
        total_chars = len(text.replace(' ', ''))  # Exclude spaces
        
        for char in text:
            # Check if character is in CJK unicode ranges
            if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
                chinese_char_count += 1
        
        # Consider it Chinese if more than 30% characters are Chinese
        return total_chars > 0 and (chinese_char_count / total_chars) > 0.3
    
    def _parse_analysis_result(self, result_text: str, original_query: str) -> QueryAnalysisResult:
        """Parse the LLM response into structured analysis result"""
        
        try:
            # Extract JSON from the response
            json_match = result_text.find('{')
            if json_match == -1:
                raise ValueError("No JSON found in response")
            
            json_str = result_text[json_match:]
            # Find the end of the JSON
            bracket_count = 0
            end_pos = 0
            for i, char in enumerate(json_str):
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos == 0:
                raise ValueError("Incomplete JSON in response")
            
            json_str = json_str[:end_pos]
            parsed_result = json.loads(json_str)
            
            # Create structured result
            return QueryAnalysisResult(
                query_id=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                original_query=original_query,
                translated_query=parsed_result.get('translated_query'),
                interpreted_intent=parsed_result.get('interpreted_intent', ''),
                visualization_type=parsed_result.get('visualization_type', ''),
                plotting_key_points=parsed_result.get('plotting_key_points', []),
                global_todo_list=parsed_result.get('global_todo_list', []),
                success_criteria=parsed_result.get('success_criteria', []),
                confidence_score=float(parsed_result.get('confidence', 0.0)),
                processing_time=0.0,  # Will be set by caller
                implementation_guidelines=None,  # Will be set by caller if needed
                query_needs_expansion=parsed_result.get('query_needs_expansion', False),
                expanded_query=parsed_result.get('expanded_query'),
                implementation_plan=parsed_result.get('implementation_plan', [])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse analysis result: {str(e)}")
            return self._create_fallback_result(original_query, str(e))
    
    def _create_fallback_result(self, query: str, error_msg: str) -> QueryAnalysisResult:
        """Create a fallback result when analysis fails"""
        
        return QueryAnalysisResult(
            query_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            original_query=query,
            translated_query=query,  # Use original query as translation
            interpreted_intent="Failed to analyze query - using basic visualization",
            visualization_type="basic_plot",
            plotting_key_points=["Create basic visualization", "Use available data", "Apply standard styling"],
            global_todo_list=[
                {
                    "id": "fallback_1",
                    "task": "Process any available data",
                    "agent": "data_processor",
                    "status": "pending",
                    "priority": "high"
                },
                {
                    "id": "fallback_2", 
                    "task": "Generate basic visualization code",
                    "agent": "code_generator",
                    "status": "pending",
                    "priority": "high"
                }
            ],
            success_criteria=["Create functional visualization"],
            confidence_score=0.1,
            processing_time=0.0,
            implementation_guidelines=None,
            query_needs_expansion=False,
            expanded_query=None,
            implementation_plan=[
                {"step": 1, "action": "Basic data loading", "details": "Load and inspect data", "functions": ["pd.read_csv"]},
                {"step": 2, "action": "Basic plot creation", "details": "Create simple plot", "functions": ["plt.plot"]},
                {"step": 3, "action": "Basic formatting", "details": "Add labels", "functions": ["plt.xlabel", "plt.ylabel"]},
                {"step": 4, "action": "Save plot", "details": "Save to file", "functions": ["plt.savefig"]}
            ]
        )
    
    def validate_todo_list(self, todo_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the generated TODO list for consistency and completeness"""
        
        validation_prompt = f"""
As Dr. Sarah Chen, please validate the following TODO list for a data visualization pipeline:

TODO List: {json.dumps(todo_list, indent=2)}

Check for:
1. Task dependencies are valid and non-circular
2. All essential pipeline stages are covered
3. Priority assignments are logical
4. Success criteria are measurable
5. Estimated times are reasonable

Provide validation results in JSON format:
{{
    "is_valid": true/false,
    "issues": ["list of identified issues"],
    "suggestions": ["improvement suggestions"],
    "completeness_score": 0.95
}}
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=validation_prompt,
                generation_config={"max_output_tokens": 12000}
            )
            result_text = response.text
            
            # Parse validation result
            json_match = result_text.find('{')
            if json_match != -1:
                json_str = result_text[json_match:]
                return json.loads(json_str)
            
        except Exception as e:
            self.logger.error(f"TODO validation failed: {str(e)}")
        
        return {
            "is_valid": False,
            "issues": ["Validation failed"],
            "suggestions": ["Manual review required"],
            "completeness_score": 0.0
        }
    
    def create_agent_message(self, result: QueryAnalysisResult, target_agent: str) -> AgentMessage:
        """Create a message for the next agent in the pipeline"""
        
        return AgentMessage(
            id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender=self.agent_id,
            receiver=target_agent,
            message_type="request",
            payload={
                "query_analysis": result.__dict__,
                "instructions": f"Process according to TODO list item for {target_agent}"
            },
            timestamp=datetime.now(),
            correlation_id=result.query_id
        )