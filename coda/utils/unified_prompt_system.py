"""
Unified Prompt System

Provides a shared context object and a unified prompt builder that eliminates
redundant context repetition across agents in the CoDA pipeline.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json


@dataclass
class SharedContext:
    """
    Shared context that flows through all agents.
    Each agent adds to it, none repeat what's already there.
    """
    # Core request
    user_query: str
    
    # Data understanding (from Data Processor)
    data_shape: tuple = None
    key_columns: List[str] = None
    data_patterns: Dict[str, Any] = None
    
    # Design decisions (from Design Explorer)  
    chart_type: str = None
    visual_mappings: Dict[str, str] = None  # {x: 'column1', y: 'column2'}
    
    # Implementation details (from Code Generator)
    code_structure: str = None
    imports_needed: List[str] = None
    
    # Quality feedback (from Visual Evaluator)
    quality_issues: List[str] = None
    improvement_hints: List[str] = None


class UnifiedPromptBuilder:
    """
    Build focused prompts that leverage shared context.
    No repetition, no verbosity.
    """
    
    @staticmethod
    def build_agent_prompt(agent_type: str, context: SharedContext, 
                          specific_input: Any = None) -> str:
        """
        Build prompt for specific agent using shared context.
        Each agent only gets what it needs, assumes rest from context.
        """
        
        prompts = {
            "query_analyzer": UnifiedPromptBuilder._query_analyzer_prompt,
            "data_processor": UnifiedPromptBuilder._data_processor_prompt,
            "design_explorer": UnifiedPromptBuilder._design_explorer_prompt,
            "code_generator": UnifiedPromptBuilder._code_generator_prompt,
            "visual_evaluator": UnifiedPromptBuilder._visual_evaluator_prompt
        }
        
        builder = prompts.get(agent_type)
        if not builder:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        return builder(context, specific_input)
    
    @staticmethod
    def _query_analyzer_prompt(context: SharedContext, data_preview: str) -> str:
        """Query understanding - establish intent"""
        return f"""
Analyze visualization request:

REQUEST: {context.user_query}

DATA PREVIEW:
{data_preview}

Output (JSON):
{{
    "intent": "what user wants to see",
    "chart_type": "best visualization type",
    "key_columns": ["columns to use"],
    "transformations": ["data operations needed"]
}}
"""
    
    @staticmethod
    def _data_processor_prompt(context: SharedContext, data_info: Dict) -> str:
        """Data processing - prepare data"""
        return f"""
Prepare data for {context.chart_type or 'visualization'}:

INTENT: {context.user_query}
KEY COLUMNS: {context.key_columns}

DATA INFO:
{json.dumps(data_info, indent=2)}

Output (JSON):
{{
    "processing_steps": ["step1", "step2"],
    "data_ready": true/false,
    "issues": ["any problems"]
}}
"""
    
    @staticmethod
    def _design_explorer_prompt(context: SharedContext, examples: Dict = None) -> str:
        """Design decisions - visual specifications"""
        
        examples_str = ""
        if examples:
            examples_str = f"\nEXAMPLES: {json.dumps(list(examples.keys()))}"
        
        return f"""
Design {context.chart_type} visualization:

DATA: {context.data_shape} with columns {context.key_columns}
{examples_str}

Output (JSON):
{{
    "visual_mappings": {{"x": "col", "y": "col", "color": "col"}},
    "style": {{"palette": "name", "figure_size": [w, h]}},
    "annotations": ["title", "labels"]
}}
"""
    
    @staticmethod
    def _code_generator_prompt(context: SharedContext, design: Dict) -> str:
        """Code generation - implement visualization"""
        return f"""
Generate matplotlib code:

CHART: {context.chart_type}
MAPPINGS: {context.visual_mappings}
DESIGN: {json.dumps(design, indent=2)}

Requirements:
1. Clean, readable code
2. Handle edge cases
3. Professional styling

Output: Complete Python code
"""
    
    @staticmethod
    def _visual_evaluator_prompt(context: SharedContext, image_path: str) -> str:
        """Quality assessment - human perspective"""
        return f"""
Evaluate visualization quality:

INTENT: {context.user_query}
TYPE: {context.chart_type}
IMAGE: {image_path}

Assess:
1. Does it answer the question?
2. Is it visually clear?
3. Any improvements needed?

Output (JSON):
{{
    "quality_score": 0.0-1.0,
    "meets_intent": true/false,
    "issues": ["list of problems"],
    "improvements": ["suggestions"]
}}
"""


class ContextManager:
    """
    Manage shared context flow through agents.
    Each agent updates context, no redundant passing.
    """
    
    def __init__(self):
        self.context = SharedContext(user_query="")
        self.history = []
        
    def update(self, agent: str, updates: Dict[str, Any]):
        """Update context with agent's output"""
        
        # Map agent outputs to context fields
        mappings = {
            "query_analyzer": {
                "chart_type": "chart_type",
                "key_columns": "key_columns"
            },
            "data_processor": {
                "data_shape": "shape",
                "data_patterns": "patterns"
            },
            "design_explorer": {
                "visual_mappings": "visual_mappings"
            },
            "code_generator": {
                "code_structure": "structure",
                "imports_needed": "imports"
            },
            "visual_evaluator": {
                "quality_issues": "issues",
                "improvement_hints": "improvements"
            }
        }
        
        if agent in mappings:
            for context_field, update_field in mappings[agent].items():
                if update_field in updates:
                    setattr(self.context, context_field, updates[update_field])
        
        # Keep history for debugging
        self.history.append({
            "agent": agent,
            "updates": updates,
            "timestamp": str(datetime.now())
        })
    
    def get_relevant_context(self, agent: str) -> Dict[str, Any]:
        """Get only relevant context for specific agent"""
        
        relevance_map = {
            "query_analyzer": ["user_query"],
            "data_processor": ["user_query", "chart_type", "key_columns"],
            "design_explorer": ["chart_type", "data_shape", "key_columns", "data_patterns"],
            "code_generator": ["chart_type", "visual_mappings", "data_shape"],
            "visual_evaluator": ["user_query", "chart_type", "visual_mappings"]
        }
        
        relevant_fields = relevance_map.get(agent, [])
        return {
            field: getattr(self.context, field) 
            for field in relevant_fields 
            if getattr(self.context, field) is not None
        }


def demonstrate_improvement():
    """Show the difference between old and new approach"""
    
    # Old approach: Each agent gets 200+ line prompt with everything
    old_prompt_size = 200 * 5  # lines per agent * number of agents
    
    # New approach: Focused prompts with shared context
    context = SharedContext(
        user_query="Show sales by region",
        data_shape=(1000, 5),
        key_columns=["sales", "region"],
        chart_type="bar"
    )
    
    builder = UnifiedPromptBuilder()
    
    # Each agent gets focused prompt
    prompts = {
        "data_processor": builder.build_agent_prompt("data_processor", context, {"shape": (1000, 5)}),
        "design_explorer": builder.build_agent_prompt("design_explorer", context),
        "code_generator": builder.build_agent_prompt("code_generator", context, {"style": "seaborn"})
    }
    
    new_prompt_size = sum(len(p.split('\n')) for p in prompts.values())
    
    print(f"Old total prompt lines: ~{old_prompt_size}")
    print(f"New total prompt lines: {new_prompt_size}")
    print(f"Reduction: {(1 - new_prompt_size/old_prompt_size)*100:.1f}%")
    
    return prompts


# Import for backward compatibility
from datetime import datetime


if __name__ == "__main__":
    # Test the new system
    prompts = demonstrate_improvement()
    
    print("\nExample focused prompts:")
    for agent, prompt in prompts.items():
        print(f"\n{agent.upper()}:")
        print(prompt)
        print(f"Lines: {len(prompt.split(chr(10)))}")