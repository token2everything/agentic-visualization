"""
BaseAgent: LLM-Based Reasoning Foundation for All Agents

Provides the LLM-powered reasoning interface and shared behaviors
used by all agents in the CoDA pipeline.
"""

import logging
import time
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
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

import litellm


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""
    id: str
    sender: str
    receiver: str
    message_type: str  # "request", "response", "feedback", "error"
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: str

class BaseAgent:
    """
    LLM-Powered Base Agent with Autonomous Reasoning Capabilities
    
    Every decision, analysis, and action is driven by LLM reasoning rather than hardcoded logic.
    This ensures genuine AI-powered behavior and adaptability.
    """
    
    def __init__(self, agent_id: str, config=None):
        self.agent_id = agent_id
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")

        # Initialize behavior tracking
        self.behavior_history = []
        self.current_session_id = None
        
        # Initialize persona through LLM reasoning
        self.persona = self._initialize_persona()
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "error_count": 0,
            "last_activity": None,
            "token_usage_log": []
        }
        
        # Agent state
        self.is_initialized = True
        self.last_request_time = None
        self.reasoning_history = []  # Track LLM reasoning decisions
        
        if hasattr(self.persona, 'get') and 'name' in self.persona:
            self.logger.info(f"LLM-powered Agent {agent_id} initialized: {self.persona['name']}")
        else:
            self.logger.info(f"LLM-powered Agent {agent_id} initialized")
    
    def _initialize_persona(self) -> Dict[str, Any]:
        """Initialize agent's professional persona through LLM reasoning - default implementation"""
        return {
            "name": f"Agent {self.agent_id}",
            "title": "AI Assistant",
            "expertise": ["general_assistance"],
            "personality_traits": ["helpful", "analytical"],
            "background": "General AI assistant"
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main LLM-powered processing method - default implementation"""
        return {
            "status": "processed",
            "agent_id": self.agent_id,
            "message": "Base processing completed"
        }
    
    def llm_reason_and_decide(self, model: str, reasoning_prompt: str, decision_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Core LLM reasoning method - all decisions go through this
        
        Args:
            reasoning_prompt: The reasoning task for the LLM
            decision_context: Additional context for decision making
            
        Returns:
            LLM's reasoning result and decision
        """
        try:
            start_time = time.time()
            
            # Enhance prompt with agent context
            enhanced_prompt = f"""
You are {self.persona['name']}, {self.persona['title']}.

Background: {self.persona.get('background', 'Expert professional')}
Expertise: {', '.join(self.persona.get('expertise', []))}

REASONING TASK:
{reasoning_prompt}

CONTEXT:
{safe_json_dumps(decision_context or {}, indent=2)}

CRITICAL INSTRUCTIONS:
1. Use your professional expertise to reason through this systematically
2. Consider multiple perspectives and potential outcomes
3. Provide detailed reasoning for your decisions
4. Return your analysis in JSON format with clear reasoning steps
5. Be thorough but decisive

Reason through this step by step and provide your professional judgment.
"""
            
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0.7,
                max_tokens=12000
            )
            response_text = response.choices[0].message.content
            reasoning_result = self._extract_structured_response(response_text)
            
            # Track token usage and reasoning quality
            reasoning_time = time.time() - start_time
            token_usage = self._extract_token_usage(response)
            
            # Track token usage only (no confidence)
            self._track_reasoning_quality(reasoning_result, reasoning_time, token_usage)
            
            return reasoning_result
            
        except Exception as e:
            self.logger.error(f"LLM reasoning failed: {e}")
            return self._emergency_reasoning_fallback(model, reasoning_prompt, decision_context or {})
    
    def _extract_structured_response(self, response_text: str) -> Dict[str, Any]:
        """Extract structured response from LLM output"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # If no JSON, create structured response from text
            return {
                "reasoning_type": "text_analysis",
                "analysis": response_text,
                "structured_extraction": True,
                "confidence": 0.7,
                "reasoning_steps": response_text.split('\n') if response_text else []
            }
            
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.warning(f"Response extraction failed: {e}")
            return {
                "reasoning_type": "fallback",
                "analysis": response_text,
                "extraction_error": str(e),
                "confidence": 0.5
            }
    
    def _emergency_reasoning_fallback(self, model: str, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Emergency fallback that still uses LLM reasoning with simplified prompts
        Even in failure cases, we maintain LLM-based reasoning
        """
        try:
            simple_prompt = f"""
As {self.persona['name']}, provide a brief professional analysis:

Task: {prompt[:200]}...

Respond with your professional judgment in simple terms.
"""
            
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": simple_prompt}],
                temperature=0.7,
                max_tokens=12000
            )
            response_text = response.choices[0].message.content
            
            return {
                "reasoning_type": "emergency_fallback",
                "analysis": response_text,
                "confidence": 0.3,
                "fallback_used": True,
                "original_context": context
            }
            
        except Exception as e:
            # Absolute last resort - but still structured
            return {
                "reasoning_type": "critical_fallback",
                "analysis": f"Unable to complete reasoning due to {type(e).__name__}",
                "confidence": 0.1,
                "error": str(e),
                "requires_manual_intervention": True
            }
    
    def llm_validate_and_improve(self, current_result: Dict[str, Any], validation_criteria: List[str]) -> Dict[str, Any]:
        """
        LLM-powered validation and improvement of results
        """
        validation_prompt = f"""
As {self.persona['name']}, critically evaluate and improve this result:

CURRENT RESULT:
{safe_json_dumps(current_result, indent=2)}

VALIDATION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in validation_criteria)}

TASK:
1. Analyze the current result against each criterion
2. Identify any issues, gaps, or improvement opportunities
3. Provide specific recommendations for enhancement
4. Rate the overall quality (0.0-1.0)

Return your evaluation in JSON format with specific improvement suggestions.
"""
        
        return self.llm_reason_and_decide(self.model_name, validation_prompt, {"current_result": current_result})
    
    def _extract_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from litellm response."""
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                
                return {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "prompt_token_count": prompt_tokens,
                    "candidates_token_count": completion_tokens,
                    "thoughts_token_count": 0,
                    "total_token_count": total_tokens,
                }
            else:
                return {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "prompt_token_count": 0,
                    "candidates_token_count": 0,
                    "thoughts_token_count": 0,
                    "total_token_count": 0,
                }
        except Exception as e:
            self.logger.warning(f"Failed to extract token usage: {e}")
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "prompt_token_count": 0,
                "candidates_token_count": 0,
                "thoughts_token_count": 0,
                "total_token_count": 0,
            }
    
    def _track_reasoning_quality(self, reasoning_result: Dict[str, Any], reasoning_time: float, token_usage: Dict[str, int] = None):
        """Track token usage and timing only (no confidence tracking)."""
        token_usage = token_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Append to token usage log (cap to last 200 entries)
        self.performance_metrics["token_usage_log"].append({
            "timestamp": time.time(),
            "reasoning_time": reasoning_time,
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
            "reasoning_type": reasoning_result.get('reasoning_type', 'unknown')
        })
        if len(self.performance_metrics["token_usage_log"]) > 200:
            self.performance_metrics["token_usage_log"] = self.performance_metrics["token_usage_log"][-200:]

        # Track in reasoning history (without score)
        self.reasoning_history.append({
            "timestamp": time.time(),
            "reasoning_time": reasoning_time,
            "reasoning_type": reasoning_result.get('reasoning_type', 'unknown'),
            "token_usage": token_usage
        })
        
        # Log token usage for debugging
        self.logger.info(f"Agent {self.agent_id} - Tokens: input={token_usage.get('input_tokens', 0)}, output={token_usage.get('output_tokens', 0)}, total={token_usage.get('total_tokens', 0)}")
    
    def get_reasoning_insights(self) -> Dict[str, Any]:
        """Get insights into the agent's reasoning performance"""
        usage = self.performance_metrics.get("token_usage_log", [])
        if not usage:
            return {"status": "no_usage_data"}

        total_in = sum(u.get("input_tokens", 0) for u in usage)
        total_out = sum(u.get("output_tokens", 0) for u in usage)
        total = sum(u.get("total_tokens", 0) for u in usage)
        times = [u.get("reasoning_time", 0.0) for u in usage]

        return {
            "average_reasoning_time": (sum(times) / len(times)) if times else 0.0,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total,
            "decisions_tracked": len(usage)
        }

    def _generate_with_usage(self, model: str, content: str, generation_config: Optional[Dict[str, Any]] = None):
        """Wrapper for litellm.completion that records token usage uniformly."""
        try:
            kwargs = {}
            if generation_config:
                if isinstance(generation_config, dict):
                    kwargs = generation_config
                elif hasattr(generation_config, "temperature"):
                    kwargs["temperature"] = generation_config.temperature
                    if hasattr(generation_config, "max_output_tokens"):
                        kwargs["max_tokens"] = generation_config.max_output_tokens

            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": content}],
                **kwargs
            )
            
            token_usage = self._extract_token_usage(response)

            self._latest_token_usage = token_usage
            self.performance_metrics["token_usage_log"].append({
                "timestamp": time.time(),
                "input_tokens": token_usage.get("input_tokens", 0),
                "output_tokens": token_usage.get("output_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
                "prompt_token_count": token_usage.get("prompt_token_count", 0),
                "candidates_token_count": token_usage.get("candidates_token_count", 0),
                "thoughts_token_count": token_usage.get("thoughts_token_count", 0),
            })
            
            class MockResponse:
                def __init__(self, text):
                    self.text = text
            return MockResponse(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent's LLM-powered capabilities"""
        reasoning_insights = self.get_reasoning_insights()
        
        return {
            "agent_id": self.agent_id,
            "persona": self.persona,
            "llm_powered": True,
            "reasoning_capabilities": [
                "autonomous_decision_making",
                "context_aware_analysis", 
                "self_validation",
                "adaptive_reasoning"
            ],
            "performance_metrics": self.performance_metrics,
            "reasoning_insights": reasoning_insights,
            "is_initialized": self.is_initialized
        }
    
    def _track_request(self, start_time: float, success: bool, error_msg: Optional[str] = None):
        """Enhanced request tracking with reasoning quality"""
        end_time = time.time()
        response_time = end_time - start_time
        
        self.performance_metrics["total_requests"] += 1
        if success:
            self.performance_metrics["successful_requests"] += 1
        else:
            self.performance_metrics["error_count"] += 1
            if error_msg:
                self.logger.error(f"Request failed: {error_msg}")
        
        # Update average response time
        total_successful = self.performance_metrics["successful_requests"]
        if total_successful > 0:
            current_avg = self.performance_metrics["average_response_time"]
            self.performance_metrics["average_response_time"] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        
        self.performance_metrics["last_activity"] = datetime.now().isoformat()
        self.last_request_time = end_time
    
    def _validate_input(self, input_data: Dict[str, Any], required_fields: List[str]) -> bool:
        """LLM-assisted input validation"""
        validation_prompt = f"""
Validate this input data for completeness and quality:

REQUIRED FIELDS: {required_fields}
INPUT DATA: {safe_json_dumps(input_data, indent=2)}

Check:
1. Are all required fields present?
2. Are the field values appropriate and meaningful?
3. Are there any quality issues?

Return JSON: {{"valid": true/false, "issues": ["list", "of", "issues"], "recommendations": ["suggestions"]}}
"""
        
        validation_result = self.llm_reason_and_decide(self.model_name, validation_prompt)
        return validation_result.get('valid', True)  # Default to true if uncertain
    
    def _handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """LLM-powered error analysis and handling"""
        error_analysis_prompt = f"""
Analyze this error and provide recovery recommendations:

ERROR TYPE: {type(error).__name__}
ERROR MESSAGE: {str(error)}
CONTEXT: {context}

As {self.persona['name']}, provide:
1. Root cause analysis
2. Impact assessment
3. Recovery strategy
4. Prevention recommendations

Be practical and specific.
"""
        
        error_analysis = self.llm_reason_and_decide(self.model_name, error_analysis_prompt, {"error_details": str(error)})
        
        return {
            "status": "error",
            "agent_id": self.agent_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "llm_analysis": error_analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Enhanced status with LLM reasoning insights"""
        reasoning_insights = self.get_reasoning_insights()
        
        return {
            "agent_id": self.agent_id,
            "is_initialized": self.is_initialized,
            "last_activity": self.performance_metrics.get("last_activity"),
            "performance_summary": {
                "total_requests": self.performance_metrics["total_requests"],
                "success_rate": (
                    self.performance_metrics["successful_requests"] / 
                    max(1, self.performance_metrics["total_requests"])
                ),
                "average_response_time": self.performance_metrics["average_response_time"]
            },
            "reasoning_performance": reasoning_insights,
            "llm_powered": True,
            "model_name": self.model_name
        }
    
    def record_behavior(self, behavior_type: str, input_data: Any, output_data: Any, 
                       processing_time: float, additional_metadata: Dict[str, Any] = None):
        """Record agent behavior for analysis and debugging"""
        behavior_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "behavior_type": behavior_type,
            "session_id": self.current_session_id,
            "input_data": self._serialize_for_behavior(input_data),
            "output_data": self._serialize_for_behavior(output_data),
            "processing_time": processing_time,
            "additional_metadata": additional_metadata or {}
        }
        
        self.behavior_history.append(behavior_record)
        
        # Keep only last 100 behaviors to avoid memory issues
        if len(self.behavior_history) > 100:
            self.behavior_history = self.behavior_history[-100:]
    
    def _serialize_for_behavior(self, data: Any) -> Dict[str, Any]:
        """Serialize data for behavior tracking"""
        try:
            if isinstance(data, str):
                return {"type": "string", "length": len(data), "preview": data[:200]}
            elif isinstance(data, dict):
                return {
                    "type": "dict",
                    "keys": list(data.keys()),
                    "preview": {k: str(v)[:100] for k, v in list(data.items())[:5]}
                }
            elif hasattr(data, '__dict__'):
                return {
                    "type": type(data).__name__,
                    "attributes": {k: str(v)[:100] for k, v in data.__dict__.items()}
                }
            else:
                return {"type": type(data).__name__, "value": str(data)[:200]}
        except Exception as e:
            return {"type": "serialization_error", "error": str(e)}
    
    def save_behaviors_to_file(self, output_dir: str, session_id: str = None):
        """Save behavior history to file"""
        import os
        from pathlib import Path
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            session_id = session_id or self.current_session_id or "default"
            behavior_file = output_path / f"{self.agent_id}_behaviors_{session_id}.json"
            
            behavior_data = {
                "agent_id": self.agent_id,
                "model_name": self.model_name,
                "session_id": session_id,
                "total_behaviors": len(self.behavior_history),
                "behaviors": self.behavior_history
            }
            
            import json
            import numpy as np
            
            # Custom JSON encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.bool_, bool)):
                        return bool(obj)
                    return super().default(obj)
            
            with open(behavior_file, 'w', encoding='utf-8') as f:
                json.dump(behavior_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            self.logger.info(f"Saved {len(self.behavior_history)} behaviors to {behavior_file}")
            return str(behavior_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save behaviors: {str(e)}")
            return None
    
    def set_session_id(self, session_id: str):
        """Set the current session ID for behavior tracking"""
        self.current_session_id = session_id 
