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
Debug Agent

Executes generated matplotlib code, diagnoses runtime errors, and applies
targeted fixes. Includes visual overlap detection and correction.
"""

import os
import time
import subprocess
import logging
import re
import shutil
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from .base import BaseAgent
from .code_generator import CodeGenerationResult
from .llm_overlap_fixer import get_overlap_fix_prompt, get_smart_overlap_analysis_prompt

@dataclass
class ExecutionResult:
    """Result of code execution"""
    returncode: int
    stdout: str
    stderr: str
    execution_time: float
    success: bool
    output_file: Optional[str] = None
    fixed_code: Optional[str] = None

class DebugAgent(BaseAgent):
    """
    Debug Agent that executes generated matplotlib code and fixes errors.

    Provides iterative debugging with visual overlap detection and correction.
    """

    def __init__(self, model_name: str, search_model_name: str, agent_id: str = "debug_agent", config=None):
        super().__init__(agent_id, config)
        self.model_name = model_name
        self.search_model_name = search_model_name
        self.logger = logging.getLogger(__name__)
        
        try:
            import vertexai
            from vertexai.generative_models import Tool, grounding

            self.grounding_tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
            self.search_enabled = True
            self.logger.info("Google Search initialized for debugging")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Google Search: {str(e)}")
            self.search_enabled = False

        # Track overlap issues for proactive fixing
        self.overlap_detection_enabled = True
    
    def execute_code(self, code_result, query_dir: str, data_file_path: Optional[str] = None, 
                    query_id: Optional[str] = None, data_path: Optional[str] = None) -> ExecutionResult:
        """
        Execute code from code generation result.
        This is the interface expected by workflow orchestrator.
        """
        if hasattr(code_result, 'generated_code'):
            code_text = code_result.generated_code
        else:
            # Fallback - try to get code from result object
            code_text = getattr(code_result, 'code', str(code_result))
        
        # Always use generated_code.py to match code_generator output
        py_filepath = "generated_code.py"
            
        return self.run_python_code(
            code_text=code_text,
            run_cwd=query_dir,
            py_filepath=py_filepath,
            timeout_seconds=30,
            copy_to_session_dir=True,
            data_file_path=data_file_path
        )
    
    def debug_code_if_needed(self, code_result, execution_result: ExecutionResult, 
                           query_dir: str, data_file_path: Optional[str] = None,
                           query_id: Optional[str] = None, data_path: Optional[str] = None) -> ExecutionResult:
        """
        Debug code if execution failed.
        This is the interface expected by workflow orchestrator.
        """
        if execution_result.success:
            return execution_result
            
        # Extract code and error message
        if hasattr(code_result, 'generated_code'):
            code_text = code_result.generated_code
        else:
            code_text = getattr(code_result, 'code', str(code_result))
            
        error_msg = execution_result.stderr or "Unknown error"
        
        # Use debug_and_fix method
        try:
            fixed_code = self.debug_and_fix(code_text, error_msg, max_attempts=3)
            if fixed_code != code_text:
                # Re-execute the fixed code
                py_filepath = "generated_code.py"  # Always use generated_code.py to match code_generator
                return self.run_python_code(
                    code_text=fixed_code,
                    run_cwd=query_dir,
                    py_filepath=py_filepath,
                    timeout_seconds=30,
                    copy_to_session_dir=True,
                    data_file_path=data_file_path
                )
        except Exception as e:
            self.logger.error(f"Debug failed: {str(e)}")
        
        # Return original failed result if debug couldn't fix
        return execution_result
    
    def _intelligent_file_path_replacement(self, code_text: str, data_file_path: Optional[str]) -> str:
        """
        Intelligently replace incorrect file paths in generated code
        by analyzing the actual data directory structure.
        """
        if not data_file_path:
            return code_text
            
        # Extract directory path from data_file_path
        if os.path.isfile(data_file_path):
            data_dir = os.path.dirname(data_file_path)
            primary_file = os.path.basename(data_file_path)
        else:
            # If data_file_path is a directory
            data_dir = data_file_path
            primary_file = None
        
        # Find all data files in the directory
        available_files = []
        if os.path.isdir(data_dir):
            for file in os.listdir(data_dir):
                if file.lower().endswith(('.csv', '.json', '.xlsx', '.parquet')):
                    available_files.append(file)
        
        if not available_files:
            self.logger.warning(f"No data files found in directory: {data_dir}")
            return code_text
        
        # Sort files by preference: data.csv first, then others
        available_files.sort(key=lambda x: (
            0 if x.lower() == 'data.csv' else
            1 if x.lower().endswith('.csv') else
            2 if x.lower().endswith('.json') else 3
        ))
        
        self.logger.info(f"Available data files: {available_files}")
        
        # Define patterns to match incorrect file paths
        problematic_patterns = [
            # Common incorrect patterns
            r"pd\.read_csv\(['\"](\d+)['\"]",           # '81', '87', etc.
            r"pd\.read_csv\(['\"](\d+\.csv)['\"]",      # '81.csv', '87.csv', etc.
            r"pd\.read_csv\(['\"](\d+\.json)['\"]",     # '81.json', etc.
            r"pd\.read_csv\(['\"]DataFrame['\"]",       # 'DataFrame' placeholder
            r"pd\.read_json\(['\"](\d+)['\"]",          # For JSON files
            r"pd\.read_json\(['\"](\d+\.json)['\"]",    # '81.json' patterns for read_json
        ]
        
        replaced = False
        original_code = code_text
        
        for pattern in problematic_patterns:
            matches = re.findall(pattern, code_text)
            if matches:
                # Choose appropriate file based on the pattern
                if 'read_json' in pattern:
                    # Prefer JSON files for read_json calls
                    json_files = [f for f in available_files if f.lower().endswith('.json')]
                    replacement_file = json_files[0] if json_files else available_files[0]
                else:
                    # For CSV patterns, prefer CSV files
                    csv_files = [f for f in available_files if f.lower().endswith('.csv')]
                    replacement_file = csv_files[0] if csv_files else available_files[0]
                
                # Determine the function call based on file extension
                if replacement_file.lower().endswith('.json'):
                    func_call = 'pd.read_json'
                else:
                    func_call = 'pd.read_csv'
                
                # Replace the pattern
                if 'read_json' in pattern:
                    code_text = re.sub(pattern, f"{func_call}('{replacement_file}'", code_text)
                else:
                    code_text = re.sub(pattern, f"{func_call}('{replacement_file}'", code_text)
                replaced = True
                self.logger.info(f"Replaced pattern {pattern} with file: {replacement_file}")
                break
        
        if not replaced:
            # Check for any other suspicious patterns
            suspicious_patterns = [
                r"pd\.read_csv\(['\"]([^'\"]*[^/\\]*)['\"]",  # Any suspicious filename
            ]
            
            for pattern in suspicious_patterns:
                matches = re.findall(pattern, code_text)
                if matches:
                    for match in matches:
                        # Check if the matched filename exists in available files
                        if match not in available_files and not os.path.exists(match):
                            # This looks like an incorrect filename
                            replacement_file = available_files[0]  # Use the first available file
                            code_text = code_text.replace(f"'{match}'", f"'{replacement_file}'")
                            replaced = True
                            self.logger.info(f"Replaced suspicious filename '{match}' with '{replacement_file}'")
                            break
                if replaced:
                    break
        
        if replaced:
            self.logger.info(f"File path replacement completed. Using files: {available_files}")
        else:
            self.logger.info("No file path replacements needed")
            
        return code_text
        
    def run_python_code(
        self,
        code_text: str,
        run_cwd: str,
        py_filepath: str,
        timeout_seconds: int = 30,
        copy_to_session_dir: bool = True,
        data_file_path: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute Python code and return the result.
        This is the main entry point for code execution.
        """
        start_time = time.time()
        
        # Ensure the working directory exists
        os.makedirs(run_cwd, exist_ok=True)
        
        # Replace data file placeholder if needed
        if data_file_path:
            # First try standard DATA_FILE_PATH replacement
            if 'DATA_FILE_PATH' in code_text:
                code_text = code_text.replace('DATA_FILE_PATH', f"'{data_file_path}'")
                self.logger.info(f"Replaced DATA_FILE_PATH placeholder with: {data_file_path}")
            else:
                # For benchmark data, intelligently detect and replace incorrect file paths
                code_text = self._intelligent_file_path_replacement(code_text, data_file_path)
        
        # Write the Python code to the file
        try:
            with open(py_filepath, 'w', encoding='utf-8') as f:
                f.write(code_text)
            self.logger.info(f"Code written to: {py_filepath}")
        except Exception as e:
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Failed to write code to file: {str(e)}",
                execution_time=0,
                success=False
            )
        
        # Execute the code
        result = self._execute_code(py_filepath, run_cwd, timeout_seconds)
        
        # Look for generated PNG files
        if result.success and copy_to_session_dir:
            png_files = []
            for file in os.listdir(run_cwd):
                if file.endswith('.png'):
                    png_files.append(os.path.join(run_cwd, file))
            
            if png_files:
                # Use the first PNG file found
                result.output_file = png_files[0]
                self.logger.info(f"Found output file: {result.output_file}")
        
        return result
    
    def _execute_code(self, py_filepath: str, run_cwd: str, timeout_seconds: int) -> ExecutionResult:
        """Execute Python code file and return results"""
        start_time = time.time()
        
        try:
            # Use only the filename if the py_filepath is within run_cwd
            if os.path.isabs(py_filepath):
                # If absolute path, use as is
                exec_path = py_filepath
            elif py_filepath.startswith(run_cwd):
                # If path starts with run_cwd, extract just the filename
                exec_path = os.path.basename(py_filepath)
            else:
                # Otherwise use just the basename
                exec_path = os.path.basename(py_filepath)
            
            # Run the Python code
            result = subprocess.run(
                ["python", exec_path],
                cwd=run_cwd,
                timeout=timeout_seconds,
                capture_output=True,
                text=True
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                self.logger.info(f"Code executed successfully in {execution_time:.2f}s")
            else:
                self.logger.warning(f"Code execution failed with return code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"Stderr: {result.stderr[:500]}")  # Log first 500 chars of stderr
                if result.stdout:
                    self.logger.info(f"Stdout: {result.stdout[:500]}")  # Log first 500 chars of stdout
            
            return ExecutionResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                success=success
            )
            
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Code execution timed out after {timeout_seconds}s")
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Execution timed out after {timeout_seconds} seconds",
                execution_time=execution_time,
                success=False
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Code execution error: {str(e)}")
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time=execution_time,
                success=False
            )
    
    def debug_and_fix(self, code: str, error_msg: str, max_attempts: int = 3, 
                     run_cwd: str = None, py_filepath: str = None,
                     data_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Debug and fix the code using intelligent analysis.
        
        Args:
            code: The code that failed
            error_msg: The error message
            max_attempts: Maximum number of fix attempts
            run_cwd: Working directory for execution
            py_filepath: Path to python file
            data_file_path: Path to data file if needed
            
        Returns:
            Dictionary with debug results
        """
        self.logger.info(f"Starting intelligent debug process with {max_attempts} max attempts")

        # First, try proactive overlap prevention if enabled
        if self.overlap_detection_enabled:
            if self._llm_detect_overlap(code, error_msg):
                self.logger.info("Detected potential overlap issues, applying LLM-based fixes")
                code = self._llm_fix_overlaps(code)
                # If this was just an overlap issue without real error, return fixed code
                if not error_msg or 'success' in error_msg.lower():
                    return {'fixed_code': code, 'fix_applied': 'proactive_overlap_prevention', 'success': True}

        fixed_code = code
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            self.logger.info(f"Debug attempt {attempts}/{max_attempts}")
            
            # Try intelligent fix
            if attempts == 1:
                fixed_code = self._intelligent_fix(fixed_code, error_msg)
            else:
                # For subsequent attempts, include previous failure context
                fixed_code = self._intelligent_fix(fixed_code, error_msg, 
                                                             previous_attempts=attempts-1)
            
            if not fixed_code:
                self.logger.warning(f"No intelligent fix found in attempt {attempts}")
                break
            
            # Test the fixed code
            if run_cwd and py_filepath:
                test_result = self.run_python_code(
                    fixed_code, run_cwd, py_filepath, 
                    timeout_seconds=30, data_file_path=data_file_path
                )
                
                if test_result.success:
                    self.logger.info(f"Code fixed successfully in {attempts} attempts")
                    return {
                        "success": True,
                        "fixed_code": fixed_code,
                        "attempts": attempts,
                        "execution_result": test_result,
                        "original_error": error_msg
                    }
                else:
                    error_msg = test_result.stderr
                    self.logger.warning(f"Attempt {attempts} failed: {error_msg}")
            else:
                # If we can't test, return the fixed code
                return {
                    "success": True,
                    "fixed_code": fixed_code,
                    "attempts": attempts,
                    "execution_result": None,
                    "original_error": error_msg
                }
        
        # All attempts failed
        return {
            "success": False,
            "fixed_code": fixed_code,
            "attempts": attempts,
            "execution_result": None,
            "original_error": error_msg
        }
    
    def _intelligent_fix(self, code: str, error_msg: str, 
                                    previous_attempts: int = 0) -> Optional[str]:
        """
        Fix the code based on error analysis.
        
        Args:
            code: The code to fix
            error_msg: The error message
            previous_attempts: Number of previous attempts (for context)
            
        Returns:
            Fixed code or None if no fix found
        """
        try:
            # Step 1: Analyze the error comprehensively 
            error_analysis = self._analyze_error(code, error_msg)
            self.logger.info(f"Error analysis: {error_analysis.get('error_type', 'Unknown')}")
            
            # Step 2: Search for solutions
            solution = self._search_solution(error_analysis, code, error_msg)
            
            # Step 4: Apply the intelligent fix
            if solution and solution.get('fixed_code'):
                fixed_code = solution['fixed_code']
                
                # Validate that fixed_code is actually Python code, not explanatory text
                if self._is_valid_python_code(fixed_code):
                    self.logger.info(f"Applied intelligent fix: {solution.get('solution_explanation', 'Unknown fix')}")
                    return fixed_code
                else:
                    self.logger.warning("Fixed code contains non-Python content, trying to extract code")
                    # Try to extract Python code from the text
                    extracted_code = self._extract_python_code_from_text(fixed_code)
                    if extracted_code and self._is_valid_python_code(extracted_code):
                        self.logger.info("Successfully extracted Python code from response")
                        return extracted_code
                    else:
                        self.logger.warning("Could not extract valid Python code, using fallback fixes")
            else:
                self.logger.warning("No intelligent solution found, trying fallback fixes")
                return self._apply_fallback_fixes(code, error_msg)
                
        except Exception as e:
            self.logger.error(f"Intelligent fix failed: {str(e)}")
            return self._apply_fallback_fixes(code, error_msg)
    
    def _apply_fallback_fixes(self, code: str, error_msg: str) -> Optional[str]:
        """
        Apply common fallback fixes for matplotlib code.
        """
        fixed_code = code
        
        # Fix common import issues
        if "No module named" in error_msg or "import" in error_msg.lower():
            fixed_code = self._fix_import_issues(fixed_code, error_msg)
        
        # Fix common matplotlib backend issues
        if "backend" in error_msg.lower() or "display" in error_msg.lower():
            fixed_code = self._fix_backend_issues(fixed_code)
        
        # Fix common data loading issues
        if "file" in error_msg.lower() or "path" in error_msg.lower():
            fixed_code = self._fix_file_path_issues(fixed_code, error_msg)
        
        # Fix common variable name issues
        if "not defined" in error_msg or "NameError" in error_msg:
            fixed_code = self._fix_variable_issues(fixed_code, error_msg)
        
        return fixed_code if fixed_code != code else None
    
    def _fix_import_issues(self, code: str, error_msg: str) -> str:
        """Fix common import-related issues"""
        lines = code.split('\n')
        fixed_lines = []
        imports_added = False
        
        # Add missing standard imports at the beginning
        if not any('import matplotlib.pyplot as plt' in line for line in lines):
            if not imports_added:
                fixed_lines.extend([
                    'import matplotlib.pyplot as plt',
                    'import numpy as np',
                    'import pandas as pd',
                    ''
                ])
                imports_added = True
        
        # Add specific imports based on error message
        missing_modules = re.findall(r"No module named '([^']+)'", error_msg)
        for module in missing_modules:
            if module not in code:
                fixed_lines.append(f'import {module}')
        
        if imports_added or missing_modules:
            fixed_lines.append('')
        
        fixed_lines.extend(lines)
        return '\n'.join(fixed_lines)
    
    def _fix_backend_issues(self, code: str) -> str:
        """Fix matplotlib backend issues"""
        lines = code.split('\n')
        fixed_lines = []
        
        # Add Agg backend at the beginning
        backend_set = False
        for line in lines:
            if 'matplotlib' in line and 'import' in line and not backend_set:
                fixed_lines.append(line)
                fixed_lines.append('import matplotlib')
                fixed_lines.append("matplotlib.use('Agg')")
                backend_set = True
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_file_path_issues(self, code: str, error_msg: str) -> str:
        """Fix common file path issues"""
        # This is a simple implementation - could be enhanced
        return code.replace('DATA_FILE_PATH', "'data.csv'")
    
    def _fix_variable_issues(self, code: str, error_msg: str) -> str:
        """Fix common variable naming issues"""
        # Extract undefined variable names from error message
        undefined_vars = re.findall(r"name '([^']+)' is not defined", error_msg)
        
        fixed_lines = []
        for line in code.split('\n'):
            fixed_line = line
            for var in undefined_vars:
                # Try common substitutions
                if var == 'data':
                    fixed_line = fixed_line.replace(var, 'df')
                elif var == 'df':
                    fixed_line = fixed_line.replace(var, 'data')
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _analyze_error(self, code: str, error_msg: str) -> Dict[str, Any]:
        """
        Analyze error and extract key information.

        Args:
            code: The code that caused the error
            error_msg: The error message

        Returns:
            Dictionary with error analysis
        """
        try:
            # Use LLM to detect if this is an overlap issue
            is_overlap_issue = self._llm_detect_overlap(code, error_msg) if self.overlap_detection_enabled else False

            # Build analysis prompt - let LLM reason about the solution
            if is_overlap_issue:
                issue_description = "VISUAL ISSUE: Elements are overlapping in the generated plot"
                analysis_focus = """Analyze the visual overlap problem:
1. Identify which elements are likely overlapping
2. What spacing/positioning issues exist?
3. Use your matplotlib expertise to determine the best fix"""
            else:
                issue_description = f"ERROR MESSAGE:\n{error_msg}"
                analysis_focus = """Analyze the error:
1. What is causing the error?
2. What is missing or wrong in the code?
3. Determine the simplest fix"""

            analysis_prompt = f"""
You are a matplotlib expert debugging visualization code.

{issue_description}

CODE:
```python
{code}
```

{analysis_focus}

Provide your analysis in this JSON format:
{{
    "error_type": "visual_overlap|syntax|runtime|import|logic",
    "root_cause": "detailed explanation of the issue",
    "overlapping_elements": ["if overlap, list affected elements"],
    "missing_requirements": "what needs to be added or changed",
    "error_location": "where the issue occurs in the code",
    "fix_strategy": "your recommended matplotlib solution",
    "confidence": 0.0-1.0
}}
"""
            
            response = self._generate_with_usage(
                model=self.model_name,
                content=analysis_prompt,
                generation_config={
                    "max_output_tokens": 2000,
                    "temperature": 0.1
                }
            )
            
            result_text = response.text
            parsed_result = self._safe_json_parse(result_text, {})
            
            if not parsed_result:
                self.logger.warning("Failed to parse error analysis, using fallback")
                return {
                    "error_type": "unknown",
                    "root_cause": "Analysis failed",
                    "error_location": "unknown",
                    "severity": "major",
                    "fix_strategy": "Manual review needed",
                    "confidence": 0.3
                }
                
            self.logger.info(f"Error analysis completed with {parsed_result.get('confidence', 0.5)} confidence")
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Error analysis failed: {str(e)}")
            return {
                "error_type": "analysis_failed",
                "root_cause": f"Analysis error: {str(e)}",
                "error_location": "unknown",
                "severity": "major",
                "fix_strategy": "Fallback to simple fixes",
                "confidence": 0.2
            }
    
    def _search_solution(self, error_analysis: Dict[str, Any], code: str, error_msg: str) -> Dict[str, Any]:
        """
        Search for solutions using Google Search and LLM based on error analysis.
        
        Args:
            error_analysis: The error analysis from LLM
            code: The original code
            error_msg: The error message
            
        Returns:
            Dictionary with solution including fixed code
        """
        try:
            # Step 1: Use Google Search to find solutions for the specific error
            search_results = self._search_bug_solutions(error_analysis, error_msg)
            
            # Step 2: Use search results to generate a fixed solution
            if search_results:
                # Simplify the solution prompt - no hardcoded strategies
                solution_prompt = f"""
You are a matplotlib expert fixing visualization issues.

ISSUE ANALYSIS:
{json.dumps(error_analysis, indent=2)}

SEARCH RESULTS:
{search_results}

CURRENT CODE:
```python
{code}
```

ERROR MESSAGE:
{error_msg}

TASK: Use your matplotlib expertise to fix this issue completely.

Provide solution in JSON format:
{{
    "objective": "What the code is trying to achieve",
    "issue_identified": "Root cause of the problem",
    "solution_explanation": "How your fix solves the issue",
    "search_insights": "Relevant insights from search",
    "fixed_code": "Complete corrected Python code",
    "changes_made": ["List of specific changes made"],
    "confidence": 0.0-1.0
}}

Return ONLY valid JSON.
"""
            else:
                # Fallback to reasoning if search fails
                solution_prompt = f"""
You are a matplotlib debugging expert. Based on this error analysis, provide a complete fixed code:

ERROR ANALYSIS:  
{json.dumps(error_analysis, indent=2)}

ORIGINAL CODE:
```python
{code}
```

FULL ERROR:
{error_msg}

Provide a complete working solution in JSON format:
{{
    "solution_explanation": "Brief explanation of the fix applied",
    "search_insights": "Reasoning used to solve this error",
    "fixed_code": "Complete corrected Python code", 
    "changes_made": ["List of specific changes made to fix the error"],
    "confidence": 0.0-1.0
}}

IMPORTANT: The fixed_code must be complete, executable Python code that resolves all issues.
"""
            
            response = self._generate_with_usage(
                model=self.model_name,
                content=solution_prompt,
                generation_config={
                    "max_output_tokens": 8000,
                    "temperature": 0.1
                }
            )
            
            result_text = response.text
            parsed_result = self._safe_json_parse(result_text, {})
            
            if parsed_result and parsed_result.get('fixed_code'):
                search_info = "with Google search" if search_results else "with LLM reasoning"
                self.logger.info(f"Solution found {search_info} with {parsed_result.get('confidence', 0.5)} confidence")
                return parsed_result
            
            # Fallback: try to extract code from response
            code_blocks = re.findall(r'```python\n(.*?)\n```', result_text, re.DOTALL)
            if code_blocks:
                return {
                    "solution_explanation": "Code extracted from response",
                    "search_insights": "Code block extraction",
                    "fixed_code": code_blocks[-1],
                    "changes_made": ["Extracted from response"],
                    "confidence": 0.6
                }
                
        except Exception as e:
            self.logger.error(f"Failed to search solution: {str(e)}")
        
        return None
    
    def _is_valid_python_code(self, code: str) -> bool:
        """Check if the given text is valid Python code"""
        try:
            # Check if text starts with obvious non-code patterns
            lines = code.strip().split('\n')
            if not lines:
                return False
                
            first_line = lines[0].strip()
            # Check for obvious non-code patterns
            non_code_patterns = [
                '* ', '1.', '2.', '3.', '4.', '5.',  # Numbered lists
                'The code', 'This is', 'Here is', 'Based on',  # Explanatory text
                'Final Plan:', 'Solution:', 'Error:', 'Note:'  # Common prefixes
            ]
            
            for pattern in non_code_patterns:
                if first_line.startswith(pattern):
                    return False
            
            # Try to compile the code (basic syntax check)
            compile(code, '<string>', 'exec')
            return True
        except:
            return False
    
    def _extract_python_code_from_text(self, text: str) -> Optional[str]:
        """Extract Python code from text that may contain explanations"""
        import re
        
        # First try to find code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        
        # Try to find code after common patterns
        patterns = [
            r'corrected_code["\']:\s*["\']([^"\']*)["\']',
            r'corrected_code["\']:\s*"""([^"]*)"""',
            r'```python\n(.*?)```',
            r'^(import\s+.*?)(?=\n\n|\Z)',  # Start from import statements
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                candidate = match.group(1).strip()
                if self._is_valid_python_code(candidate):
                    return candidate
        
        return None
    
    def _search_bug_solutions(self, error_analysis: Dict[str, Any], error_msg: str) -> str:
        """
        Use Google Search to find solutions for specific bug/error.
        
        Args:
            error_analysis: Structured error analysis
            error_msg: Original error message
            
        Returns:
            Search results as text, or empty string if search fails
        """
        if not self.search_enabled:
            self.logger.warning("Google Search not available")
            return ""
        
        try:
            # Create search query based on error analysis
            error_type = error_analysis.get('error_type', 'unknown')
            root_cause = error_analysis.get('root_cause', '')
            
            # Extract key error message parts for search
            key_error_parts = []
            if 'Error:' in error_msg:
                error_line = error_msg.split('Error:')[-1].split('\n')[0].strip()
                key_error_parts.append(error_line)
            
            # Build search query - optimize for overlap issues
            if error_type == 'visual_overlap' or 'overlap' in root_cause.lower():
                # Specific search for overlap issues
                overlapping_elements = error_analysis.get('overlapping_elements', [])
                if overlapping_elements:
                    element_str = ' '.join(overlapping_elements[:2])  # First 2 elements
                    search_query = f"matplotlib fix overlapping {element_str} tight_layout subplots_adjust"
                else:
                    search_query = "matplotlib prevent overlapping labels legend tight_layout constrained_layout"
            else:
                # Standard error search
                search_query = f"python matplotlib {error_type} {root_cause}"
                if key_error_parts:
                    search_query += f" {key_error_parts[0][:50]}"  # Limit length
                search_query += " solution fix"
            
            self.logger.info(f"Searching for bug solutions with query: {search_query}")
            
            search_model = generative_models.GenerativeModel(self.search_model_name)
            response = search_model.generate_content(
                f"Find solutions for this Python matplotlib error: {search_query}",
                tools=[self.grounding_tool],
                generation_config=generative_models.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.1
                )
            )
            
            search_results = response.text
            self.logger.info(f"Google Search returned {len(search_results)} characters of results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Google Search failed: {str(e)}")
            return ""
    
    def _safe_json_parse(self, response_text: str, fallback_value=None):
        """Safely parse JSON from LLM response"""
        try:
            # Try to find JSON in code blocks first
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                if json_end != -1:
                    json_text = response_text[json_start:json_end].strip()
                    return json.loads(json_text)
            
            # Try to find JSON object
            if '{' in response_text:
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
                    
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in JSON parsing: {str(e)}")
            
        return fallback_value
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Simple debug agent cleanup completed")

    def _llm_detect_overlap(self, code: str, error_msg: str) -> bool:
        """
        Use LLM to detect if the issue is about visual overlaps.

        Args:
            code: The matplotlib code
            error_msg: Error message (might be empty for overlap issues)

        Returns:
            True if overlap issue detected, False otherwise
        """
        try:
            # Quick heuristic check first
            if error_msg and 'error' in error_msg.lower() and 'overlap' not in error_msg.lower():
                return False  # Likely a real error, not overlap

            # Use LLM for intelligent analysis
            analysis_prompt = get_smart_overlap_analysis_prompt(code, error_msg)
            response = self._generate_with_usage(
                model=self.model_name,
                content=analysis_prompt,
                generation_config=generative_models.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.2
                )
            )

            result = self._safe_json_parse(response.text, {})
            has_overlap = result.get('has_overlap_risk', False)
            severity = result.get('severity', 'low')

            # Log the analysis
            if has_overlap:
                self.logger.info(f"LLM detected {severity} overlap risk: {result.get('overlap_type', 'unknown')}")

            return has_overlap and severity in ['high', 'medium']

        except Exception as e:
            self.logger.warning(f"LLM overlap detection failed: {e}, using fallback")
            # Simple fallback
            return 'tight_layout' not in code and ('pie(' in code or 'bar(' in code)

    def _llm_fix_overlaps(self, code: str) -> str:
        """
        Use LLM to intelligently fix overlap issues in matplotlib code.

        Args:
            code: Original matplotlib code

        Returns:
            Code with overlap prevention measures
        """
        try:
            # Get LLM to fix the overlaps
            fix_prompt = get_overlap_fix_prompt(code)
            response = self._generate_with_usage(
                model=self.model_name,
                content=fix_prompt,
                generation_config=generative_models.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.3
                )
            )

            # Extract fixed code
            fixed_code = self._extract_python_code_from_text(response.text)
            if fixed_code and self._is_valid_python_code(fixed_code):
                self.logger.info("Successfully applied LLM-based overlap fixes")
                return fixed_code
            else:
                self.logger.warning("Failed to extract valid code from LLM response")
                # Fallback: simple tight_layout addition
                return self._simple_overlap_fallback(code)

        except Exception as e:
            self.logger.error(f"LLM overlap fix failed: {e}")
            return self._simple_overlap_fallback(code)

    def _simple_overlap_fallback(self, code: str) -> str:
        """
        Simple fallback for overlap fixing - just add tight_layout.
        """
        if 'tight_layout' not in code and 'plt.savefig' in code:
            return code.replace('plt.savefig', 'plt.tight_layout()\nplt.savefig')
        return code
