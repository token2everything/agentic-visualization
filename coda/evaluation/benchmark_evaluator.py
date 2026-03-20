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
Benchmark Evaluator for agentic_vis_core - Only AgenticBenchmarkEvaluator
"""

import os
import json
import logging
import glob
from typing import Dict, List, Any, Optional
from datetime import datetime

from .llm_evaluator import LLMEvaluator


class AgenticBenchmarkEvaluator:
    """
    Standard evaluator for agentic method data structure
    - Uses matplotbench_data/benchmark_instructions.json
    - Works with ground truth images in matplotbench_data/ground_truth/
    - Standard 100-query benchmark dataset
    - Code is stored in generated_code.py
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize LLM evaluator
        self.llm_evaluator = LLMEvaluator(config)

        # Default paths - relative to agentic_vis_core
        self.benchmark_instructions_path = self.config.get(
            'benchmark_instructions_path',
            'matplotbench_data/benchmark_instructions.json'
        )
        self.ground_truth_dir = self.config.get(
            'ground_truth_dir',
            'matplotbench_data/ground_truth'
        )
        self.results_dir = self.config.get(
            'results_dir',
            'benchmark_outputs'
        )

        self.logger.info("AgenticBenchmarkEvaluator initialized")

    def load_benchmark_instructions(self) -> List[Dict[str, Any]]:
        """Load benchmark instructions from JSON file"""
        try:
            with open(self.benchmark_instructions_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load benchmark instructions: {str(e)}")
            return []

    def get_query_paths(self, query_id: int) -> Dict[str, str]:
        """
        Get all relevant paths for a query

        Args:
            query_id: Query ID (1-based)

        Returns:
            Dictionary with paths for code, folder, ground_truth, etc.
        """
        query_dir = self._find_query_directory(query_id)

        if not query_dir:
            # Fallback to simple structure
            query_dir = os.path.join(self.results_dir, f"query_{query_id}")

        paths = {
            'query_dir': query_dir,
            'code_path': os.path.join(query_dir, 'generated_code.py'),
            'folder_path': query_dir,
            'ground_truth_path': os.path.join(self.ground_truth_dir, f"example_{query_id}.png"),
            'rollback_path': os.path.join(query_dir, 'rollback.png'),
            'config_path': os.path.join(query_dir, 'config.json'),
            'log_path': os.path.join(query_dir, 'execution.log')
        }

        return paths

    def _find_query_directory(self, query_id: int) -> str:
        """
        Find query directory handling different naming patterns

        Expected structures:
        1. results_dir/query_X/query_X/ (standard)
        2. results_dir/query_X/ (simple)

        Args:
            query_id: Query ID (1-based)

        Returns:
            Path to query directory if found, None otherwise
        """
        if not os.path.exists(self.results_dir):
            return None

        # Strategy 1: Look for query_X/query_X structure
        query_dir = os.path.join(self.results_dir, f"query_{query_id}")
        if os.path.exists(query_dir):
            query_subdir = os.path.join(query_dir, f"query_{query_id}")
            if os.path.exists(query_subdir):
                return query_subdir

        # Strategy 2: Look for model_timestamp/query_X/
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isdir(item_path):
                query_subdir = os.path.join(item_path, f"query_{query_id}")
                if os.path.exists(query_subdir):
                    return query_subdir

        # Strategy 3: Direct query_X in results_dir
        direct_query_dir = os.path.join(self.results_dir, f"query_{query_id}")
        if os.path.exists(direct_query_dir):
            return direct_query_dir

        return None

    def has_csv_data(self, query_id: int) -> bool:
        """
        Check if query has CSV data files available

        Args:
            query_id: Query ID

        Returns:
            True if query has CSV data files, False otherwise
        """
        if query_id in range(76, 101):
            # Extract base path from benchmark_instructions_path
            base_path = os.path.dirname(self.benchmark_instructions_path)
            source_dir = os.path.join(base_path, 'data', str(query_id))

            if os.path.exists(source_dir):
                csv_files = glob.glob(os.path.join(source_dir, '*.csv'))
                json_files = glob.glob(os.path.join(source_dir, '*.json'))
                return len(csv_files) > 0 or len(json_files) > 0
        return False

    def evaluate_single_query(self, query_id: int,
                            benchmark_instructions: List[Dict[str, Any]] = None,
                            workspace_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate single agentic query

        Args:
            query_id: Query ID (1-based)
            benchmark_instructions: Pre-loaded benchmark instructions
            workspace_dir: Optional workspace directory for evaluation

        Returns:
            Evaluation results dictionary
        """
        self.logger.info(f"Evaluating agentic query {query_id}")

        # Load standard benchmark instructions
        if benchmark_instructions is None:
            benchmark_instructions = self.load_benchmark_instructions()

        if not benchmark_instructions or query_id > len(benchmark_instructions):
            return {
                'query_id': query_id,
                'error': f'Query ID {query_id} out of range or no benchmark instructions',
                'success': False
            }

        # Get query instruction (1-based indexing)
        query_instruction = benchmark_instructions[query_id - 1]
        simple_instruction = query_instruction.get("simple_instruction", "")
        expert_instruction = query_instruction.get("expert_instruction", "")

        # Get paths
        paths = self.get_query_paths(query_id)

        # Check if required files exist
        if not os.path.exists(paths['code_path']):
            return {
                'query_id': query_id,
                'error': f"Generated code not found: {paths['code_path']}",
                'success': False
            }

        # Load generated code
        try:
            with open(paths['code_path'], 'r') as f:
                generated_code = f.read()
        except Exception as e:
            return {
                'query_id': query_id,
                'error': f"Failed to read generated code: {str(e)}",
                'success': False
            }

        # Check if query has CSV data
        has_csv_data = self.has_csv_data(query_id)

        # Run LLM evaluation with ground truth
        try:
            evaluation_results = self.llm_evaluator.evaluate_query_result(
                query_id=str(query_id),
                query_text=simple_instruction,
                code=generated_code,
                folder_path=paths['folder_path'],
                ground_truth_path=paths['ground_truth_path'],
                rollback_path=paths.get('rollback_path'),
                has_csv_data=has_csv_data
            )

            # Add agentic-specific metadata
            evaluation_results.update({
                'benchmark_info': {
                    'simple_instruction': simple_instruction,
                    'expert_instruction': expert_instruction,
                    'query_id': query_id,
                    'method_type': 'agentic'
                },
                'file_paths': paths,
                'success': True,
                'evaluation_timestamp': datetime.now().isoformat()
            })

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Evaluation failed for agentic query {query_id}: {str(e)}")
            return {
                'query_id': query_id,
                'error': f"Evaluation failed: {str(e)}",
                'success': False
            }

    def evaluate_benchmark(self, query_range: Optional[tuple] = None,
                         workspace_dir: str = None,
                         save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate a range of benchmark queries

        Args:
            query_range: Tuple of (start, end) query IDs, None for all
            workspace_dir: Optional workspace directory for evaluation
            save_results: Whether to save results to file

        Returns:
            Complete benchmark evaluation results
        """
        self.logger.info("Starting benchmark evaluation")

        # Load benchmark instructions
        benchmark_instructions = self.load_benchmark_instructions()
        if not benchmark_instructions:
            return {'error': 'Failed to load benchmark instructions', 'success': False}

        # Determine query range
        if query_range is None:
            start_id, end_id = 1, len(benchmark_instructions)
        else:
            start_id, end_id = query_range

        self.logger.info(f"Evaluating queries {start_id} to {end_id}")

        # Evaluate each query
        results = {
            'evaluation_info': {
                'start_id': start_id,
                'end_id': end_id,
                'total_queries': end_id - start_id + 1,
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluator_config': self.config
            },
            'query_results': {},
            'summary_stats': {}
        }

        successful_evaluations = 0
        total_code_score = 0
        total_vision_score = 0

        for query_id in range(start_id, end_id + 1):
            try:
                query_result = self.evaluate_single_query(
                    query_id, benchmark_instructions, workspace_dir
                )
                results['query_results'][str(query_id)] = query_result

                if query_result.get('success', False):
                    successful_evaluations += 1
                    scores = query_result.get('overall_scores', {})
                    total_code_score += scores.get('code_score', 0)
                    total_vision_score += scores.get('vision_score', 0)

            except Exception as e:
                self.logger.error(f"Failed to evaluate query {query_id}: {str(e)}")
                results['query_results'][str(query_id)] = {
                    'query_id': query_id,
                    'error': str(e),
                    'success': False
                }

        # Calculate summary statistics
        if successful_evaluations > 0:
            results['summary_stats'] = {
                'successful_evaluations': successful_evaluations,
                'failed_evaluations': (end_id - start_id + 1) - successful_evaluations,
                'success_rate': successful_evaluations / (end_id - start_id + 1),
                'average_code_score': total_code_score / successful_evaluations,
                'average_vision_score': total_vision_score / successful_evaluations,
                'average_overall_score': (total_code_score + total_vision_score) / (2 * successful_evaluations)
            }

        # Save results if requested
        if save_results:
            output_dir = workspace_dir or self.results_dir
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(
                output_dir,
                f'benchmark_evaluation_{start_id}_{end_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            try:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                self.logger.info(f"Evaluation results saved to: {output_path}")
                results['output_path'] = output_path
            except Exception as e:
                self.logger.error(f"Failed to save results: {str(e)}")

        self.logger.info(f"Benchmark evaluation completed: {successful_evaluations}/{end_id - start_id + 1} successful")
        return results
