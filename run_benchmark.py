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
Benchmark runner for CoDA

Runs benchmark queries using the agentic visualization system.
Data loading is handled automatically by the orchestrator.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coda.workflow import Workflow
from coda.workflow.orchestrator import WorkflowConfiguration


def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )


def load_benchmark_instructions(data_path: str) -> list:
    """Load benchmark instructions from JSON file"""
    instructions_file = Path(data_path) / "benchmark_instructions.json"

    if not instructions_file.exists():
        raise FileNotFoundError(f"Benchmark instructions not found: {instructions_file}")

    with open(instructions_file, 'r') as f:
        return json.load(f)


def run_single_query(query_id: int, query_info: Dict[str, Any],
                     config: WorkflowConfiguration,
                     model_name: str,
                     search_model_name: str,
                     data_path: str) -> Dict[str, Any]:
    """
    Run a single benchmark query

    Args:
        query_id: Query ID (1-based)
        query_info: Query information from benchmark_instructions.json
        config: Workflow configuration
        data_path: Base path to benchmark data

    Returns:
        Result dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info(f"{'='*50}")
    logger.info(f"Processing Query {query_id}")
    logger.info(f"{'='*50}")

    try:
        # Create orchestrator for this query
        orchestrator = Workflow(
            model_name=model_name,
            search_model_name=search_model_name,
            config=config,
            query_id=query_id
        )

        # Construct data file path (only for queries 76-100 with CSV data)
        data_dir = Path(data_path) / "data" / str(query_id)

        # Prepare workflow context
        workflow_ctx = {
            'test_id': query_id,
            'expert_instruction': query_info['expert_instruction'],
            'simple_instruction': query_info['simple_instruction']
        }

        # Only add data_file_path if data directory exists
        if data_dir.exists():
            workflow_ctx['data_file_path'] = str(data_dir)

        # Execute workflow - orchestrator handles all data loading
        result = orchestrator.execute_workflow(
            query=query_info['expert_instruction'],
            data_input=None,  # Let orchestrator auto-detect and load
            workflow_context=workflow_ctx
        )

        # Cleanup
        orchestrator.cleanup()

        logger.info(f"Query {query_id}: {'✓ Success' if result.final_success else '✗ Failed'}")
        logger.info(f"Quality score: {result.final_quality_score:.3f}")
        logger.info(f"Iterations: {result.total_iterations}")
        logger.info(f"Time: {result.total_execution_time:.1f}s")

        return {
            'query_id': query_id,
            'success': result.final_success,
            'quality_score': result.final_quality_score,
            'iterations': result.total_iterations,
            'execution_time': result.total_execution_time,
            'output_path': result.final_output_path
        }

    except Exception as e:
        logger.error(f"Query {query_id} failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        return {
            'query_id': query_id,
            'success': False,
            'error': str(e)
        }


def run_benchmark(
    model_name: str,
    search_model_name: str,
    start_id: int = 1,
    end_id: int = 100,
    data_path: str = "matplotbench_data",
    output_dir: str = "benchmark_outputs",
    quality_threshold: float = 0.9,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Run benchmark evaluation

    Args:
        model_name: LLM model name
        search_model_name: Model for search tasks
        start_id: Start query ID (1-based)
        end_id: End query ID (inclusive)
        data_path: Path to benchmark data directory
        output_dir: Output directory for results
        quality_threshold: Quality threshold for workflow
        max_iterations: Maximum refinement iterations

    Returns:
        Summary results dictionary
    """
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("Agentic Visualization Benchmark")
    logger.info("="*60)
    logger.info(f"Query range: {start_id} - {end_id}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quality threshold: {quality_threshold}")
    logger.info(f"Max iterations: {max_iterations}")
    logger.info("="*60)

    # Load benchmark instructions
    instructions = load_benchmark_instructions(data_path)
    logger.info(f"Loaded {len(instructions)} benchmark instructions")

    # Validate range
    if end_id > len(instructions):
        logger.warning(f"End ID {end_id} > available queries {len(instructions)}, adjusting to {len(instructions)}")
        end_id = len(instructions)

    # Create workflow configuration
    config = WorkflowConfiguration(
        quality_threshold=quality_threshold,
        max_iterations=max_iterations,
        save_intermediate_results=True,
        output_directory=output_dir,
        data_path=data_path,
        enable_search_agent=True,
        enable_global_todo=True
    )

    # Run queries
    results = []
    successful = 0
    failed = 0

    for query_id in range(start_id, end_id + 1):
        query_info = instructions[query_id - 1]

        result = run_single_query(query_id, query_info, config, model_name, search_model_name, data_path)
        results.append(result)

        if result['success']:
            successful += 1
        else:
            failed += 1

    # Summary
    logger.info("="*60)
    logger.info("Benchmark Complete!")
    logger.info("="*60)
    logger.info(f"Total queries: {end_id - start_id + 1}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful / (end_id - start_id + 1):.1%}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)

    # Save summary
    summary = {
        'benchmark_info': {
            'start_id': start_id,
            'end_id': end_id,
            'total_queries': end_id - start_id + 1,
            'data_path': data_path,
            'output_dir': output_dir,
            'quality_threshold': quality_threshold,
            'max_iterations': max_iterations,
            'timestamp': datetime.now().isoformat()
        },
        'results': results,
        'summary': {
            'successful': successful,
            'failed': failed,
            'success_rate': successful / (end_id - start_id + 1)
        }
    }

    summary_file = Path(output_dir) / f'benchmark_summary_{start_id}_{end_id}.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run agentic visualization benchmark')
    parser.add_argument('--start', type=int, default=1, help='Start query ID (default: 1)')
    parser.add_argument('--end', type=int, default=100, help='End query ID (default: 100)')
    parser.add_argument('--data-path', default='matplotbench_data', help='Path to benchmark data')
    parser.add_argument('--output-dir', default='benchmark_outputs', help='Output directory')
    parser.add_argument('--quality-threshold', type=float, default=0.9, help='Quality threshold (default: 0.9)')
    parser.add_argument('--max-iterations', type=int, default=3, help='Max refinement iterations (default: 3)')
    parser.add_argument('--model-name', required=True, help='LLM model name for general tasks')
    parser.add_argument('--search-model-name', required=True, help='LLM model name for search tasks')
    parser.add_argument('--log-file', help='Log file path (optional)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    try:
        run_benchmark(
            start_id=args.start,
            end_id=args.end,
            data_path=args.data_path,
            output_dir=args.output_dir,
            quality_threshold=args.quality_threshold,
            max_iterations=args.max_iterations,
            model_name=args.model_name,
            search_model_name=args.search_model_name
        )
        return 0
    except Exception as e:
        logging.error(f"Benchmark failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
