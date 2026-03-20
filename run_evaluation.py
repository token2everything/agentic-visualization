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
Evaluation runner for CoDA benchmark results

Evaluates generated visualizations using LLM-based assessment.
"""

import argparse
import os
import sys
import json
import logging
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coda.evaluation import AgenticBenchmarkEvaluator


def evaluate_query_worker(args):
    """Worker function for parallel evaluation"""
    query_id, results_dir, config = args

    # Setup logging for this worker
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Create evaluator for this worker
        evaluator_config = config.copy()
        evaluator_config['results_dir'] = results_dir

        evaluator = AgenticBenchmarkEvaluator(evaluator_config)
        result = evaluator.evaluate_single_query(query_id)

        logger.info(f"Query {query_id} evaluated: success={result.get('success', False)}")
        return query_id, result

    except Exception as e:
        import traceback
        error_details = f"Worker failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Query {query_id} failed: {error_details}")
        return query_id, {
            'query_id': query_id,
            'error': error_details,
            'success': False
        }


def run_parallel_evaluation(
    model_name: str,
    vision_model_name: str,
    results_dir: str,
    start_id: int = 1,
    end_id: int = 100,
    data_path: str = 'matplotbench_data',
    num_processes: int = 10,
) -> Dict[str, Any]:
    """
    Run parallel evaluation on benchmark results

    Args:
        results_dir: Directory containing benchmark results
        start_id: Start query ID
        end_id: End query ID
        data_path: Path to benchmark data (for instructions and ground truth)
        num_processes: Number of parallel processes

    Returns:
        Evaluation results dictionary
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_dir, f"evaluation_{timestamp}.log")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting parallel evaluation of queries {start_id}-{end_id} with {num_processes} processes")

    # Configuration for workers
    config = {
        'benchmark_instructions_path': os.path.join(data_path, 'benchmark_instructions.json'),
        'ground_truth_dir': os.path.join(data_path, 'ground_truth'),
        'results_dir': results_dir,
        'model_name': model_name,
        'vision_model_name': vision_model_name
    }

    # Prepare worker arguments
    worker_args = []
    for query_id in range(start_id, end_id + 1):
        worker_args.append((query_id, results_dir, config))

    # Results structure
    results = {
        'evaluation_info': {
            'start_id': start_id,
            'end_id': end_id,
            'total_queries': end_id - start_id + 1,
            'num_processes': num_processes,
            'evaluation_timestamp': datetime.now().isoformat(),
            'results_dir': results_dir,
            'data_path': data_path
        },
        'query_results': {},
        'summary_stats': {}
    }

    successful_evaluations = 0
    total_code_score = 0
    total_vision_score = 0

    # Run parallel evaluation
    with mp.Pool(processes=num_processes) as pool:
        logger.info(f"Starting evaluation with {num_processes} processes...")

        for query_id, result in pool.imap(evaluate_query_worker, worker_args):
            results['query_results'][str(query_id)] = result

            if result.get('success', False):
                successful_evaluations += 1
                scores = result.get('overall_scores', {})
                total_code_score += scores.get('code_score', 0)
                total_vision_score += scores.get('vision_score', 0)
                logger.info(f"✓ Query {query_id}: Code={scores.get('code_score', 'N/A')}, Vision={scores.get('vision_score', 'N/A')}")
            else:
                logger.error(f"✗ Query {query_id}: {result.get('error', 'Unknown error')}")

    # Calculate summary statistics
    total_queries = end_id - start_id + 1
    if successful_evaluations > 0:
        results['summary_stats'] = {
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': total_queries - successful_evaluations,
            'success_rate': successful_evaluations / total_queries,
            'average_code_score': total_code_score / successful_evaluations,
            'average_vision_score': total_vision_score / successful_evaluations,
            'average_overall_score': (total_code_score + total_vision_score) / (2 * successful_evaluations)
        }

    # Save results
    output_path = os.path.join(results_dir, f'evaluation_results_{start_id}_{end_id}_{timestamp}.json')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved to: {output_path}")
        results['output_path'] = output_path
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")

    # Print summary
    logger.info("="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total queries: {total_queries}")
    logger.info(f"Successful evaluations: {successful_evaluations}")
    logger.info(f"Failed evaluations: {total_queries - successful_evaluations}")
    logger.info(f"Success rate: {successful_evaluations/total_queries:.1%}")

    if successful_evaluations > 0:
        stats = results['summary_stats']
        logger.info(f"Average code score: {stats['average_code_score']:.1f}/100")
        logger.info(f"Average vision score: {stats['average_vision_score']:.1f}/100")
        logger.info(f"Average overall score: {stats['average_overall_score']:.1f}/100")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate benchmark results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to benchmark results directory')
    parser.add_argument('--start', type=int, default=1, help='Start query ID')
    parser.add_argument('--end', type=int, default=100, help='End query ID')
    parser.add_argument('--data-path', type=str, default='matplotbench_data',
                       help='Path to benchmark data (instructions and ground truth)')
    parser.add_argument('--processes', type=int, default=10,
                       help='Number of parallel processes')
    parser.add_argument('--model-name', type=str, required=True,
                       help='LLM model name for evaluation')
    parser.add_argument('--vision-model-name', type=str, required=True,
                       help='Vision model name for evaluation')

    args = parser.parse_args()

    try:
        results = run_parallel_evaluation(
            results_dir=args.results_dir,
            start_id=args.start,
            end_id=args.end,
            data_path=args.data_path,
            num_processes=args.processes,
            model_name=args.model_name,
            vision_model_name=args.vision_model_name
        )

        if results.get('summary_stats'):
            print("\nEvaluation completed successfully!")
            return 0
        else:
            print("\nEvaluation failed")
            return 1

    except Exception as e:
        print(f"Evaluation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
