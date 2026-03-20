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
Workflow Orchestrator

Coordinates the complete CoDA agent pipeline from query analysis through
visual refinement, implementing quality-gated feedback loops.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import os
from pathlib import Path

from ..agents.base import BaseAgent, AgentMessage
from ..agents.query_analyzer import QueryAnalyzer, QueryAnalysisResult
from ..agents.data_processor import DataProcessor, DataProcessingResult
from ..agents.viz_mapper import VisualizationMappingAgent, VisualizationMapping
from ..agents.design_explorer import DesignExplorer, DesignExplorationResult
from ..agents.code_generator import CodeGenerator, CodeGenerationResult
from ..agents.debug_agent import DebugAgent, ExecutionResult
from ..agents.visual_evaluator import VisualEvaluator, VisualAssessment
from ..agents.search_agent import SearchAgent

@dataclass
class WorkflowConfiguration:
    """Configuration for workflow execution."""
    quality_threshold: float = 0.9
    max_iterations: int = 3
    max_debug_iterations: int = 3
    save_intermediate_results: bool = True
    output_directory: str = "workflow_outputs"
    timeout_per_agent: int = 1800  # 30 minutes per agent
    enable_search_agent: bool = True
    enable_global_todo: bool = True
    todo_completion_threshold: float = 0.8  # Threshold for TODO completion rate
    data_path: str = "data"  # Base path to dataset directory
    
@dataclass
class WorkflowIteration:
    """Results from a single workflow iteration"""
    iteration_number: int
    code_generation_result: CodeGenerationResult
    debug_result: ExecutionResult  # Execution result object
    visual_assessment: Dict[str, Any]  # Simplified assessment
    refinement_result: Optional[Dict[str, Any]]  # Simplified refinement
    todo_evaluation: Optional[Dict[str, Any]]  # TODO completion evaluation
    achieved_quality: float
    iteration_success: bool
    iteration_time: float
    
@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    workflow_id: str
    query_analysis: QueryAnalysisResult
    data_processing: DataProcessingResult
    design_exploration: DesignExplorationResult
    iterations: List[WorkflowIteration]
    final_success: bool
    final_quality_score: float
    total_iterations: int
    total_execution_time: float
    final_output_path: Optional[str]
    workflow_summary: Dict[str, Any]

class WorkflowOrchestrator:
    """
    Workflow Orchestrator for the CoDA pipeline.

    Manages all eight agents across four phases with quality-gated feedback
    loops and iterative refinement until thresholds are met.
    """
    
    def __init__(self,
        model_name: str,
        search_model_name: str,
        config: WorkflowConfiguration = None,
        query_id: str = None
    ):
        self.config = config or WorkflowConfiguration()
        self.model_name = model_name
        self.search_model_name = search_model_name
        self.logger = logging.getLogger(__name__)
        self.workflow_id = f"workflow_query_{query_id}" if query_id else "workflow_default"
        self.query_id = query_id

        self.total_llm_calls = 0

        self.query_analyzer = QueryAnalyzer(model_name=model_name)
        self.data_processor = DataProcessor(model_name=model_name)
        self.viz_mapping_agent = VisualizationMappingAgent(model_name=model_name)

        if self.config.enable_search_agent:
            self.search_agent = SearchAgent(model_name=model_name, search_model_name=search_model_name)
        else:
            self.search_agent = None

        self.design_explorer = DesignExplorer(model_name=model_name)
        self.code_generator = CodeGenerator(model_name=model_name, search_model_name=search_model_name)
        self.debug_agent = DebugAgent(model_name=model_name, search_model_name=search_model_name)
        self.visual_evaluator = VisualEvaluator(model_name=model_name, search_model_name=search_model_name)
        
        session_id = f"session_{self.workflow_id}"
        self.query_analyzer.set_session_id(session_id)
        self.data_processor.set_session_id(session_id)
        self.viz_mapping_agent.set_session_id(session_id)
        
        if self.search_agent:
            self.search_agent.set_session_id(session_id)
            
        self.design_explorer.set_session_id(session_id)
        self.code_generator.set_session_id(session_id)
        self.debug_agent.set_session_id(session_id)
        self.visual_evaluator.set_session_id(session_id)
        
        self.agent_metadata = {
            "query_analyzer": [],
            "data_processor": [],
            "viz_mapping_agent": [],
            "search_agent": [],
            "design_explorer": [],
            "code_generator": [],
            "debug_agent": [],
            "visual_evaluator": []
        }
        
        if "/" in model_name:
            model_short = model_name.split("/")[-1]
        else:
            model_short = model_name
        model_short = model_short.replace("-", "_")
        if query_id:
            self.session_dir = Path(self.config.output_directory) / f"{model_short}_query_{query_id}"
        else:
            self.session_dir = Path(self.config.output_directory) / f"{model_short}_session"
        
        if not hasattr(self, 'session_dir') or self.session_dir is None:
            self.session_dir = Path(self.config.output_directory) / f"{model_short}_default"
            
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_da_benchmark_folder(self, folder_path: Path) -> bool:
        """Detect if a folder is a multi-file dataset task folder."""
        if not folder_path.is_dir():
            return False
        
        has_yaml_config = any(folder_path.glob("*.yaml")) or any(folder_path.glob("*.yml"))
        has_readme = any(folder_path.glob("README.md")) or any(folder_path.glob("readme.md"))
        has_data_files = (any(folder_path.glob("*.csv")) or
                         any(folder_path.glob("*.sqlite")) or
                         any(folder_path.glob("*.db")) or
                         any(folder_path.glob("*.xlsx")) or
                         any(folder_path.glob("*.json")))

        da_indicators = (has_data_files and (has_yaml_config or has_readme))

        folder_name = folder_path.name.lower()
        has_da_naming = folder_name.startswith('plot-') and any(chart_type in folder_name for chart_type in ['bar', 'line', 'pie', 'scatter'])
        
        return da_indicators or has_da_naming

    def _check_for_data_file(self, data_input: Any, workflow_context: Optional[Dict[str, Any]]) -> bool:
        """Return True if a data file or DataFrame is available for processing."""
        if data_input is not None:
            if isinstance(data_input, pd.DataFrame) and not data_input.empty:
                return True
            if isinstance(data_input, (str, Path)):
                data_path = Path(data_input)
                if data_path.exists() and data_path.is_file():
                    # Check for common data file extensions
                    if data_path.suffix.lower() in ['.csv', '.json', '.xlsx', '.parquet', '.tsv', '.txt']:
                        return True

        if workflow_context:
            data_file_path = workflow_context.get('data_file_path')
            if data_file_path:
                data_path = Path(data_file_path)
                if data_path.exists() and (data_path.is_file() or data_path.is_dir()):
                    return True

        self.logger.info("No external data file found - using generated data")
        return False

    def _record_agent_metadata(self, agent_name: str, input_data: Any, output_data: Any,
                              execution_time: float, iteration: int = 0) -> None:
        """Record complete input/output metadata for each agent"""
        metadata_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "execution_time_seconds": execution_time,
            "input_data": self._serialize_data_for_metadata(input_data),
            "output_data": self._serialize_data_for_metadata(output_data),
            "input_size": self._calculate_data_size(input_data),
            "output_size": self._calculate_data_size(output_data)
        }
        
        self.agent_metadata[agent_name].append(metadata_entry)
    
    def _serialize_data_for_metadata(self, data: Any) -> Dict[str, Any]:
        """Serialize data for metadata storage"""
        try:
            if isinstance(data, pd.DataFrame):
                return {
                    "type": "DataFrame",
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": data.dtypes.to_dict(),
                    "sample_data": data.head(3).to_dict()
                }
            elif hasattr(data, '__dict__'):  # For dataclass objects
                return {
                    "type": type(data).__name__,
                    "attributes": {k: str(v)[:500] for k, v in asdict(data).items()}  # Limit size
                }
            elif isinstance(data, dict):
                return {
                    "type": "dict",
                    "keys": list(data.keys()),
                    "sample_content": {k: str(v)[:200] for k, v in list(data.items())[:5]}
                }
            elif isinstance(data, str):
                return {
                    "type": "string",
                    "length": len(data),
                    "content_preview": data[:300]
                }
            else:
                return {
                    "type": type(data).__name__,
                    "content": str(data)[:300]
                }
        except Exception as e:
            return {
                "type": "serialization_error",
                "error": str(e),
                "raw_type": type(data).__name__
            }
    
    def _calculate_data_size(self, data: Any) -> int:
        """Calculate approximate data size in bytes"""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            elif hasattr(data, '__dict__'):
                return len(str(data).encode('utf-8'))
            else:
                return len(str(data).encode('utf-8'))
        except:
            return 0
    
    def _save_complete_metadata(self, workflow_id: str) -> None:
        """Save complete metadata to results directory"""
        metadata_file = self.session_dir / f"metadata_{workflow_id}.json"
        
        complete_metadata = {
            "workflow_id": workflow_id,
            "timestamp": datetime.now().isoformat(),
            "configuration": asdict(self.config),
            "agent_metadata": self.agent_metadata,
            "summary": {
                "total_query_analyzer_calls": len(self.agent_metadata["query_analyzer"]),
                "total_data_processor_calls": len(self.agent_metadata["data_processor"]),
                "total_viz_mapping_calls": len(self.agent_metadata["viz_mapping_agent"]),
                "total_search_agent_calls": len(self.agent_metadata["search_agent"]),
                "total_design_explorer_calls": len(self.agent_metadata["design_explorer"]),
                "total_code_generator_calls": len(self.agent_metadata["code_generator"]),
                "total_debug_agent_calls": len(self.agent_metadata["debug_agent"]),
                "total_visual_evaluator_calls": len(self.agent_metadata["visual_evaluator"])
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(complete_metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Complete metadata saved to: {metadata_file}")
        
    def execute_workflow(self, query: str, data_input: Any,
                        workflow_context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute the complete workflow with feedback loop.
        
        Args:
            query: User query string
            data_input: Data input (file path or DataFrame)
            workflow_context: Optional context information
            
        Returns:
            WorkflowResult with complete execution results
        """
        start_time = datetime.now()
        # Use consistent workflow_id without timestamp
        workflow_id = self.workflow_id
        
        # Create query-specific directory
        query_id = workflow_context.get('test_id', 1) if workflow_context else 1
        self.query_dir = self.session_dir / f"query_{query_id}"
        self.query_dir.mkdir(parents=True, exist_ok=True)
        
        da_task_context = None
        
        if workflow_context and ('test_id' in workflow_context or 'data_file_path' in workflow_context) and not da_task_context:
            possible_paths = []

            if 'data_file_path' in workflow_context:
                explicit_path = Path(workflow_context['data_file_path'])
                if explicit_path.exists():
                    if explicit_path.is_file():
                        # Single file
                        possible_paths.append(explicit_path)
                    elif explicit_path.is_dir():
                        csv_files = list(explicit_path.glob("*.csv"))
                        json_files = list(explicit_path.glob("*.json"))

                        for csv_file in csv_files:
                            possible_paths.append(csv_file)
                        for json_file in json_files:
                            possible_paths.append(json_file)
                        
                        self.logger.info(f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files in directory: {explicit_path}")
            
            if 'test_id' in workflow_context:
                test_id = workflow_context['test_id']
                base_path = Path(self.config.data_path)
                
                # Check multiple possible data file locations
                possible_paths.extend([
                    base_path / f"data/{test_id}/data.csv",
                    base_path / f"data/{test_id}/data.json",
                ])
            
            if 'test_id' in workflow_context:
                test_id = workflow_context['test_id']
                base_path = Path(self.config.data_path)
                data_dir = base_path / f"data/{test_id}"
                if data_dir.exists():
                    csv_files = list(data_dir.glob("*.csv"))
                    json_files = list(data_dir.glob("*.json"))
                    excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
                    sqlite_files = list(data_dir.glob("*.sqlite")) + list(data_dir.glob("*.db"))
                    yaml_files = list(data_dir.glob("*.yaml")) + list(data_dir.glob("*.yml"))

                    for csv_file in csv_files:
                        if csv_file not in possible_paths:
                            possible_paths.append(csv_file)
                    for json_file in json_files:
                        if json_file not in possible_paths:
                            possible_paths.append(json_file)
                    for excel_file in excel_files:
                        if excel_file not in possible_paths:
                            possible_paths.append(excel_file)
                    for sqlite_file in sqlite_files:
                        if sqlite_file not in possible_paths:
                            possible_paths.append(sqlite_file)
                    for yaml_file in yaml_files:
                        if yaml_file not in possible_paths:
                            possible_paths.append(yaml_file)
            
            copied_files = []
            
            for data_file_path in possible_paths:
                if data_file_path.exists():
                    source_file = Path(data_file_path)
                    # Create unique target filename for multiple files
                    if len(possible_paths) > 1:
                        target_file = self.query_dir / source_file.name
                    else:
                        target_file = self.query_dir / f"data{source_file.suffix}"
                    
                    import shutil
                    shutil.copy2(data_file_path, target_file)
                    copied_files.append((source_file, target_file))
                    self.logger.info(f"Data file copied: {data_file_path} -> {target_file}")
            
            if len(copied_files) > 1:
                if workflow_context is not None:
                    workflow_context['multi_files'] = True
                    workflow_context['multi_files_files'] = [tgt.name for _, tgt in copied_files]

            if copied_files:
                if len(copied_files) == 1:
                    source_file, target_file = copied_files[0]
                    try:
                        if source_file.suffix.lower() == '.csv':
                            if isinstance(data_input, dict) and data_input.get("use_local_file"):
                                # Keep the metadata dictionary as is
                                self.logger.info(f"Using CSV file metadata: {data_input.get('metadata', {}).get('filename', 'unknown')}")
                                self.logger.info(f"CSV file shape from metadata: {data_input.get('metadata', {}).get('shape', 'unknown')}")
                            else:
                                # Load actual data only if not using metadata approach
                                actual_data = pd.read_csv(target_file)
                                data_input = actual_data
                                self.logger.info(f"Loaded CSV data with shape {actual_data.shape} and columns {list(actual_data.columns)}")
                            
                        elif source_file.suffix.lower() == '.json':
                            # Check if we already have metadata-based input
                            if isinstance(data_input, dict) and data_input.get("use_local_file"):
                                # Keep the metadata dictionary as is
                                self.logger.info(f"Using JSON file metadata: {data_input.get('metadata', {}).get('filename', 'unknown')}")
                                self.logger.info(f"JSON file shape from metadata: {data_input.get('metadata', {}).get('shape', 'unknown')}")
                            else:
                                # Use safe JSON loading to handle Sankey JSON structures
                                actual_data = self._safe_load_json_data(target_file)
                                data_input = actual_data
                                if hasattr(actual_data, 'shape'):
                                    self.logger.info(f"Loaded JSON data with shape {actual_data.shape} and columns {list(actual_data.columns)}")
                                else:
                                    self.logger.info(f"Loaded JSON data (non-DataFrame format)")
                                
                        elif source_file.suffix.lower() in ['.xlsx', '.xls']:
                            if isinstance(data_input, dict) and data_input.get("use_local_file"):
                                self.logger.info(f"Using Excel file metadata: {data_input.get('metadata', {}).get('filename', 'unknown')}")
                            else:
                                actual_data = pd.read_excel(target_file)
                                data_input = actual_data
                                self.logger.info(f"Loaded Excel data with shape {actual_data.shape}")

                        elif source_file.suffix.lower() in ['.sqlite', '.db']:
                            if isinstance(data_input, dict) and data_input.get("use_local_file"):
                                self.logger.info(f"Using SQLite file metadata: {data_input.get('metadata', {}).get('filename', 'unknown')}")
                            else:
                                self.logger.info(f"SQLite file will use metadata approach: {source_file.name}")

                        elif source_file.suffix.lower() in ['.yaml', '.yml']:
                            import yaml
                            with open(target_file, 'r') as f:
                                yaml_data = yaml.safe_load(f)
                            if isinstance(yaml_data, list):
                                actual_data = pd.DataFrame(yaml_data)
                                data_input = actual_data
                                self.logger.info(f"Loaded YAML data with shape {actual_data.shape}")
                            else:
                                self.logger.info(f"YAML file contains non-list data: {type(yaml_data)}")

                        else:
                            self.logger.info(f"Unknown file type {source_file.suffix}, keeping original data_input")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load data file: {str(e)}")

                workflow_context['copied_data_files'] = [str(tf) for sf, tf in copied_files]
                workflow_context['data_file_count'] = len(copied_files)
                workflow_context['original_data_path'] = str(copied_files[0][0].parent) if len(copied_files) > 1 else str(copied_files[0][0])
                
                if len(copied_files) == 1:
                    workflow_context['copied_data_file'] = str(copied_files[0][1])
                    workflow_context['data_file_type'] = copied_files[0][0].suffix
                else:
                    workflow_context['multiple_data_files'] = True
                    
                self.logger.info(f"Copied {len(copied_files)} data files to query directory")
            else:
                self.logger.warning(f"Data file not found for query {test_id}")
        
        self.logger.info(f"Starting workflow execution: {workflow_id}")
        
        try:
            # Phase 1: Query Analysis
            self.logger.info("Phase 1: Query Analysis")
            phase1_start = datetime.now()
            data_file_path = workflow_context.get('original_data_path') if workflow_context else None
            context_for_analyzer = workflow_context.copy() if workflow_context else {}
            if data_file_path:
                context_for_analyzer['data_file_path'] = data_file_path

            query_result = self.query_analyzer.analyze_query(query, context_for_analyzer)

            if query_result.translated_query and query_result.translated_query != query:
                self.logger.info(f"Query was translated: {query_result.translated_query[:100]}...")
            phase1_time = (datetime.now() - phase1_start).total_seconds()
            
            # Record metadata for query analyzer
            self._record_agent_metadata("query_analyzer", query, query_result, phase1_time, 0)
            self.logger.info(f"Query analysis completed in {phase1_time:.2f}s")
            
            if self.config.save_intermediate_results:
                self._save_intermediate_result(workflow_id, "query_analysis", query_result)
            
            # Phase 2: Data Processing
            has_data_file = self._check_for_data_file(data_input, workflow_context)

            if has_data_file:
                self.logger.info("Phase 2: Data Processing")
                phase2_start = datetime.now()
                data_todos = [todo for todo in query_result.global_todo_list if todo.get('agent') == 'data_processor']
                data_result = self.data_processor.process_data(data_input, data_todos, workflow_context)
                phase2_time = (datetime.now() - phase2_start).total_seconds()

                # Record metadata for data processor
                data_processor_input = {"data_input": data_input, "data_todos": data_todos, "workflow_context": workflow_context}
                self._record_agent_metadata("data_processor", data_processor_input, data_result, phase2_time, 0)
                self.logger.info(f"Data processing completed in {phase2_time:.2f}s")

                if self.config.save_intermediate_results:
                    self._save_intermediate_result(workflow_id, "data_processing", data_result)
            else:
                self.logger.info("Phase 2: Skipping Data Processing (no input data file)")
                import pandas as pd
                data_result = DataProcessingResult(
                    processing_id="skipped_data_processing",
                    processed_data=pd.DataFrame(),
                    data_insights={"status": "no_external_data", "message": "No external data file provided - using generated data"},
                    processing_steps=["Skipped data processing - no input file"],
                    processing_time=0.0,
                    data_quality_score=1.0,
                    recommendations=[{"type": "synthetic_data", "message": "Generate synthetic data as needed"}],
                    data_summary={"status": "no_external_data", "type": "generated"}
                )
                phase2_time = 0.0
            
            # Phase 2b: Visualization Mapping
            self.logger.info("Phase 2b: Query-to-Data Visualization Mapping")
            phase2b_start = datetime.now()
            viz_mapping_result = self.viz_mapping_agent.map_query_to_visualization(
                query, data_result.processed_data, {"query_result": query_result}
            )
            phase2b_time = (datetime.now() - phase2b_start).total_seconds()

            viz_mapping_input = {"query": query, "data": data_result.processed_data, "query_context": query_result}
            self._record_agent_metadata("viz_mapping_agent", viz_mapping_input, viz_mapping_result, phase2b_time, 0)
            self.logger.info(f"Visualization mapping completed in {phase2b_time:.2f}s")
            self.logger.info(f"Mapped to {viz_mapping_result.chart_type} chart with {len(viz_mapping_result.data_mappings)} column mappings")
            
            if self.config.save_intermediate_results:
                self._save_intermediate_result(workflow_id, "visualization_mapping", viz_mapping_result.to_dict())
            
            # Phase 2c: Search for matplotlib examples
            search_result = None
            if self.config.enable_search_agent and self.search_agent:
                self.logger.info("Phase 2c: Searching for matplotlib examples")
                phase2c_start = datetime.now()
                query_analysis_dict = {
                    "recommended_plot_types": getattr(query_result, 'recommended_plot_types', []),
                    "visualization_type": getattr(query_result, 'visualization_type', ''),
                    "chart_type": getattr(query_result, 'chart_type', '')
                }
                
                search_result = self.search_agent.process_query_output(query_analysis_dict)
                phase2c_time = (datetime.now() - phase2c_start).total_seconds()
                self._record_agent_metadata("search_agent", query_analysis_dict, search_result, phase2c_time, 0)
                self.logger.info(f"Search completed in {phase2c_time:.2f}s")
                
                if self.config.save_intermediate_results:
                    self._save_intermediate_result(workflow_id, "search_examples", search_result)
            else:
                self.logger.info("Phase 2.5: Skipping search agent (disabled)")
                search_result = {"examples": {}, "search_summary": "Search agent disabled"}
            
            # Phase 3: Design Exploration
            self.logger.info("Phase 3: Design Exploration")
            phase3_start = datetime.now()
            design_result = self.design_explorer.explore_design(query_result, data_result, None, None, viz_mapping_result.to_dict())
            phase3_time = (datetime.now() - phase3_start).total_seconds()
            
            # Record metadata for design explorer
            design_explorer_input = {"query_result": query_result, "data_result": data_result, "search_result": search_result, "viz_mapping": viz_mapping_result}
            self._record_agent_metadata("design_explorer", design_explorer_input, design_result, phase3_time, 0)
            self.logger.info(f"Design exploration completed in {phase3_time:.2f}s")
            
            if self.config.save_intermediate_results:
                self._save_intermediate_result(workflow_id, "design_exploration", design_result)
            
            # Phase 4: Code Generation and Self-Reflection
            self.logger.info("Phase 4: Code Generation and Self-Reflection")
            iterations = []
            final_success = False
            final_quality_score = 0.0

            if self.config.max_iterations == 0:
                self.logger.info("Iteration 0 mode: Direct code generation without refinement")
                iteration_result = self._execute_iteration_zero(
                    query_result, data_result, design_result,
                    search_result, workflow_id, workflow_context, query, viz_mapping_result
                )
                iterations.append(iteration_result)
                final_success = iteration_result.iteration_success
                final_quality_score = iteration_result.achieved_quality if final_success else 0.0
            else:
                # Normal iteration loop with self-reflection
                for iteration in range(self.config.max_iterations):
                    self.logger.info(f"Starting iteration {iteration + 1}")

                    iteration_result = self._execute_iteration(
                        iteration + 1, query_result, data_result, design_result,
                        search_result, iterations, workflow_id, workflow_context, query, viz_mapping_result
                    )

                    iterations.append(iteration_result)

                    # Check if both quality threshold AND TODO completion requirements are met
                    quality_met = iteration_result.achieved_quality >= self.config.quality_threshold

                    # Check TODO completion from the iteration result
                    todo_completion_rate = 0.0
                    if hasattr(iteration_result, 'todo_evaluation') and iteration_result.todo_evaluation:
                        completed = iteration_result.todo_evaluation.get('completed_count', 0)
                        total = iteration_result.todo_evaluation.get('total_count', 1)
                        todo_completion_rate = completed / total if total > 0 else 0.0

                    # Check if TODO evaluation is enabled
                    if self.config.enable_global_todo:
                        todos_completed = todo_completion_rate >= self.config.todo_completion_threshold
                    else:
                        todos_completed = True  # Skip TODO evaluation when disabled
                        self.logger.info("TODO evaluation disabled, skipping TODO completion check")

                    self.logger.info(f"Quality: {iteration_result.achieved_quality:.3f} (threshold: {self.config.quality_threshold})")
                    if self.config.enable_global_todo:
                        self.logger.info(f"TODO completion: {todo_completion_rate:.1%} (required: {self.config.todo_completion_threshold:.1%})")
                    else:
                        self.logger.info("TODO evaluation disabled for this run")

                    if quality_met and todos_completed:
                        self.logger.info("Both quality threshold and TODO completion requirements met!")
                        final_success = True
                        final_quality_score = iteration_result.achieved_quality
                        break
                    elif quality_met and not todos_completed:
                        self.logger.warning(f"Quality threshold met but TODOs incomplete ({todo_completion_rate:.1%}). Continuing...")
                    elif not quality_met and todos_completed:
                        self.logger.warning(f"All TODOs completed but quality below threshold ({iteration_result.achieved_quality:.3f}). Continuing...")
                    else:
                        self.logger.info(f"Neither quality nor TODO completion requirements met. Continuing...")

                    # Check if iteration was successful enough to continue
                    if not iteration_result.iteration_success:
                        self.logger.warning(f"Iteration {iteration + 1} failed, continuing to next iteration")
                        continue

                    final_quality_score = iteration_result.achieved_quality
            
            # If max iterations reached without meeting threshold, use the best result
            if not final_success and iterations:
                successful_iterations = [iter_result for iter_result in iterations if iter_result.iteration_success]
                if successful_iterations:
                    best_iteration = max(successful_iterations, key=lambda x: x.achieved_quality)
                    final_quality_score = best_iteration.achieved_quality
                    self.logger.info(f"Max iterations reached. Using best result with quality: {final_quality_score:.3f}")
                    # Mark as successful since we have a valid result
                    final_success = True
            
            # Calculate total execution time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Get final output path from debug results and copy to query directory
            final_output_path = None
            if iterations and iterations[-1].debug_result:
                debug_result = iterations[-1].debug_result
                if debug_result.success and debug_result.output_file:
                    final_output_path = debug_result.output_file
                    # Copy final result to query directory for easy access
                    if os.path.exists(final_output_path):
                        import shutil
                        # Use the correct query directory that was already created
                        final_query_path = self.query_dir / "final_result.png"
                        shutil.copy2(final_output_path, final_query_path)
                        self.logger.info(f"Final result saved to: {final_query_path}")
            
            # Create workflow summary
            workflow_summary = self._create_workflow_summary(
                query_result, data_result, design_result, iterations, final_success
            )
            
            # Create final workflow result
            workflow_result = WorkflowResult(
                workflow_id=workflow_id,
                query_analysis=query_result,
                data_processing=data_result,
                design_exploration=design_result,
                iterations=iterations,
                final_success=final_success,
                final_quality_score=final_quality_score,
                total_iterations=len(iterations),
                total_execution_time=total_time,
                final_output_path=final_output_path,
                workflow_summary=workflow_summary
            )
            
            # Save final workflow result
            if self.config.save_intermediate_results:
                self._save_workflow_result(workflow_result)
            
            # Save complete metadata
            self._save_complete_metadata(workflow_id)
            
            self.logger.info(f"Workflow completed in {total_time:.2f}s. Final quality: {final_quality_score:.3f}")
            return workflow_result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return self._create_error_result(workflow_id, query, str(e))
    
    def _execute_iteration_zero(self,
                              query_result: QueryAnalysisResult,
                              data_result: DataProcessingResult,
                              design_result: DesignExplorationResult,
                              search_result: Dict[str, Any],
                              workflow_id: str,
                              workflow_context: Dict[str, Any] = None,
                              original_query: str = None,
                              viz_mapping_result: VisualizationMapping = None) -> WorkflowIteration:
        """Execute iteration 0: Direct code generation without any self-reflection or refinement"""

        iteration_start = datetime.now()

        try:
            # Step 1: Direct Code Generation (no refinement)
            self.logger.info("Iteration 0: Direct Code Generation (No Self-Reflection)")

            code_gen_start = datetime.now()
            data_file_path = workflow_context.get('original_data_path') if workflow_context else None
            code_result = self.code_generator.generate_code(
                design_result, data_result, query_result, search_result,
                None,  # No code requirements/refinement
                str(self.query_dir),
                data_file_path=data_file_path,
                workflow_context=workflow_context,
                viz_mapping=viz_mapping_result
            )
            code_gen_time = (datetime.now() - code_gen_start).total_seconds()

            # Record metadata for code generator
            code_gen_input = {"design_result": design_result, "data_result": data_result, "code_requirements": None}
            self._record_agent_metadata("code_generator", code_gen_input, code_result, code_gen_time, 0)

            # Step 2: Direct Execute Generated Code (no debugging iterations)
            self.logger.info("Iteration 0: Direct Code Execution (No Debugging)")
            exec_start = datetime.now()

            # Construct data file path
            workflow_context = {"test_id": self.query_id} if self.query_id else {}
            data_file_path = self._construct_data_file_path(data_result, workflow_context)
            query_id = self.query_id

            # Execute code directly without debug iterations
            debug_result = self.debug_agent.execute_code(
                code_result,
                str(self.query_dir),
                data_file_path,
                query_id=query_id
            )
            exec_time = (datetime.now() - exec_start).total_seconds()

            # Record metadata for debug agent
            debug_input = {"code_result": code_result, "max_debug_iterations": 0}
            self._record_agent_metadata("debug_agent", debug_input, debug_result, exec_time, 0)

            # No visual evaluation or refinement for iteration 0
            visual_result = None
            refinement_result = None
            todo_evaluation = None
            achieved_quality = 0.0

            # Mark as successful only if code executed without errors
            iteration_success = debug_result.success

            if iteration_success:
                self.logger.info("Iteration 0: Code executed successfully (no quality evaluation)")
            else:
                self.logger.warning("Iteration 0: Code execution failed")

            return WorkflowIteration(
                iteration_number=0,
                code_generation_result=code_result,
                debug_result=debug_result,
                visual_assessment=visual_result,
                refinement_result=refinement_result,
                todo_evaluation=todo_evaluation,
                achieved_quality=achieved_quality,
                iteration_success=iteration_success,
                iteration_time=(datetime.now() - iteration_start).total_seconds()
            )

        except Exception as e:
            self.logger.error(f"Iteration 0 failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            # Return failed iteration result
            return WorkflowIteration(
                iteration_number=0,
                code_generation_result=None,
                debug_result=None,
                visual_assessment=None,
                refinement_result=None,
                todo_evaluation=None,
                achieved_quality=0.0,
                iteration_success=False,
                iteration_time=(datetime.now() - iteration_start).total_seconds()
            )

    def _execute_iteration(self, iteration_number: int,
                          query_result: QueryAnalysisResult,
                          data_result: DataProcessingResult,
                          design_result: DesignExplorationResult,
                          search_result: Dict[str, Any],
                          previous_iterations: List[WorkflowIteration],
                          workflow_id: str,
                          workflow_context: Dict[str, Any] = None,
                          original_query: str = None,
                          viz_mapping_result: VisualizationMapping = None) -> WorkflowIteration:
        """Execute a single iteration of the code generation and refinement loop"""
        
        iteration_start = datetime.now()
        
        try:
            # Step 1: Code Generation
            self.logger.info(f"Iteration {iteration_number}: Code Generation")
            
            code_requirements = None
            need_design_refinement = False
            
            if previous_iterations:
                last_iteration = previous_iterations[-1]
                if last_iteration.refinement_result:
                    # Analyze feedback to determine if it's design or implementation issue
                    feedback_text = " ".join(last_iteration.refinement_result.code_modifications).lower()
                    
                    design_keywords = ['color', 'layout', 'spacing', 'typography', 'aesthetic',
                                     'visual', 'appearance', 'style', 'design', 'look']
                    code_keywords = ['error', 'bug', 'crash', 'exception', 'syntax', 'import',
                                   'function', 'variable', 'data', 'calculation']

                    design_score = sum(1 for keyword in design_keywords if keyword in feedback_text)
                    code_score = sum(1 for keyword in code_keywords if keyword in feedback_text)
                    
                    if design_score > code_score and iteration_number <= 2:
                        need_design_refinement = True
                    else:
                        code_requirements = {
                            "feedback": last_iteration.refinement_result.code_modifications,
                            "priority_improvements": last_iteration.refinement_result.priority_improvements
                        }
            
            if need_design_refinement and previous_iterations:
                self.logger.info(f"Iteration {iteration_number}: Design Refinement")
                last_iteration = previous_iterations[-1]
                
                # Create design refinement feedback
                design_feedback = {
                    "visual_feedback": last_iteration.refinement_result.code_modifications,
                    "quality_issues": last_iteration.refinement_result.priority_improvements,
                    "target_quality": self.config.quality_threshold
                }
                
                # Re-run Design Explorer with feedback
                design_refine_start = datetime.now()
                design_result = self.design_explorer.refine_design(
                    design_result, design_feedback, query_result, data_result
                )
                design_refine_time = (datetime.now() - design_refine_start).total_seconds()
                self.logger.info(f"Design refinement completed in {design_refine_time:.2f}s")
            
            code_gen_start = datetime.now()
            data_file_path = workflow_context.get('original_data_path') if workflow_context else None
            code_result = self.code_generator.generate_code(
                design_result, data_result, query_result, search_result, code_requirements, str(self.query_dir), 
                data_file_path=data_file_path,
                workflow_context=workflow_context,
                viz_mapping=viz_mapping_result  # Pass visualization mapping for precise code generation
            )
            code_gen_time = (datetime.now() - code_gen_start).total_seconds()
            
            # Record metadata for code generator
            code_gen_input = {"design_result": design_result, "data_result": data_result, "code_requirements": code_requirements}
            self._record_agent_metadata("code_generator", code_gen_input, code_result, code_gen_time, iteration_number)
            
            self.logger.info(f"Iteration {iteration_number}: Debug and Execute Code")
            debug_start = datetime.now()
            workflow_context = {"test_id": self.query_id} if self.query_id else {}
            data_file_path = self._construct_data_file_path(data_result, workflow_context)
            query_id = self.query_id
            
            debug_result = self.debug_agent.execute_code(
                code_result, 
                str(self.query_dir), 
                data_file_path,
                query_id=query_id,
                data_path=self.config.data_path
            )
            
            if not debug_result.success:
                self.logger.info("Code execution failed, attempting debug...")
                debug_result = self.debug_agent.debug_code_if_needed(
                    code_result, debug_result, str(self.query_dir), data_file_path,
                    query_id=query_id, data_path=self.config.data_path
                )
            
            debug_time = (datetime.now() - debug_start).total_seconds()
            
            # Record metadata for debug agent
            debug_input = {"code_result": code_result, "query_dir": str(self.query_dir)}
            self._record_agent_metadata("debug_agent", debug_input, debug_result, debug_time, iteration_number)
            
            self.logger.info(f"Iteration {iteration_number}: Visual Evaluation")
            visual_eval_start = datetime.now()
            
            if debug_result.success and debug_result.output_file:
                visual_assessment = self.visual_evaluator.evaluate_visualization(
                    debug_result, 
                    data_result.processed_data,
                    original_query=original_query,
                    plotting_key_points=query_result.plotting_key_points
                )
                achieved_quality = visual_assessment.overall_score
                
                if achieved_quality < self.config.quality_threshold and iteration_number < self.config.max_iterations:
                    code_requirements = {
                        "feedback": visual_assessment.improvement_suggestions if hasattr(visual_assessment, 'improvement_suggestions') else visual_assessment.get('improvements', []),
                        "weaknesses": visual_assessment.weaknesses if hasattr(visual_assessment, 'weaknesses') else visual_assessment.get('issues', []),
                        "target_score": self.config.quality_threshold
                    }
            else:
                # Failed execution
                achieved_quality = 0.3
                visual_assessment = {"overall_score": achieved_quality, "error": "Execution failed"}
            
            visual_eval_time = (datetime.now() - visual_eval_start).total_seconds()
            
            # Record metadata for visual evaluator
            visual_eval_input = {"debug_result": debug_result, "processed_data": data_result.processed_data}
            self._record_agent_metadata("visual_evaluator", visual_eval_input, visual_assessment, visual_eval_time, iteration_number)
            
            todo_evaluation = None

            if self.config.enable_global_todo:
                self.logger.info(f"Iteration {iteration_number}: TODO Evaluation")
                todo_eval_start = datetime.now()

                if debug_result.success and debug_result.output_file:
                    todo_evaluation = self._evaluate_todo_completion_with_image(
                        query_result.global_todo_list,
                        debug_result.output_file,
                        data_result.processed_data,
                        original_query,
                        iteration_number
                    )

                    todo_eval_time = (datetime.now() - todo_eval_start).total_seconds()
                    self.logger.info(f"TODO evaluation completed in {todo_eval_time:.2f}s")
                    self.logger.info(f"TODO completion: {todo_evaluation['completed_count']}/{todo_evaluation['total_count']} items")

                    # Log incomplete TODOs
                    if todo_evaluation['incomplete_todos']:
                        self.logger.info(f"Found {len(todo_evaluation['incomplete_todos'])} incomplete TODO items:")
                        for todo in todo_evaluation['incomplete_todos']:
                            self.logger.info(f"  - {todo['task']}")
                            self.logger.info(f"    Reason: {todo.get('reason', 'Not completed')}")
                else:
                    todo_evaluation = {
                        'completed_count': 0,
                        'total_count': len(query_result.global_todo_list),
                        'incomplete_todos': query_result.global_todo_list,
                        'completion_feedback': "Execution failed, no image to evaluate TODOs"
                    }
                    self.logger.info("TODO evaluation skipped due to execution failure")
            else:
                todo_evaluation = {
                    'completed_count': 1,
                    'total_count': 1,
                    'incomplete_todos': [],
                    'completion_feedback': "TODO evaluation disabled for this run"
                }
                self.logger.info(f"Iteration {iteration_number}: TODO Evaluation disabled, skipping")
            
            iteration_time = (datetime.now() - iteration_start).total_seconds()
            iteration_success = debug_result.success

            refinement_feedback = None
            need_refinement = (hasattr(visual_assessment, 'improvement_suggestions') and achieved_quality < self.config.quality_threshold) or \
                            (todo_evaluation and todo_evaluation['incomplete_todos'])
            
            if need_refinement:
                suggestions = []
                if hasattr(visual_assessment, 'improvement_suggestions'):
                    suggestions.extend(visual_assessment.improvement_suggestions)
                    
                    # Add TODO completion feedback
                    if todo_evaluation and todo_evaluation['incomplete_todos']:
                        suggestions.append("INCOMPLETE TODO ITEMS:")
                        for todo in todo_evaluation['incomplete_todos']:
                            suggestions.append(f"• {todo['task']}")
                            if todo.get('reason'):
                                suggestions.append(f"  Reason: {todo['reason']}")
                        
                        if todo_evaluation.get('completion_feedback'):
                            suggestions.append(f"Overall TODO feedback: {todo_evaluation['completion_feedback']}")
                    
                    refinement_feedback = {
                        "refinement_id": f"refinement_{iteration_number}",
                        "original_assessment": visual_assessment,
                        "refinement_suggestions": suggestions,
                        "priority_improvements": getattr(visual_assessment, 'weaknesses', []) + \
                                           [todo['task'] for todo in (todo_evaluation['incomplete_todos'] if todo_evaluation else [])],
                        "code_modifications": suggestions,
                        "expected_improvements": {"overall_score": self.config.quality_threshold, "todo_completion": 1.0},
                        "refinement_rationale": "Improve visualization quality and complete remaining TODO items",
                        "implementation_difficulty": "medium",
                        "success_metrics": ["Quality score above threshold", "All TODO items completed"]
                    }
            
            iteration_result = WorkflowIteration(
                iteration_number=iteration_number,
                code_generation_result=code_result,
                debug_result=debug_result,
                visual_assessment={"overall_score": achieved_quality},
                refinement_result=refinement_feedback,
                todo_evaluation=todo_evaluation,
                achieved_quality=achieved_quality,
                iteration_success=iteration_success,
                iteration_time=iteration_time
            )
            
            # Save iteration result
            if self.config.save_intermediate_results:
                self._save_iteration_result(workflow_id, iteration_result)
            
            self.logger.info(f"Iteration {iteration_number} completed. Quality: {achieved_quality:.3f}")
            return iteration_result
            
        except Exception as e:
            self.logger.error(f"Iteration {iteration_number} failed: {str(e)}")
            return self._create_error_iteration(iteration_number, str(e))
    
    def _create_workflow_summary(self, query_result: QueryAnalysisResult,
                               data_result: DataProcessingResult,
                               design_result: DesignExplorationResult,
                               iterations: List[WorkflowIteration],
                               final_success: bool) -> Dict[str, Any]:
        """Create comprehensive workflow summary"""
        
        # Calculate quality progression
        quality_progression = [iteration.achieved_quality for iteration in iterations]
        
        # Analyze improvement patterns
        improvement_pattern = []
        for i in range(1, len(quality_progression)):
            improvement = quality_progression[i] - quality_progression[i-1]
            improvement_pattern.append(improvement)
        
        # Count successes and failures
        successful_iterations = sum(1 for iteration in iterations if iteration.iteration_success)
        failed_iterations = len(iterations) - successful_iterations
        
        # Calculate average iteration time
        avg_iteration_time = sum(iteration.iteration_time for iteration in iterations) / len(iterations) if iterations else 0
        
        # Analyze final code quality
        final_code_quality = iterations[-1].code_generation_result.code_quality_score if iterations else 0
        
        return {
            "workflow_overview": {
                "total_iterations": len(iterations),
                "successful_iterations": successful_iterations,
                "failed_iterations": failed_iterations,
                "final_success": final_success,
                "quality_threshold": self.config.quality_threshold,
                "average_iteration_time": avg_iteration_time
            },
            "quality_analysis": {
                "initial_quality": quality_progression[0] if quality_progression else 0,
                "final_quality": quality_progression[-1] if quality_progression else 0,
                "quality_progression": quality_progression,
                "improvement_pattern": improvement_pattern,
                "quality_improvement": quality_progression[-1] - quality_progression[0] if len(quality_progression) > 1 else 0
            },
            "component_analysis": {
                "query_confidence": query_result.confidence_score,
                "data_quality": data_result.data_quality_score,
                "design_quality": design_result.quality_metrics.get("overall_quality", 0),
                "final_code_quality": final_code_quality
            },
            "performance_metrics": {
                "total_execution_time": sum(iteration.iteration_time for iteration in iterations),
                "average_debug_time": sum(iteration.debug_result.execution_time for iteration in iterations) / len(iterations) if iterations else 0,
                "code_generation_efficiency": final_code_quality / avg_iteration_time if avg_iteration_time > 0 else 0
            },
            "improvement_insights": {
                "most_effective_iteration": max(range(len(iterations)), key=lambda i: iterations[i].achieved_quality) + 1 if iterations else 0,
                "biggest_quality_jump": max(improvement_pattern) if improvement_pattern else 0,
                "consistency_score": 1.0 - (sum(abs(imp) for imp in improvement_pattern) / len(improvement_pattern)) if improvement_pattern else 1.0
            }
        }
    
    def _save_intermediate_result(self, workflow_id: str, phase: str, result: Any, iteration: int = 0):
        """Save intermediate results to disk with iteration support"""
        
        try:
            phase_dir = self.query_dir / workflow_id / phase
            phase_dir.mkdir(parents=True, exist_ok=True)

            if hasattr(result, '__dict__'):
                result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
            else:
                result_dict = result
            
            if iteration > 0:
                output_file = phase_dir / f"{phase}_result_iteration_{iteration}.json"
            else:
                output_file = phase_dir / f"{phase}_result.json"
                
            with open(output_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            self.logger.debug(f"Saved intermediate result: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save intermediate result for {phase}: {str(e)}")
    
    def _save_iteration_result(self, workflow_id: str, iteration: WorkflowIteration):
        """Save iteration results to disk"""
        
        try:
            iteration_dir = self.query_dir / workflow_id / "iterations" / f"iteration_{iteration.iteration_number}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            iteration_dict = asdict(iteration)
            with open(iteration_dir / "iteration_summary.json", 'w') as f:
                json.dump(iteration_dict, f, indent=2, default=str)
            
            if iteration.code_generation_result.generated_code:
                with open(iteration_dir / "generated_code.py", 'w') as f:
                    f.write(iteration.code_generation_result.generated_code)

            if hasattr(iteration.debug_result, 'fixed_code') and iteration.debug_result.fixed_code and iteration.debug_result.fixed_code != iteration.code_generation_result.generated_code:
                with open(iteration_dir / "fixed_code.py", 'w') as f:
                    f.write(iteration.debug_result.fixed_code)
            
            # Copy output image if available
            if iteration.debug_result.output_file and os.path.exists(iteration.debug_result.output_file):
                import shutil
                shutil.copy2(iteration.debug_result.output_file, iteration_dir / "output.png")
            
            self.logger.debug(f"Saved iteration result: {iteration_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save iteration result: {str(e)}")
    
    def _save_workflow_result(self, workflow_result: WorkflowResult):
        """Save complete workflow result"""
        
        try:
            workflow_dir = self.query_dir / workflow_result.workflow_id
            workflow_dir.mkdir(parents=True, exist_ok=True)

            result_dict = asdict(workflow_result)
            with open(workflow_dir / "workflow_result.json", 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            with open(workflow_dir / "workflow_summary.json", 'w') as f:
                json.dump(workflow_result.workflow_summary, f, indent=2, default=str)
            
            # Copy final output if available
            if workflow_result.final_output_path and os.path.exists(workflow_result.final_output_path):
                import shutil
                shutil.copy2(workflow_result.final_output_path, workflow_dir / "final_output.png")
            
            self.logger.info(f"Saved workflow result: {workflow_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save workflow result: {str(e)}")
    
    def _create_error_result(self, workflow_id: str, query: str, error_msg: str) -> WorkflowResult:
        """Create error result when workflow fails"""
        
        # Create minimal results for error case
        error_query_result = QueryAnalysisResult(
            query_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            original_query=query,
            interpreted_intent="Error in workflow execution",
            visualization_type="unknown",
            plotting_key_points=["Error occurred during processing"],
            global_todo_list=[],
            success_criteria=[],
            complexity_assessment={"error": error_msg},
            confidence_score=0.0,
            processing_time=0.0
        )
        
        error_data_result = DataProcessingResult(
            processing_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            processed_data=pd.DataFrame(),
            data_insights={"error": error_msg},
            processing_steps=[],
            processing_time=0.0,
            original_data_path="error",
            data_summary={"error": error_msg},
            data_quality_score=0.0,
            issues_found=[error_msg],
            recommendations=[],
            metadata={"error": error_msg}
        )
        
        from ..agents.design_explorer import DesignExplorationResult, DesignSpecification
        
        error_design_result = DesignExplorationResult(
            exploration_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            primary_design=DesignSpecification(
                design_id="error",
                visualization_type="unknown",
                design_rationale="Error in workflow",
                color_scheme={},
                layout_specifications={},
                typography={},
                interactive_elements=[],
                accessibility_features=[],
                responsive_design={},
                aesthetic_choices={},
                user_experience_considerations=[],
                design_confidence=0.0,
                creation_time=0.0
            ),
            alternative_designs=[],
            design_recommendations=[],
            implementation_guidelines={},
            quality_metrics={"overall_quality": 0.0},
            potential_challenges=[error_msg],
            success_indicators=[],
            processing_time=0.0
        )
        
        return WorkflowResult(
            workflow_id=workflow_id,
            query_analysis=error_query_result,
            data_processing=error_data_result,
            design_exploration=error_design_result,
            iterations=[],
            final_success=False,
            final_quality_score=0.0,
            total_iterations=0,
            total_execution_time=0.0,
            final_output_path=None,
            workflow_summary={"error": error_msg}
        )
    
    def _create_error_iteration(self, iteration_number: int, error_msg: str) -> WorkflowIteration:
        """Create error iteration when iteration fails"""
        
        # Create minimal error results
        error_code_result = CodeGenerationResult(
            generation_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_code="# Error in code generation",
            code_quality_score=0.0,
            code_complexity={"error": error_msg},
            dependencies=[],
            code_documentation=f"Error: {error_msg}",
            error_handling=[],
            performance_optimizations=[],
            testing_suggestions=[],
            maintenance_notes=[],
            code_metrics={"error": error_msg},
            generation_time=0.0
        )
        
        error_debug_result = ExecutionResult(
            returncode=1,
            stdout="",
            stderr=error_msg,
            execution_time=0.0,
            success=False,
            output_file=None
        )
        
        error_visual_assessment = {
            "overall_score": 0.0,
            "error": error_msg
        }
        
        return WorkflowIteration(
            iteration_number=iteration_number,
            code_generation_result=error_code_result,
            debug_result=error_debug_result,
            visual_assessment=error_visual_assessment,
            refinement_result=None,
            todo_evaluation=None,  # No TODO evaluation in error case
            achieved_quality=0.0,
            iteration_success=False,
            iteration_time=0.0
        )
    
    def save_all_agent_behaviors(self):
        """Save all agent behaviors to files"""
        
        try:
            behavior_dir = self.session_dir / "agent_behaviors"
            behavior_dir.mkdir(parents=True, exist_ok=True)
            
            agents = [
                ("query_analyzer", self.query_analyzer),
                ("data_processor", self.data_processor),
                ("design_explorer", self.design_explorer),
                ("code_generator", self.code_generator),
                ("debug_agent", self.debug_agent),
                ("visual_evaluator", self.visual_evaluator)
            ]
            
            saved_files = []
            for agent_name, agent in agents:
                behavior_file = agent.save_behaviors_to_file(str(behavior_dir))
                if behavior_file:
                    saved_files.append(behavior_file)
                    self.logger.info(f"Saved behaviors for {agent_name}: {behavior_file}")
            
            # Create behavior summary
            summary_file = behavior_dir / "behavior_summary.json"
            summary_data = {
                "workflow_id": self.workflow_id,
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "agents": {
                    agent_name: {
                        "behavior_count": len(agent.behavior_history),
                        "session_id": agent.current_session_id
                    }
                    for agent_name, agent in agents
                },
                "saved_files": saved_files
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved behavior summary: {summary_file}")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Failed to save agent behaviors: {str(e)}")
            return []

    def _construct_data_file_path(self, data_result: DataProcessingResult, context: Dict[str, Any]) -> Optional[str]:
        """Resolve a concrete data file path from the processing result or benchmark context."""
        if (data_result.original_data_path and 
            data_result.original_data_path not in ["expert_generated_data", "DataFrame_input", "DataFrame"] and
            data_result.original_data_path.endswith(('.csv', '.json', '.xlsx', '.parquet'))):
            return data_result.original_data_path
            
        if context and 'test_id' in context:
            query_id = context['test_id']
            base_path = Path(self.config.data_path)
            possible_paths = [
                base_path / f"data/{query_id}/data.csv",
                base_path / f"data/{query_id}/data.json",
            ]
            
            if query_id == 98:
                csv_dir = base_path / f"data/{query_id}"
                if csv_dir.exists():
                    csv_files = list(csv_dir.glob("*.csv"))
                    if csv_files:
                        return str(csv_files[0])
            
            for path in possible_paths:
                if path.exists():
                    self.logger.info(f"Found data file: {path}")
                    return str(path)

            self.logger.warning(f"No data file found for query {query_id} in {base_path}")

        return None

    def _evaluate_todo_completion_with_image(self, todo_list: List[Dict], image_path: str, 
                                           data: Any, original_query: str, iteration_number: int) -> Dict[str, Any]:
        """
        Evaluate TODO completion using image analysis with LLM
        
        Args:
            todo_list: List of TODO items from query analysis
            image_path: Path to the generated visualization image
            data: Processed data
            original_query: Original query text
            iteration_number: Current iteration number
            
        Returns:
            Dictionary with completion analysis
        """
        try:
            self.logger.info(f"Evaluating TODO completion using image analysis")
            
            with open(image_path, "rb") as f:
                image_data = f.read()

            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            image_part = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                }
            }
            
            todo_text = "\n".join([f"{i+1}. {todo['task']} (Agent: {todo.get('agent', 'unknown')}, Priority: {todo.get('priority', 'medium')})"
                                 for i, todo in enumerate(todo_list)])
            
            evaluation_prompt = f"""
You are a visualization quality assessor. Analyze this generated plot image to determine if all the specified TODO requirements have been completed.

ORIGINAL QUERY:
{original_query}

TODO REQUIREMENTS TO CHECK:
{todo_text}

DATA CONTEXT:
- Data shape: {getattr(data, 'shape', 'unknown')}
- Data columns: {list(getattr(data, 'columns', []))}

EVALUATION TASK:
1. Look at the generated visualization image carefully
2. For each TODO item, determine if it has been completed based on what you see in the image
3. If a TODO is not completed, explain specifically what is missing or wrong
4. Provide specific actionable feedback for incomplete items

Please respond in this exact JSON format:
{{
    "overall_completion_score": 0.0-1.0,
    "completed_todos": [
        {{
            "todo_id": "todo_1",
            "task": "exact task text",
            "completed": true,
            "evidence": "what in the image shows this is completed"
        }}
    ],
    "incomplete_todos": [
        {{
            "todo_id": "todo_2", 
            "task": "exact task text",
            "completed": false,
            "reason": "specific reason why not completed",
            "actionable_feedback": "what needs to be done to complete this"
        }}
    ],
    "completion_feedback": "overall assessment and next steps"
}}
"""

            generation_config = {
                "max_output_tokens": 8000,
                "temperature": 0.1,
                "top_p": 0.8,
            }

            response = self.visual_evaluator._generate_with_usage(
                model=self.visual_evaluator.model_name,
                content=[
                    {"type": "text", "text": evaluation_prompt},
                    image_part
                ],
                generation_config=generation_config
            )
            
            response_text = response.text.strip()
            self.logger.info(f"TODO evaluation response length: {len(response_text)}")
            
            import json
            try:
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    # Try to find JSON object
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    json_text = response_text[json_start:json_end]

                evaluation_result = json.loads(json_text)

                if 'incomplete_todos' not in evaluation_result:
                    evaluation_result['incomplete_todos'] = []
                if 'completed_todos' not in evaluation_result:
                    evaluation_result['completed_todos'] = []
                
                evaluation_result['completed_count'] = len(evaluation_result['completed_todos'])
                evaluation_result['total_count'] = len(todo_list)
                
                self.logger.info(f"TODO evaluation successful: {evaluation_result['completed_count']}/{evaluation_result['total_count']} completed")
                return evaluation_result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse TODO evaluation JSON: {str(e)}")
                self.logger.error(f"Response text: {response_text[:500]}...")
                
                # Fallback response
                return {
                    'completed_count': 0,
                    'total_count': len(todo_list),
                    'incomplete_todos': [{'task': todo['task'], 'reason': 'Evaluation parsing failed'} for todo in todo_list],
                    'completion_feedback': f"TODO evaluation failed due to parsing error: {str(e)}"
                }
                
        except Exception as e:
            self.logger.error(f"TODO evaluation failed: {str(e)}")
            return {
                'completed_count': 0,
                'total_count': len(todo_list),
                'incomplete_todos': [{'task': todo['task'], 'reason': f'Evaluation error: {str(e)}'} for todo in todo_list],
                'completion_feedback': f"TODO evaluation failed: {str(e)}"
            }

    def _safe_load_json_data(self, file_path: str) -> pd.DataFrame:
        """Safely load JSON data, handling special cases like Sankey JSON structures"""
        import json
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Check if this is a Plotly Sankey JSON structure
            if self._is_sankey_json(json_data):
                # For Sankey JSON, create a representative DataFrame with key info
                return self._create_sankey_dataframe(json_data)
            else:
                # Try normal JSON to DataFrame conversion
                return pd.read_json(file_path)
                
        except Exception as e:
            # If all else fails, create a minimal DataFrame
            self.logger.warning(f"Failed to load JSON data from {file_path}: {str(e)}")
            return pd.DataFrame({'file_path': [str(file_path)], 'error': [str(e)]})
    
    def _is_sankey_json(self, json_data: dict) -> bool:
        """Check if JSON data represents a Plotly Sankey diagram"""
        try:
            # Check for Plotly structure with data and layout
            if 'data' in json_data and isinstance(json_data['data'], list):
                if len(json_data['data']) > 0:
                    first_data = json_data['data'][0]
                    # Check for Sankey-specific keys
                    if (first_data.get('type') == 'sankey' and 
                        'node' in first_data and 
                        'link' in first_data):
                        return True
            return False
        except Exception:
            return False
    
    def _create_sankey_dataframe(self, json_data: dict) -> pd.DataFrame:
        """Create a representative DataFrame from Sankey JSON data"""
        try:
            sankey_data = json_data['data'][0]
            
            # Extract node and link information
            node_labels = sankey_data['node']['label']
            link_sources = sankey_data['link']['source']
            link_targets = sankey_data['link']['target']
            link_values = sankey_data['link']['value']
            
            # Create a DataFrame representing the links
            links_df = pd.DataFrame({
                'source_index': link_sources,
                'target_index': link_targets,
                'source_label': [node_labels[i] for i in link_sources],
                'target_label': [node_labels[i] for i in link_targets],
                'value': link_values
            })
            
            return links_df
            
        except Exception as e:
            self.logger.warning(f"Failed to create Sankey DataFrame: {str(e)}")
            # Return a minimal DataFrame with basic info
            return pd.DataFrame({
                'data_type': ['sankey'],
                'node_count': [len(json_data.get('data', [{}])[0].get('node', {}).get('label', []))],
                'link_count': [len(json_data.get('data', [{}])[0].get('link', {}).get('source', []))]
            })

    def _collect_llm_statistics(self) -> int:
        """Collect total LLM calls from all agents"""
        total_calls = 0
        agents = [
            self.query_analyzer, self.data_processor, self.viz_mapping_agent,
            self.design_explorer, self.code_generator, self.debug_agent,
            self.visual_evaluator
        ]

        # Add search agent if enabled
        if hasattr(self, 'search_agent') and self.search_agent:
            agents.append(self.search_agent)

        for agent in agents:
            if hasattr(agent, 'performance_metrics'):
                agent_calls = agent.performance_metrics.get('total_requests', 0)
                total_calls += agent_calls
                self.logger.info(f"Agent {agent.agent_id}: {agent_calls} LLM calls")

        self.logger.info(f"Total LLM calls across all agents: {total_calls}")
        return total_calls

    def cleanup(self):
        """Clean up resources"""

        try:
            # Save all agent behaviors before cleanup
            self.save_all_agent_behaviors()

            # Log final statistics
            total_llm_calls = self._collect_llm_statistics()
            self.logger.info(f"Final workflow statistics: {total_llm_calls} total LLM calls, TODO enabled: {self.config.enable_global_todo}")

            # Clean up debug agent resources
            if hasattr(self.debug_agent, 'cleanup'):
                self.debug_agent.cleanup()

            self.logger.info("Workflow orchestrator cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()
