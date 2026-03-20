"""
Data Processor Agent

Parses multiple data formats, infers schema and statistics,
and prepares clean data structures for downstream visualization agents.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from .base import BaseAgent

# Import utilities from the new package structure
from ..utils.prompt_builder import PromptBuilder
from ..utils.meta_extractor import SmartMetaExtractor

@dataclass
class DataProcessingResult:
    """Result from data processing."""
    processing_id: str
    processed_data: pd.DataFrame
    data_insights: Dict[str, Any]
    processing_steps: List[str]
    processing_time: float
    original_data_path: str = ""
    data_summary: Dict[str, Any] = None
    data_quality_score: float = 0.8
    issues_found: List[str] = None
    recommendations: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data_summary is None:
            self.data_summary = self.data_insights
        if self.issues_found is None:
            self.issues_found = []
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}
    

class DataProcessor(BaseAgent):
    """
    Data Processor agent that handles diverse input formats and prepares
    data for the visualization pipeline.
    """

    def __init__(self, model_name: str, agent_id: str = "data_processor", config=None):
        super().__init__(agent_id, config)
        self.model_name = model_name
        self.persona = "Data processing expert"
        self.meta_extractor = SmartMetaExtractor()
        self.prompt_builder = PromptBuilder()
        
    def process_data(self, data_input: Union[str, pd.DataFrame],
                    todo_requirements: List[Dict[str, Any]],
                    context: Optional[Dict[str, Any]] = None) -> DataProcessingResult:
        """
        Process input data and return a cleaned, analysis-ready result.
        """
        start_time = datetime.now()
        
        # Step 1: Get DataFrame (everything becomes DataFrame)
        if isinstance(data_input, str):
            # Extract metadata instead of uploading
            metadata = self.meta_extractor.extract(data_input)
            # Load actual data for processing
            df = self._load_dataframe(data_input)
        elif isinstance(data_input, pd.DataFrame):
            df = data_input
            # Generate metadata from DataFrame
            metadata = self._dataframe_to_metadata(df)
        elif data_input is None:
            # Expert-generated data scenario - no external data file
            # Generate intelligent data based on query requirements
            df = self._generate_intelligent_data(todo_requirements, context)
            metadata = self._dataframe_to_metadata(df)
        else:
            raise ValueError(f"Data input must be path, DataFrame, or None (for expert-generated data), not {type(data_input)}")
            
        # Step 2: Analyze with clean prompt
        analysis = self._analyze_data(metadata, todo_requirements)
        
        # Step 3: Apply processing
        processed_df = self._apply_processing(df, analysis['processing_steps'])
        
        from .data_proxy import DataProxy
        
        return DataProcessingResult(
            processing_id=f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            processed_data=DataProxy(processed_df),  # Smart proxy with full DataFrame interface
            data_insights=analysis['insights'],
            processing_steps=analysis['processing_steps'],
            processing_time=(datetime.now() - start_time).total_seconds(),
            # Compatibility fields
            original_data_path=data_input if isinstance(data_input, str) else "DataFrame",
            data_summary=analysis['insights'],
            data_quality_score=0.9,
            issues_found=analysis['insights'].get('quality_issues', []),
            recommendations=[],
            metadata=metadata
        )
        
    def _analyze_data(self, metadata: Dict[str, Any],
                     todo_requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data and return processing steps and insights."""
        data_section = self.prompt_builder.build_data_section(metadata)
        
        # Simple TODO list
        todo_text = "\n".join([f"- {t['task']}" for t in todo_requirements[:5]])  # Max 5, not 20
        
        # The prompt - clear and focused
        prompt = f"""
Analyze this data for visualization.

{data_section}

TASKS TO COMPLETE:
{todo_text}

ANALYSIS NEEDED:
1. What transformations are required? (groupby, pivot, filter)
2. Which columns are key for visualization?
3. Any data quality issues to fix?
4. What's the simplest way to prepare this data?

Output JSON:
{{
    "processing_steps": [
        "step 1: specific transformation",
        "step 2: another transformation"
    ],
    "insights": {{
        "key_columns": ["col1", "col2"],
        "aggregations_needed": ["sum sales by region"],
        "quality_issues": ["nulls in X column"]
    }},
    "visualization_hint": "best chart type for this data"
}}
"""
        
        try:
            response = self._generate_with_usage(
                model=self.model_name,
                content=prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8
                }
            )
            
            # Parse response
            text = response.text.strip()
            if "```json" in text:
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_text = text[json_start:json_end]
            else:
                json_text = text
                
            return json.loads(json_text)
            
        except Exception as e:
            self.logger.warning(f"Analysis failed: {e}")
            return {
                "processing_steps": ["Use data as-is"],
                "insights": {"key_columns": list(metadata['data'][0]['columns'][:3])},
                "visualization_hint": "scatter"
            }
            
    def _apply_processing(self, df: pd.DataFrame, steps: List[str]) -> pd.DataFrame:
        """Apply the specified processing steps to the DataFrame."""
        result = df.copy()
        
        for step in steps:
            step_lower = step.lower()
            
            # Simple pattern matching for common operations
            if 'group' in step_lower and 'by' in step_lower:
                # Extract groupby column (simple heuristic)
                cols = [col for col in df.columns if col.lower() in step_lower]
                if cols:
                    # Simple aggregation
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        result = df.groupby(cols[0])[numeric_cols[0]].sum().reset_index()
                        
            elif 'filter' in step_lower:
                # Keep as is - too complex to parse reliably
                pass
                
            elif 'drop' in step_lower and 'null' in step_lower:
                result = result.dropna()
                
            elif 'sort' in step_lower:
                # Sort by first numeric column descending
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = result.sort_values(numeric_cols[0], ascending=False)
                    
        return result
        
    def _load_dataframe(self, path: str) -> pd.DataFrame:
        """Load a data file as a DataFrame."""
        # Reuse the extractor's loading logic
        from pathlib import Path
        path = Path(path)
        
        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
            
            # Let LLM intelligently convert complex JSON to DataFrame
            return self._llm_json_to_dataframe(data, str(path))
        elif path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif path.suffix in ['.sqlite', '.db']:
            import sqlite3
            conn = sqlite3.connect(path)
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if not tables.empty:
                df = pd.read_sql_query(f"SELECT * FROM {tables.iloc[0]['name']}", conn)
                conn.close()
                return df
            conn.close()
            return pd.DataFrame()
        else:
            # Try CSV as default
            return pd.read_csv(path)
            
    def _dataframe_to_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert DataFrame to metadata format"""
        return {
            "source": {
                "file": "DataFrame",
                "type": ".dataframe",
                "tables": 1
            },
            "data": [{
                "name": "main",
                "shape": df.shape,
                "columns": list(df.columns),
                "head": df.head(3).to_dict('records'),
                "patterns": {
                    "numeric_cols": [col for col in df.columns 
                                   if pd.api.types.is_numeric_dtype(df[col])],
                    "categorical_cols": [col for col in df.columns 
                                       if pd.api.types.is_object_dtype(df[col])],
                    "has_nulls": df.isnull().any().any()
                }
            }]
        }
    
    def _llm_json_to_dataframe(self, json_data: dict, file_path: str) -> pd.DataFrame:
        """
        Let LLM intelligently convert complex JSON to DataFrame.
        Handles any JSON structure - Plotly configs, nested data, etc.
        """
        try:
            # First try simple approaches
            if isinstance(json_data, list):
                return pd.DataFrame(json_data)
            
            # For complex structures, ask LLM to analyze and convert
            # Analyze JSON structure deeply
            def get_json_structure(obj, max_depth=4, current_depth=0):
                """Get JSON structure without all the data"""
                if current_depth >= max_depth:
                    return "..."
                
                if isinstance(obj, dict):
                    structure = {}
                    for key, value in obj.items():
                        if isinstance(value, (list, tuple)) and len(value) > 0:
                            # Show list structure with first item as example
                            structure[key] = [get_json_structure(value[0], max_depth, current_depth+1)] + [f"... {len(value)-1} more items"]
                        elif isinstance(value, dict):
                            structure[key] = get_json_structure(value, max_depth, current_depth+1)
                        else:
                            # For primitive values, show type and example
                            structure[key] = f"<{type(value).__name__}> e.g., {str(value)[:50]}"
                    return structure
                elif isinstance(obj, (list, tuple)) and len(obj) > 0:
                    return [get_json_structure(obj[0], max_depth, current_depth+1)] + [f"... {len(obj)-1} more items"]
                else:
                    return f"<{type(obj).__name__}>"
            
            json_structure = get_json_structure(json_data)
            
            prompt = f"""
You are a data expert. Understand this JSON structure and convert it to the most useful pandas DataFrame for visualization.

JSON STRUCTURE (complete schema):
{json.dumps(json_structure, indent=2)}

File: {file_path}

Key insights about this data:
- Total keys at root: {len(json_data.keys()) if isinstance(json_data, dict) else 'N/A (list)'}
- Data type: {type(json_data).__name__}
{f"- First level keys: {list(json_data.keys())[:10]}" if isinstance(json_data, dict) else ""}

This JSON contains visualization data. Your task:
1. Understand the complete data structure and relationships
2. Identify the core data for plotting (nodes, links, values, categories, etc.)
3. Extract it into a clean tabular format optimal for matplotlib
4. Generate Python code to create the DataFrame

Generate ONLY executable Python code that:
1. Takes the variable `json_data` (the parsed JSON)
2. Extracts the most useful data for visualization  
3. Returns a pandas DataFrame named `df`
4. Handles this specific JSON structure intelligently

For Sankey/network data: extract source-target-value relationships
For nested data: flatten appropriately for plotting
For configuration data: extract the actual data points

Example output format:
```python
import pandas as pd
import json

# Extract data from JSON structure  
if 'data' in json_data and json_data['data']:
    # Your intelligent extraction logic here
    df = pd.DataFrame(extracted_data)
else:
    # Fallback extraction
    df = pd.json_normalize(json_data)
```

Output ONLY the Python code:
"""
            
            # Use base generate method
            response = self._generate_with_usage(
                model=self.model_name,
                content=prompt,
                generation_config={
                    "temperature": 0.7
                }
            )
            
            code = response.text.strip()
            
            # Clean code
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0]
            elif '```' in code:
                code = code.split('```')[1].split('```')[0]
            
            # Execute LLM-generated conversion code
            exec_globals = {
                'pd': pd, 
                'json': json, 
                'np': np,
                'json_data': json_data
            }
            exec_locals = {}
            
            exec(code, exec_globals, exec_locals)
            
            # Return the generated DataFrame
            if 'df' in exec_locals and isinstance(exec_locals['df'], pd.DataFrame):
                return exec_locals['df']
                
        except Exception as e:
            self.logger.warning(f"LLM JSON conversion failed: {e}, using fallback")
        
        # Fallback to simple normalization
        try:
            return pd.json_normalize(json_data)
        except:
            # Ultimate fallback - convert to single-row DataFrame
            return pd.DataFrame([json_data])
        
    def _send_message(self, message) -> any:
        """Send message to LLM - reuse base agent functionality"""
        return self._generate_with_usage(model=self.model_name, content=message.content)
    
    def _generate_intelligent_data(self, todo_requirements: List[Dict[str, Any]], 
                                   context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Use LLM to generate appropriate synthetic data based on query requirements.
        """
        # Extract keywords from TODOs to understand data needs
        todo_text = ' '.join([str(todo.get('task', '')) for todo in todo_requirements])
        
        # Get expert instruction if available for better understanding
        expert_instruction = context.get('expert_instruction', '') if context else ''
        query_text = expert_instruction or todo_text
        
        # Ask LLM to generate simple working data
        prompt = f"""Create simple data for a matplotlib visualization.

The visualization requirements are:
{query_text}

TODO items from analysis:
{todo_text}

Generate Python code that creates the RIGHT data (pandas DataFrame) that works for this specific plot.

Deep understanding approach:
1. ANALYZE the visualization requirements carefully
2. UNDERSTAND what type of data this plot needs
3. DETERMINE the appropriate data structure and format
4. DECIDE the optimal number of data points based on plot type:
   - Time series: enough points to show trends (maybe 30-100)
   - Scatter plot: sufficient for patterns (maybe 50-200) 
   - Bar chart: logical categories (maybe 5-20)
   - Histogram: enough for distribution (maybe 100-1000)
   - Heatmap: appropriate grid size (maybe 10x10 to 20x20)
   - Network/correlation: based on complexity needed

Requirements:
- Import pandas as pd and numpy as np
- Create a DataFrame variable named 'df'
- Column names MUST match what the visualization actually needs
- Data format MUST be compatible with the plot type
- Data volume MUST be appropriate for the specific visualization
- Values should be realistic and demonstrate the intended pattern

Output ONLY executable Python code, no explanations:
"""
        
        try:
            # Get LLM to generate data creation code
            response = self._generate_with_usage(
                model=self.model_name,
                content=prompt,
                generation_config={
                    "temperature": 0.7
                }
            )
            
            # Extract and execute the code
            code = response.text.strip()
            
            # Remove markdown if present
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0]
            elif '```' in code:
                code = code.split('```')[1].split('```')[0]
            
            # Create a safe execution environment
            exec_globals = {
                'pd': pd,
                'np': np,
                'datetime': datetime,
                'random': np.random
            }
            exec_locals = {}
            
            # Execute the generated code with column conflict protection
            try:
                exec(code, exec_globals, exec_locals)
            except ValueError as ve:
                if "already exists" in str(ve) or "cannot insert" in str(ve):
                    self.logger.warning(f"Column conflict detected: {ve}")
                    # Add column name deduplication to the code
                    code_with_dedup = code + """
# Auto-fix for column conflicts
if 'df' in locals() and hasattr(df, 'columns'):
    # Make column names unique
    df.columns = pd.io.common.dedup_names(df.columns, is_potential_multiindex=False)
"""
                    exec_locals = {}  # Reset locals
                    exec(code_with_dedup, exec_globals, exec_locals)
                else:
                    raise ve
            
            # Find the DataFrame in the executed code
            for var_name, var_value in exec_locals.items():
                if isinstance(var_value, pd.DataFrame):
                    return var_value
            
            # If no DataFrame found directly, try to find 'df' or 'data'
            for name in ['df', 'data', 'dataset', 'DataFrame']:
                if name in exec_locals and isinstance(exec_locals[name], pd.DataFrame):
                    return exec_locals[name]
            
            # Fallback: try to evaluate the last line as DataFrame
            last_line = code.strip().split('\n')[-1]
            if 'DataFrame' in last_line:
                return eval(last_line, exec_globals, exec_locals)
                
        except Exception as e:
            self.logger.warning(f"Failed to generate data via LLM: {e}")
            # Try alternative approaches for common errors
            if "already exists" in str(e) or "cannot insert" in str(e):
                # Try with explicit column names
                try:
                    simple_prompt = f"""
Generate basic data for the visualization. Create a simple DataFrame with unique column names:

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Create simple data that works
df = pd.DataFrame({{
    'category': ['A', 'B', 'C', 'D', 'E'][:5],
    'values': np.random.uniform(10, 100, 5),
    'group': np.random.choice(['X', 'Y'], 5)
}})
"""
                    
                    response = self._generate_with_usage(
                        model=self.model_name,
                        content=simple_prompt,
                        generation_config={"temperature": 0.3}
                    )
                    
                    code = response.text.strip()
                    if '```python' in code:
                        code = code.split('```python')[1].split('```')[0]
                    elif '```' in code:
                        code = code.split('```')[1].split('```')[0]
                    
                    exec_globals = {'pd': pd, 'np': np}
                    exec_locals = {}
                    exec(code, exec_globals, exec_locals)
                    
                    if 'df' in exec_locals and isinstance(exec_locals['df'], pd.DataFrame):
                        return exec_locals['df']
                        
                except Exception as e2:
                    self.logger.warning(f"Fallback data generation also failed: {e2}")
            
            # Continue to ultimate fallback
            pass
        
        # Ultimate fallback: simple general-purpose data
        return pd.DataFrame({
            'x': range(20),
            'y': np.random.randn(20).cumsum() + 10,
            'category': np.random.choice(['A', 'B', 'C'], 20),
            'value': np.random.uniform(10, 100, 20)
        })
        
    def _generate_smart_data_summary(self, df: pd.DataFrame) -> str:
        """Generate comprehensive but compact data summary for all column types"""
        import io
        
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Column types and examples
        summary_parts.append("\nColumn Information:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric column stats
                stats = df[col].describe()
                example_str = f"range: [{stats['min']:.2f}, {stats['max']:.2f}], mean: {stats['mean']:.2f}"
            elif pd.api.types.is_object_dtype(df[col]):
                # Text/categorical column stats
                unique_count = df[col].nunique()
                if unique_count <= 10:
                    unique_values = list(df[col].dropna().unique())[:5]  # Show first 5
                    example_str = f"unique: {unique_count}, values: {unique_values}"
                else:
                    top_values = df[col].value_counts().head(3).index.tolist()
                    example_str = f"unique: {unique_count}, top: {top_values}"
            else:
                # Other types (datetime, etc.)
                unique_count = df[col].nunique()
                example_str = f"unique: {unique_count}"
            
            summary_parts.append(f"  - {col} ({dtype}): {non_null} non-null, {example_str}")
        
        # Data preview (head + tail without full values)
        summary_parts.append("\nData Preview:")
        summary_parts.append("Head (3 rows):")
        head_dict = df.head(3).to_dict('records')
        for i, row in enumerate(head_dict):
            row_preview = {k: (f"{v:.2f}" if isinstance(v, (int, float)) else str(v)[:20]) for k, v in row.items()}
            summary_parts.append(f"  Row {i}: {row_preview}")
        
        if len(df) > 6:  # Only show tail if there are enough rows
            summary_parts.append("Tail (3 rows):")
            tail_dict = df.tail(3).to_dict('records')
            for i, row in enumerate(tail_dict, len(df)-3):
                row_preview = {k: (f"{v:.2f}" if isinstance(v, (int, float)) else str(v)[:20]) for k, v in row.items()}
                summary_parts.append(f"  Row {i}: {row_preview}")
        
        return "\n".join(summary_parts)