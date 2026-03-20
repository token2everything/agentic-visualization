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
Prompt Builder

Constructs structured, token-efficient prompts for data analysis
tasks within the CoDA pipeline.
"""

from typing import Dict, Any, List
import json

class PromptBuilder:
    """
    Build prompts that actually help LLM understand data.
    No special cases, no verbosity.
    """
    
    @staticmethod
    def build_data_section(metadata: Dict[str, Any]) -> str:
        """
        Convert metadata to LLM-readable format.
        Good taste: show what matters, hide what doesn't.
        """
        
        # Use the new unified metadata structure
        source = metadata.get("source", {})
        data_tables = metadata.get("data", [])
        
        prompt = f"""
DATA STRUCTURE:
File: {source.get('file', 'unknown')} ({source.get('type', 'unknown')})
Tables/Sheets: {source.get('tables', 1)}

"""
        
        # For each data container (table/sheet)
        for idx, table in enumerate(data_tables):
            name = table.get("name", f"table_{idx}")
            shape = table.get("shape", (0, 0))
            columns = table.get("columns", [])
            patterns = table.get("patterns", {})
            head = table.get("head", [])
            
            prompt += f"""
{name.upper()} ({shape[0]} rows × {shape[1]} cols):
  Columns: {', '.join(columns)}
  Numeric: {', '.join(patterns.get('numeric_cols', [])) or 'None'}
  Categorical: {', '.join(patterns.get('categorical_cols', [])) or 'None'}
  Has nulls: {patterns.get('has_nulls', False)}
  
  Sample (first 3 rows):
"""
            # Smart sample display - show only key-value pairs, not full dict
            for i, row in enumerate(head[:3], 1):
                # Show only first 3 columns to save tokens
                sample_cols = list(row.keys())[:3]
                sample_vals = [f"{k}={row[k]}" for k in sample_cols]
                if len(row) > 3:
                    sample_vals.append(f"...{len(row)-3} more cols")
                prompt += f"    Row {i}: {', '.join(sample_vals)}\n"
            
            if idx < len(data_tables) - 1:
                prompt += "\n"
        
        return prompt
    
    @staticmethod
    def build_analysis_prompt(query: str, metadata: Dict[str, Any], 
                             expert_instruction: str = None) -> str:
        """Build a complete analysis prompt from query and metadata."""
        
        # Data section - same for all file types
        data_section = PromptBuilder.build_data_section(metadata)
        
        # Expert instruction if provided
        expert_section = ""
        if expert_instruction:
            expert_section = f"\nEXPERT GUIDANCE:\n{expert_instruction}\n"
        
        # The prompt - clear and focused
        return f"""
Analyze this visualization request with the provided data structure.

USER REQUEST:
{query}
{expert_section}
{data_section}

ANALYSIS TASKS:
1. What chart type best fits this data structure?
2. Which columns should be used for x, y, color, size?
3. Any data transformations needed (groupby, pivot, merge)?
4. What's the single clearest way to show this information?

Output JSON with:
- visualization_type: specific chart type
- data_mapping: {{x: column, y: column, ...}}
- transformations: [list of operations needed]
- implementation_notes: key points for code generation
"""

    @staticmethod
    def compare_prompts():
        """
        Demonstration: Old verbose prompt vs new concise prompt
        """
        
        # Sample metadata
        test_meta = {
            "source": {"file": "sales.csv", "type": ".csv", "tables": 1},
            "data": [{
                "name": "main",
                "shape": (1000, 5),
                "columns": ["date", "product", "sales", "region", "profit"],
                "patterns": {
                    "numeric_cols": ["sales", "profit"],
                    "categorical_cols": ["product", "region"],
                    "has_nulls": False
                },
                "head": [
                    {"date": "2024-01", "product": "A", "sales": 100, "region": "North", "profit": 20},
                    {"date": "2024-01", "product": "B", "sales": 150, "region": "South", "profit": 35},
                    {"date": "2024-02", "product": "A", "sales": 120, "region": "North", "profit": 25}
                ]
            }]
        }
        
        # Old way: dumps entire sample_data dict
        old_prompt_size = len(str(test_meta["data"][0]["head"]) * 10)  # 10 rows
        
        # New way: structured display
        new_prompt = PromptBuilder.build_data_section(test_meta)
        new_prompt_size = len(new_prompt)
        
        print(f"Old prompt size: ~{old_prompt_size} chars")
        print(f"New prompt size: {new_prompt_size} chars")
        print(f"Reduction: {(1 - new_prompt_size/old_prompt_size)*100:.1f}%")
        
        return new_prompt


if __name__ == "__main__":
    # Test it
    builder = PromptBuilder()
    
    # Test with multi-table SQLite
    sqlite_meta = {
        "source": {"file": "database.sqlite", "type": ".sqlite", "tables": 3},
        "data": [
            {
                "name": "users",
                "shape": (5000, 10),
                "columns": ["id", "name", "email", "created_at", "status"],
                "patterns": {
                    "numeric_cols": ["id"],
                    "categorical_cols": ["status"],
                    "has_nulls": True
                },
                "head": [
                    {"id": 1, "name": "Alice", "email": "alice@example.com", "created_at": "2024-01-01", "status": "active"},
                    {"id": 2, "name": "Bob", "email": "bob@example.com", "created_at": "2024-01-02", "status": "inactive"}
                ]
            },
            {
                "name": "orders",
                "shape": (15000, 8),
                "columns": ["order_id", "user_id", "product_id", "amount", "date"],
                "patterns": {
                    "numeric_cols": ["order_id", "user_id", "product_id", "amount"],
                    "categorical_cols": [],
                    "has_nulls": False
                },
                "head": [
                    {"order_id": 1001, "user_id": 1, "product_id": 501, "amount": 99.99, "date": "2024-01-15"}
                ]
            }
        ]
    }
    
    prompt = builder.build_data_section(sqlite_meta)
    print("SQLite Prompt:")
    print(prompt)
    print(f"\nTotal length: {len(prompt)} chars")