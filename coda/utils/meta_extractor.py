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
Meta Extractor

Extracts structured metadata from CSV, JSON, Excel, Parquet, and SQLite files
for use by the CoDA data processing pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import json
import sqlite3

class MetaExtractor:
    """
    Unified extractor for all supported data formats.
    "Good taste means eliminating special cases, not adding them."
    """
    
    def extract(self, path: str) -> Dict[str, Any]:
        """
        Extract meta info from any data file.
        No if/elif chains. Data is data.
        """
        path = Path(path)
        
        # Core principle: Everything becomes DataFrames (plural for multi-table)
        data_info = self._load_data(path)
        
        # Universal meta structure - same for ALL file types
        meta = {
            # File context - LLM needs to know what it's dealing with
            "source": {
                "file": path.name,
                "type": path.suffix.lower(),
                "tables": data_info.get("tables", 1)  # How many data containers
            },
            
            # Data structure - could be single or multiple
            "data": []
        }
        
        # Process each data container (table/sheet/etc)
        for name, df in data_info["dataframes"].items():
            meta["data"].append({
                "name": name,
                "shape": df.shape,
                "columns": list(df.columns),
                "head": df.head(3).to_dict('records'),
                "patterns": self._extract_patterns(df)
            })
        
        return meta
    
    def _load_data(self, path: Path) -> Dict[str, Any]:
        """
        Load data from any file type into DataFrames.
        Returns dict with 'dataframes' and metadata.
        """
        # Map extensions to loaders - data structure, not code
        loaders = {
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.parquet': self._load_parquet,
            '.sqlite': self._load_sqlite,
            '.db': self._load_sqlite,
        }
        
        loader = loaders.get(path.suffix.lower(), self._load_csv)
        return loader(path)
    
    def _load_csv(self, path: Path) -> Dict[str, Any]:
        """Load CSV - always single table"""
        return {
            "dataframes": {"main": pd.read_csv(path)},
            "tables": 1
        }
    
    def _load_parquet(self, path: Path) -> Dict[str, Any]:
        """Load Parquet - always single table"""
        return {
            "dataframes": {"main": pd.read_parquet(path)},
            "tables": 1
        }
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON - could be nested"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        df = self._json_to_dataframe(data)
        return {
            "dataframes": {"main": df},
            "tables": 1
        }
    
    def _load_excel(self, path: Path) -> Dict[str, Any]:
        """Load Excel - could have multiple sheets"""
        xl_file = pd.ExcelFile(path)
        dataframes = {}
        
        for sheet_name in xl_file.sheet_names:
            dataframes[sheet_name] = pd.read_excel(path, sheet_name=sheet_name)
        
        return {
            "dataframes": dataframes,
            "tables": len(xl_file.sheet_names)
        }
    
    def _load_sqlite(self, path: Path) -> Dict[str, Any]:
        """Load SQLite - could have multiple tables"""
        conn = sqlite3.connect(path)
        
        # Get all tables
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", 
            conn
        )
        
        dataframes = {}
        for table_name in tables['name']:
            dataframes[table_name] = pd.read_sql_query(
                f"SELECT * FROM {table_name}", 
                conn
            )
        
        conn.close()
        return {
            "dataframes": dataframes,
            "tables": len(tables)
        }
    
    def _json_to_dataframe(self, data: Any) -> pd.DataFrame:
        """JSON to DataFrame - handle both array and object formats"""
        # Normalize to DataFrame - let pandas handle complexity
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Plotly Sankey or nested structure
            if 'data' in data and isinstance(data['data'], list):
                # Extract the meaningful part
                return self._extract_plotly_data(data)
            else:
                # Single record
                return pd.DataFrame([data])
        else:
            # Primitive value
            return pd.DataFrame({'value': [data]})
    
    def _extract_plotly_data(self, data: dict) -> pd.DataFrame:
        """Extract meaningful data from Plotly structure"""
        if data.get('data', [{}])[0].get('type') == 'sankey':
            sankey = data['data'][0]
            # Convert sankey to tabular format
            return pd.DataFrame({
                'source': sankey.get('link', {}).get('source', []),
                'target': sankey.get('link', {}).get('target', []), 
                'value': sankey.get('link', {}).get('value', [])
            })
        # Default: flatten the structure
        return pd.json_normalize(data)
    
    
    def _extract_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract only the patterns that matter for visualization.
        No statistical masturbation.
        """
        return {
            # Numeric vs Categorical - fundamental for chart selection
            "numeric_cols": [col for col in df.columns 
                           if pd.api.types.is_numeric_dtype(df[col])],
            "categorical_cols": [col for col in df.columns 
                               if pd.api.types.is_object_dtype(df[col])],
            
            "has_nulls": df.isnull().any().any(),
            "high_cardinality": [col for col in df.columns
                               if df[col].nunique() > df.shape[0] * 0.9],
        }


class SmartMetaExtractor(MetaExtractor):
    """
    MetaExtractor with intelligent sampling for large datasets.
    """
    
    def extract(self, path: str) -> Dict[str, Any]:
        """Enhanced extraction with smart sampling"""
        meta = super().extract(path)
        df = self._to_dataframe(path)
        
        # Add intelligent samples instead of blind head(3)
        meta["samples"] = self._smart_sample(df)
        
        return meta
    
    def _smart_sample(self, df: pd.DataFrame, n_samples: int = 3) -> Dict[str, Any]:
        """
        Smart sampling: show representative data, not just first rows.
        """
        samples = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric: show min, median, max
                samples[col] = {
                    "min": df[col].min(),
                    "median": df[col].median(),
                    "max": df[col].max()
                }
            else:
                # For categorical: show most frequent values
                value_counts = df[col].value_counts()
                samples[col] = value_counts.head(3).to_dict()
        
        return samples


def compare_extractors():
    """
    Demonstration: Less code, more capability
    """
    # Old way: 410 lines, 2 classes, 7 special methods
    # New way: 150 lines, 1 class, 0 special cases

    # Example path - replace with your own data file
    test_file = "benchmark_data/data/76/data.csv"

    # Old approach (commented out - requires legacy code)
    # from local_info_preextractor import LocalInfoPreExtractor
    old_extractor = LocalInfoPreExtractor()
    old_result = old_extractor.extract_complete_info(test_file)
    
    # New approach  
    new_extractor = SmartMetaExtractor()
    new_result = new_extractor.extract(test_file)
    
    print("Old result keys:", len(old_result.keys()))  # ~12 keys
    print("New result keys:", len(new_result.keys()))  # 4 keys
    
    # But new result has everything that matters
    return new_result


if __name__ == "__main__":
    # Example usage - replace these paths with your own data files
    extractor = SmartMetaExtractor()

    # Works for ANY file type with SAME interface
    # Note: These are example paths - update to match your data location
    for test_file in [
        "benchmark_data/data/1/data.csv",
        "benchmark_data/data/82/data.json",
        "benchmark_data/data/98/Consumption.csv"
    ]:
        if Path(test_file).exists():
            meta = extractor.extract(test_file)
            print(f"\n{test_file}:")
            print(f"  Shape: {meta['shape']}")
            print(f"  Columns: {meta['columns'][:5]}...")