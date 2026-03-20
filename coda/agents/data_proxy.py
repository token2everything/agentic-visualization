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
Smart DataProxy for providing DataFrame interface without full data transmission
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union, Tuple


class DataProxy:
    """
    Intelligent DataFrame proxy that provides all necessary DataFrame methods
    while only storing compressed metadata and samples.
    
    从第一性原理思考：agents需要什么？
    1. 结构信息：shape, columns, dtypes
    2. 数据样本：head(), tail(), sample()
    3. 统计信息：describe(), value_counts()
    4. 语义理解：column meanings, relationships
    """
    
    def __init__(self, df: pd.DataFrame, max_sample_rows: int = 10):
        """Initialize with intelligent data compression"""
        self._original_shape = df.shape
        self._columns = list(df.columns)
        self._dtypes = df.dtypes.to_dict()
        self._index_name = df.index.name
        
        # Store intelligent samples
        self._head_sample = df.head(min(5, len(df))).copy()
        self._tail_sample = df.tail(min(5, len(df))).copy() if len(df) > 5 else self._head_sample.copy()
        
        # Random sampling for diversity
        if len(df) > max_sample_rows:
            self._random_sample = df.sample(min(max_sample_rows, len(df))).copy()
        else:
            self._random_sample = df.copy()
            
        # Statistical summaries
        self._numeric_stats = {}
        self._categorical_stats = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self._numeric_stats[col] = {
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'median': float(df[col].median()) if not df[col].isna().all() else None,
                    'null_count': int(df[col].isnull().sum()),
                    'unique_count': int(df[col].nunique())
                }
            else:
                unique_values = df[col].value_counts().head(10)
                self._categorical_stats[col] = {
                    'unique_count': int(df[col].nunique()),
                    'null_count': int(df[col].isnull().sum()),
                    'top_values': unique_values.to_dict(),
                    'most_frequent': unique_values.index[0] if len(unique_values) > 0 else None
                }
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return original shape"""
        return self._original_shape
    
    @property
    def columns(self) -> pd.Index:
        """Return columns as pandas Index"""
        return pd.Index(self._columns)
    
    @property
    def dtypes(self) -> pd.Series:
        """Return dtypes as pandas Series"""
        return pd.Series(self._dtypes)
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Return first n rows"""
        return self._head_sample.head(n)
    
    def tail(self, n: int = 5) -> pd.DataFrame:
        """Return last n rows"""
        return self._tail_sample.tail(n)
    
    def sample(self, n: int = 5) -> pd.DataFrame:
        """Return random sample"""
        return self._random_sample.sample(min(n, len(self._random_sample)))
    
    def describe(self) -> pd.DataFrame:
        """Return statistical description"""
        desc_data = {}
        
        for col in self._columns:
            if col in self._numeric_stats:
                stats = self._numeric_stats[col]
                desc_data[col] = {
                    'count': self._original_shape[0] - stats['null_count'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    '25%': stats['min'],  # Approximate
                    '50%': stats['median'],
                    '75%': stats['max'],  # Approximate
                    'max': stats['max']
                }
            else:
                stats = self._categorical_stats[col]
                desc_data[col] = {
                    'count': self._original_shape[0] - stats['null_count'],
                    'unique': stats['unique_count'],
                    'top': stats['most_frequent'],
                    'freq': list(stats['top_values'].values())[0] if stats['top_values'] else 0
                }
        
        return pd.DataFrame(desc_data).T
    
    def nunique(self) -> pd.Series:
        """Return unique counts"""
        unique_counts = {}
        for col in self._columns:
            if col in self._numeric_stats:
                unique_counts[col] = self._numeric_stats[col]['unique_count']
            else:
                unique_counts[col] = self._categorical_stats[col]['unique_count']
        return pd.Series(unique_counts)
    
    def isnull(self):
        """Return a proxy for null checking"""
        class NullProxy:
            def __init__(self, numeric_stats, categorical_stats, shape):
                self.numeric_stats = numeric_stats
                self.categorical_stats = categorical_stats
                self.shape = shape
                
            def sum(self):
                null_counts = {}
                for col, stats in self.numeric_stats.items():
                    null_counts[col] = stats['null_count']
                for col, stats in self.categorical_stats.items():
                    null_counts[col] = stats['null_count']
                return pd.Series(null_counts)
                
        return NullProxy(self._numeric_stats, self._categorical_stats, self._original_shape)
    
    def select_dtypes(self, include=None, exclude=None):
        """Select columns by data type"""
        selected_cols = []
        
        for col, dtype in self._dtypes.items():
            dtype_str = str(dtype)
            
            include_match = True
            if include is not None:
                include_match = any(inc in dtype_str for inc in (include if isinstance(include, list) else [include]))
                
            exclude_match = False  
            if exclude is not None:
                exclude_match = any(exc in dtype_str for exc in (exclude if isinstance(exclude, list) else [exclude]))
                
            if include_match and not exclude_match:
                selected_cols.append(col)
        
        # Return a mini DataFrame with just these columns
        mini_df = self._head_sample[selected_cols] if selected_cols else pd.DataFrame()
        return DataProxy(mini_df) if not mini_df.empty else mini_df
    
    def to_dict(self, orient: str = 'dict') -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self._random_sample.to_dict(orient=orient)
    
    def __getitem__(self, key):
        """Support column selection df[col]"""
        if isinstance(key, str) and key in self._columns:
            # Return a Series proxy
            return self._random_sample[key] if key in self._random_sample.columns else pd.Series(dtype=self._dtypes[key])
        elif isinstance(key, list):
            # Return subset DataProxy
            valid_cols = [c for c in key if c in self._columns]
            if valid_cols:
                subset_df = self._random_sample[valid_cols] if all(c in self._random_sample.columns for c in valid_cols) else pd.DataFrame(columns=valid_cols)
                return DataProxy(subset_df)
        return None
    
    def __len__(self) -> int:
        """Return original length"""
        return self._original_shape[0]
    
    def __repr__(self) -> str:
        """String representation"""
        return f"DataProxy({self._original_shape[0]} rows × {self._original_shape[1]} columns)\nColumns: {self._columns[:5]}{'...' if len(self._columns) > 5 else ''}"
    
    def memory_usage(self, index: bool = True, deep: bool = False) -> pd.Series:
        """Return estimated memory usage"""
        memory_data = {}
        for col in self._columns:
            # Estimate memory usage based on dtype and row count
            if pd.api.types.is_numeric_dtype(self._dtypes[col]):
                memory_data[col] = self._original_shape[0] * 8  # Rough estimate for numeric
            else:
                memory_data[col] = self._original_shape[0] * 50  # Rough estimate for strings
        
        if index:
            memory_data['Index'] = self._original_shape[0] * 8
            
        return pd.Series(memory_data)
    
    def info(self, verbose: bool = None, buf=None, max_cols=None, memory_usage=None, show_counts=None):
        """Print DataFrame info"""
        info_lines = [
            f"<class 'DataProxy'>",
            f"RangeIndex: {self._original_shape[0]} entries, 0 to {self._original_shape[0]-1}",
            f"Data columns (total {self._original_shape[1]} columns):",
        ]
        
        for i, col in enumerate(self._columns):
            non_null = self._original_shape[0]
            if col in self._numeric_stats:
                non_null -= self._numeric_stats[col]['null_count']
            elif col in self._categorical_stats:
                non_null -= self._categorical_stats[col]['null_count']
            
            info_lines.append(f" #{i:<3} {col:<20} {non_null} non-null {str(self._dtypes[col])}")
        
        info_text = "\n".join(info_lines)
        if buf is None:
            print(info_text)
        else:
            buf.write(info_text)

    def get_semantic_summary(self) -> str:
        """Generate comprehensive semantic summary for LLMs"""
        summary_parts = [
            f"DataFrame: {self._original_shape[0]} rows × {self._original_shape[1]} columns",
            f"Columns: {', '.join(self._columns)}",
            "",
            "=== Column Analysis ===",
        ]
        
        for col in self._columns:
            if col in self._numeric_stats:
                stats = self._numeric_stats[col]
                summary_parts.append(f"{col} ({self._dtypes[col]}): range [{stats['min']:.2f}, {stats['max']:.2f}], mean={stats['mean']:.2f}")
            else:
                stats = self._categorical_stats[col]
                top_vals = list(stats['top_values'].keys())[:3]
                summary_parts.append(f"{col} ({self._dtypes[col]}): {stats['unique_count']} unique, top: {top_vals}")
        
        summary_parts.extend([
            "",
            "=== Data Sample ===",
            str(self._head_sample),
        ])
        
        return "\n".join(summary_parts)