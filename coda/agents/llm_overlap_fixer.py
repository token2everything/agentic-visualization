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
LLM-based Overlap Detection and Fixing

Simple, effective overlap prevention using LLM reasoning instead of complex rules.
"""

def get_overlap_fix_prompt(code: str) -> str:
    """Generate prompt for LLM to fix overlaps"""

    return f"""You are a matplotlib visualization expert.

CODE TO ANALYZE:
```python
{code}
```

TASK: Fix any potential visual overlap issues in this matplotlib code.

USE YOUR EXPERTISE TO:
1. Identify elements that might overlap (labels, legends, titles, annotations)
2. Apply matplotlib best practices to prevent overlaps
3. Ensure all text and visual elements are clearly readable
4. Maintain the original visualization's intent and data

Return ONLY the fixed Python code, no explanations.
"""

def get_smart_overlap_analysis_prompt(code: str, error_msg: str = "") -> str:
    """Generate prompt for intelligent overlap analysis"""

    context = f"ERROR/ISSUE: {error_msg}" if error_msg else "Check for potential visual overlap issues."

    return f"""As a matplotlib expert, analyze this visualization code.

CODE:
```python
{code}
```

{context}

Analyze and determine if there are visual overlap risks.

Provide your analysis in JSON format:
{{
    "has_overlap_risk": true/false,
    "overlap_type": "describe the type of overlap if any",
    "severity": "high|medium|low",
    "affected_elements": ["list elements that may overlap"],
    "recommended_fixes": ["your expert recommendations"],
    "confidence": 0.0-1.0
}}
"""