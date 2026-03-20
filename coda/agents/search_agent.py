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

import logging
import requests
from typing import Dict, List, Optional, Any
import re
from urllib.parse import urljoin
import time
from vertexai import generative_models
from .base import BaseAgent

# Optional Google Search imports
try:
    import google.generativeai as genai
    from google.generativeai import types
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False
    genai = None
    types = None

class SearchAgent(BaseAgent):
    """
    LLM-powered search agent that intelligently finds matplotlib examples based on plot type recommendations
    """
    
    def __init__(self, model_name: str, search_model_name: str, agent_name: str = "matplotlib_search_agent", config=None):
        super().__init__(agent_name, config)
        self.model_name = model_name
        self.search_model_name = search_model_name
        self.persona = "Dr. Michael Zhang - MIT PhD Computer Science, Expert in Data Visualization and Web Search"
        self.specialization = "Intelligent search and code example retrieval for data visualization"
        self.matplotlib_base_url = "https://matplotlib.org/stable/"
        self.matplotlib_gallery_url = "https://matplotlib.org/stable/gallery/index.html"
        
        # Simplified mapping for direct searches
        self.plot_types_mapping = {
            # Basic plots
            "line": "plot_types/basic/plot.html",
            "plot": "plot_types/basic/plot.html",
            "scatter": "plot_types/basic/scatter_plot.html", 
            "bar": "plot_types/basic/bar.html",
            "barh": "plot_types/basic/bar.html",
            "stem": "plot_types/basic/stem.html",
            "step": "plot_types/basic/plot.html",
            "fill_between": "plot_types/basic/fill_between.html",
            "stackplot": "plot_types/basic/stackplot.html",
            "stairs": "plot_types/basic/stairs.html",
            
            # Statistical plots
            "histogram": "plot_types/stats/hist_plot.html",
            "hist": "plot_types/stats/hist_plot.html",
            "boxplot": "plot_types/stats/boxplot_plot.html",
            "violin": "plot_types/stats/violin.html",
            "violinplot": "plot_types/stats/violin.html",
            "errorbar": "plot_types/stats/errorbar_plot.html",
            "pie": "plot_types/stats/pie.html",
            "ecdf": "plot_types/stats/ecdf.html",
            "eventplot": "plot_types/stats/eventplot.html",
            "hist2d": "plot_types/stats/hist2d.html",
            "hexbin": "plot_types/stats/hexbin.html",
            
            # Grid data plots
            "heatmap": "plot_types/array/imshow.html",
            "imshow": "plot_types/array/imshow.html",
            "pcolormesh": "plot_types/array/pcolormesh.html",
            "contour": "plot_types/array/contour.html",
            "contourf": "plot_types/array/contourf.html",
            "quiver": "plot_types/array/quiver.html",
            "streamplot": "plot_types/array/streamplot.html",
            "barbs": "plot_types/array/barbs.html",
            
            # Unstructured data plots
            "tricontour": "plot_types/unstructured/tricontour.html",
            "tricontourf": "plot_types/unstructured/tricontourf.html",
            "tripcolor": "plot_types/unstructured/tripcolor.html",
            "triplot": "plot_types/unstructured/triplot.html",
            
            # 3D plots
            "3d_scatter": "plot_types/3D/scatter3d.html",
            "3d_plot": "plot_types/3D/plot3d.html",
            "3d_bar": "plot_types/3D/bar3d.html",
            "surface": "plot_types/3D/surface3d.html",
            "wireframe": "plot_types/3D/wireframe3d.html",
            "trisurf": "plot_types/3D/trisurf3d.html"
        }
        
    def search_matplotlib_examples(self, plot_types: List[str]) -> Dict[str, Any]:
        """
        Use LLM to intelligently search for matplotlib examples based on plot types
        
        Args:
            plot_types: List of plot type names
            
        Returns:
            Dictionary containing examples for each plot type
        """
        start_time = time.time()
        results = {}
        
        for plot_type in plot_types:
            try:
                # Use LLM to generate intelligent matplotlib example
                example_result = self._generate_intelligent_example(plot_type)
                if example_result:
                    results[plot_type] = example_result
                    self.logger.info(f"Generated intelligent example for {plot_type}: {len(example_result['code'])} characters")
                else:
                    self.logger.warning(f"Could not generate example for {plot_type}")
            except Exception as e:
                self.logger.error(f"Error generating example for {plot_type}: {e}")
                    
        execution_time = time.time() - start_time
        self.logger.info(f"Search completed in {execution_time:.2f}s, found {len(results)} examples")
        
        return results
    
    def _generate_intelligent_example(self, plot_type: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to generate intelligent matplotlib example for a given plot type
        """
        try:
            search_prompt = f"""
As Dr. Michael Zhang, an expert in data visualization and matplotlib, I need you to generate a high-quality matplotlib example for the plot type: "{plot_type}".

IMPORTANT CONSTRAINT: You must base your code PRIMARILY on matplotlib examples found at https://matplotlib.org/stable/gallery/index.html and its subpages. Additionally, you can reference https://matplotlib.org/stable/plot_types/index.html for plot type categorization. Do NOT invent or create variations - use the official matplotlib documentation patterns.

Your task:
1. Understand what type of visualization "{plot_type}" refers to according to matplotlib's official plot types
2. Generate a complete, executable matplotlib code example following official matplotlib patterns
3. Use the exact style and approach shown in matplotlib's official documentation
4. Include proper imports, sample data, styling, and annotations as shown in official examples
5. Follow matplotlib's official best practices and naming conventions

Requirements for the matplotlib code:
- Use ONLY matplotlib.pyplot (import matplotlib.pyplot as plt) 
- Follow the exact patterns from https://matplotlib.org/stable/gallery/ documentation examples
- Include numpy for data generation if needed (as shown in official examples)
- Create realistic sample data appropriate for the plot type (following official examples)
- Add proper labels, title, and styling (matching official documentation style)
- Include plt.show() at the end
- Make the code self-contained and executable
- Add informative comments that match matplotlib documentation style

Respond with ONLY the Python code in this format:
```python
# [Brief description matching matplotlib docs style]
import matplotlib.pyplot as plt
import numpy as np

# Your complete example code here following official matplotlib patterns
# Include comments matching matplotlib documentation style

plt.show()
```

Plot type to implement: {plot_type}
Primary reference source: https://matplotlib.org/stable/gallery/index.html
Secondary reference: https://matplotlib.org/stable/plot_types/index.html
"""

            generation_config = generative_models.GenerationConfig(
                max_output_tokens=12000
            )
            response = self._generate_with_usage(
                model=self.model_name,
                content=search_prompt,
                generation_config=generation_config
            )
            
            if response and response.text:
                # Extract code from the response
                code_blocks = re.findall(r'```python\n(.*?)\n```', response.text, re.DOTALL)
                if code_blocks:
                    generated_code = code_blocks[0].strip()
                    
                    # Validate the code follows matplotlib official patterns
                    if self._validate_matplotlib_compliance(generated_code, plot_type):
                        return {
                            "url": f"llm_generated_example_{plot_type}",
                            "code": generated_code,
                            "plot_type": plot_type,
                            "source": "llm_generated_matplotlib_official",
                            "description": f"LLM-generated matplotlib example for {plot_type} following official documentation"
                        }
                    else:
                        self.logger.warning(f"Generated code for {plot_type} does not follow matplotlib official patterns")
                else:
                    self.logger.warning(f"No code blocks found in LLM response for {plot_type}")
            else:
                self.logger.warning(f"No response from LLM for {plot_type}")
                
        except Exception as e:
            self.logger.error(f"Error generating intelligent example for {plot_type}: {e}")
        
        # Fallback to template if LLM fails
        return self._get_fallback_example(plot_type)
    
    def _get_fallback_example(self, plot_type: str) -> Dict[str, Any]:
        """
        Fallback method to provide basic examples when LLM generation fails
        """
        template_code = self._get_plot_template_for_type(plot_type)
        
        return {
            "url": f"fallback_template_{plot_type}",
            "code": template_code,
            "plot_type": plot_type,
            "source": "fallback_template",
            "description": f"Fallback template for {plot_type}"
        }
    
    def _get_plot_template_for_type(self, plot_type: str) -> str:
        """
        Get a basic template based on plot type
        """
        plot_type_lower = plot_type.lower()
        
        if any(keyword in plot_type_lower for keyword in ['bar', 'column']):
            return """# Bar chart example following matplotlib official documentation
import matplotlib.pyplot as plt
import numpy as np

# Sample data following matplotlib patterns
x = np.arange(4)
heights = [1, 4, 2, 3]

# Create bar plot following official matplotlib style
fig, ax = plt.subplots()
ax.bar(x, heights, width=1, edgecolor="white", linewidth=0.7)

# Set labels and title following matplotlib conventions
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('bar(x, height)')

plt.show()"""
        
        elif any(keyword in plot_type_lower for keyword in ['scatter', 'point']):
            return """# Scatter plot example following matplotlib official documentation
import matplotlib.pyplot as plt
import numpy as np

# Sample data following matplotlib patterns
N = 50
x = np.random.randn(N)
y = np.random.randn(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# Create scatter plot following official matplotlib style
fig, ax = plt.subplots()
ax.scatter(x, y, c=colors, s=area, alpha=0.5)

# Set labels and title following matplotlib conventions
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('scatter(x, y)')

plt.show()"""
        
        elif any(keyword in plot_type_lower for keyword in ['hist', 'distribution']):
            return """# Histogram example following matplotlib official documentation
import matplotlib.pyplot as plt
import numpy as np

# Sample data following matplotlib patterns
np.random.seed(19680801)
x = np.random.randn(1000)

# Create histogram following official matplotlib style
fig, ax = plt.subplots()
ax.hist(x, bins=30, density=True, alpha=0.7, edgecolor='black')

# Set labels and title following matplotlib conventions
ax.set_xlabel('X')
ax.set_ylabel('Probability density')
ax.set_title('hist(x)')

plt.show()"""
        
        elif any(keyword in plot_type_lower for keyword in ['line', 'time', 'series']):
            return """# Line plot example following matplotlib official documentation
import matplotlib.pyplot as plt
import numpy as np

# Sample data following matplotlib patterns
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create line plot following official matplotlib style
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)

# Set labels and title following matplotlib conventions
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('plot(x, y)')

plt.show()"""
        
        else:
            return """# Generic plot example following matplotlib official documentation
import matplotlib.pyplot as plt
import numpy as np

# Sample data following matplotlib patterns
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create plot following official matplotlib style
fig, ax = plt.subplots()
ax.plot(x, y)

# Set labels and title following matplotlib conventions
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Basic plot')

plt.show()"""
    
    def _fetch_matplotlib_example(self, relative_url: str) -> Optional[str]:
        """
        Fetch example code from matplotlib documentation or return template
        """
        try:
            full_url = urljoin(self.matplotlib_base_url, relative_url)
            response = requests.get(full_url, timeout=10)
            response.raise_for_status()
            
            # Extract code blocks from the HTML
            content = response.text
            
            # Look for code blocks in various formats
            code_patterns = [
                r'<div class="highlight-python[^"]*"><div class="highlight"><pre><span></span>(.*?)</pre></div></div>',
                r'<pre class="literal-block">(.*?)</pre>',
                r'<code class="[^"]*python[^"]*">(.*?)</code>',
                r'```python\n(.*?)\n```'
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    # Take the first substantial code block
                    for match in matches:
                        cleaned_code = self._clean_html_code(match)
                        if len(cleaned_code) > 50:  # Filter out very short snippets
                            return cleaned_code
            
            # If no code found but URL is valid, return a template
            return self._get_plot_template(relative_url)
                            
        except Exception as e:
            self.logger.warning(f"Error fetching from {relative_url}: {e}")
            # Return template as fallback
            return self._get_plot_template(relative_url)
    
    def _get_plot_template(self, relative_url: str) -> str:
        """
        Return a basic template for plot types when direct code extraction fails
        """
        if "bar" in relative_url:
            return """import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.arange(4)
y = [1, 4, 2, 3]

# Create bar plot
plt.bar(x, y, width=0.8, edgecolor='white', linewidth=0.7)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot Example')
plt.show()"""
        
        elif "scatter" in relative_url:
            return """import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.random.randn(50)
y = np.random.randn(50)

# Create scatter plot
plt.scatter(x, y, alpha=0.7)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot Example')
plt.show()"""
        
        elif "hist" in relative_url:
            return """import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.normal(0, 1, 1000)

# Create histogram
plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.show()"""
        
        elif "plot" in relative_url:
            return """import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create line plot
plt.plot(x, y, linewidth=2)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Line Plot Example')
plt.show()"""
        
        else:
            return """import matplotlib.pyplot as plt
import numpy as np

# Sample data and plot
x = np.arange(10)
y = np.random.rand(10)

plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generic Plot')
plt.show()"""
    
    def _clean_html_code(self, html_code: str) -> str:
        """
        Clean HTML entities and tags from code
        """
        import html
        
        # Remove HTML tags
        code = re.sub(r'<[^>]+>', '', html_code)
        
        # Decode HTML entities
        code = html.unescape(code)
        
        # Clean up whitespace
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                
        return '\n'.join(cleaned_lines)
    
    def _search_with_google(self, plot_type: str) -> Optional[Dict[str, Any]]:
        """
        Use Google Search as fallback to find matplotlib examples
        """
        if not GOOGLE_SEARCH_AVAILABLE:
            self.logger.warning("Google Search not available - google.generativeai not installed")
            return None
            
        try:
            # Configure Google Search tool
            search_query = f"matplotlib {plot_type} example code python site:matplotlib.org"
            
            response = self._generate_with_usage(
                content=f"Find a complete matplotlib example for {plot_type} plot. Search: {search_query}",
                model=self.search_model_name,
                generation_config={"tools": [{"google_search": {}}]}
            )
            
            if response and response.text:
                # Extract code from the response
                code_match = re.search(r'```python\n(.*?)\n```', response.text, re.DOTALL)
                if code_match:
                    return {
                        "url": "google_search_result",
                        "code": code_match.group(1),
                        "plot_type": plot_type,
                        "source": "google_search"
                    }
                    
        except Exception as e:
            self.logger.warning(f"Google search failed for {plot_type}: {e}")
            
        return None
    
    def _validate_matplotlib_compliance(self, code: str, plot_type: str) -> bool:
        """
        Validate that generated code follows matplotlib official documentation patterns
        """
        # Check essential matplotlib imports
        if 'import matplotlib.pyplot as plt' not in code:
            self.logger.warning(f"Code for {plot_type} missing matplotlib.pyplot import")
            return False
            
        # Check for matplotlib function calls
        if not any(pattern in code for pattern in ['plt.', 'ax.']):
            self.logger.warning(f"Code for {plot_type} missing matplotlib function calls")
            return False
            
        # Check for basic plot structure
        essential_elements = ['plt.show()', 'fig, ax = plt.subplots()']
        if not any(element in code for element in essential_elements):
            self.logger.warning(f"Code for {plot_type} missing essential matplotlib structure")
            return False
            
        return True
    
    def process_query_output(self, query_analysis_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method - extract plot types from query analysis and search for examples
        
        Args:
            query_analysis_output: Output from query analyzer containing plot type recommendations
            
        Returns:
            Dictionary containing matplotlib examples for recommended plot types
        """
        start_time = time.time()
        
        # Extract plot types from query analysis
        plot_types = []
        
        if "recommended_plot_types" in query_analysis_output:
            plot_types.extend(query_analysis_output["recommended_plot_types"])
        
        if "visualization_type" in query_analysis_output:
            plot_types.append(query_analysis_output["visualization_type"])
            
        if "chart_type" in query_analysis_output:
            plot_types.append(query_analysis_output["chart_type"])
        
        # Remove duplicates and empty values - handle both strings and other data types
        cleaned_plot_types = []
        for pt in plot_types:
            if pt:  # Check if not empty/None
                if isinstance(pt, str):
                    cleaned_pt = pt.strip()
                    if cleaned_pt:  # Only add non-empty strings
                        cleaned_plot_types.append(cleaned_pt)
                elif isinstance(pt, dict):
                    # If it's a dict, try to extract a string representation
                    if 'type' in pt:
                        plot_type_str = str(pt['type']).strip()
                    elif 'name' in pt:
                        plot_type_str = str(pt['name']).strip()
                    else:
                        plot_type_str = str(pt).strip()
                    if plot_type_str:
                        cleaned_plot_types.append(plot_type_str)
                else:
                    # Convert to string for other types
                    plot_type_str = str(pt).strip()
                    if plot_type_str:
                        cleaned_plot_types.append(plot_type_str)
        
        plot_types = list(set(cleaned_plot_types))
        
        if not plot_types:
            self.logger.warning("No plot types found in query analysis output")
            return {"examples": {}, "search_summary": "No plot types to search for"}
        
        self.logger.info(f"Searching for examples for plot types: {plot_types}")
        
        # Search for examples
        examples = self.search_matplotlib_examples(plot_types)
        
        execution_time = time.time() - start_time
        
        result = {
            "examples": examples,
            "searched_plot_types": plot_types,
            "search_summary": f"Found {len(examples)} examples out of {len(plot_types)} requested plot types",
            "execution_time": execution_time
        }
        
        self.logger.info(f"Search agent completed in {execution_time:.2f}s")
        return result

if __name__ == "__main__":
    # Test the search agent
    agent = SearchAgent()
    
    # Test with sample query analysis output
    test_output = {
        "recommended_plot_types": ["bar", "scatter", "line"],
        "visualization_type": "histogram"
    }
    
    result = agent.process_query_output(test_output)
    print("Search Results:")
    for plot_type, example in result["examples"].items():
        print(f"\n{plot_type}:")
        print(f"  URL: {example['url']}")
        print(f"  Code length: {len(example['code'])} characters")
        print(f"  Code preview: {example['code'][:200]}...")
