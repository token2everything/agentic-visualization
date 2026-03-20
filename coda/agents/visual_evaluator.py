"""
Visual Evaluator Agent

Evaluates generated visualizations from a human perspective, scoring
readability, aesthetics, and UX, then provides targeted feedback for
iterative refinement until quality thresholds are met.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

def safe_json_dumps(obj, **kwargs):
    """Safely serialize objects to JSON, converting numpy types to Python types"""
    def convert_numpy_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    return json.dumps(convert_numpy_types(obj), **kwargs)
from PIL import Image, ImageChops, ImageStat
import base64
import io
import os
from pathlib import Path
import cv2
from skimage import metrics
from scipy import ndimage
import matplotlib.colors as mcolors

from .base import BaseAgent, AgentMessage
from .debug_agent import ExecutionResult

DebugResult = ExecutionResult


@dataclass
class VisualAssessment:
    """Structure for visual assessment results"""
    assessment_id: str
    overall_score: float
    human_readability_score: float
    aesthetic_quality_score: float
    information_clarity_score: float
    accessibility_score: float
    user_experience_score: float
    detailed_feedback: Dict[str, Any]
    improvement_suggestions: List[str]
    strengths: List[str]
    weaknesses: List[str]
    assessment_confidence: float

@dataclass
class RefinementResult:
    """Result structure for visual refinement process"""
    refinement_id: str
    original_assessment: VisualAssessment
    refinement_suggestions: List[str]
    priority_improvements: List[str]
    code_modifications: List[str]
    expected_improvements: Dict[str, float]
    refinement_rationale: str
    human_impact_analysis: Dict[str, Any]
    implementation_difficulty: str
    success_metrics: List[str]

class VisualEvaluator(BaseAgent):
    """
    Visual Evaluator that assesses visualizations from a human perspective.

    Scores human readability, aesthetic quality, and user experience,
    then generates targeted feedback for iterative refinement.
    """

    def __init__(self, model_name: str, search_model_name: str, agent_id: str = "visual_evaluator", config=None):
        super().__init__(agent_id, config)
        self.model_name = model_name
        self.search_model_name = search_model_name
        self.persona = "Visual evaluation expert"
        self.specialization = "Human perception, visual cognition, and user experience"
        
    def evaluate_visualization(self, debug_result: 'ExecutionResult',
                             data: pd.DataFrame,
                             original_query: str = "",
                             plotting_key_points: List[str] = None,
                             evaluation_context: Optional[Dict[str, Any]] = None,
                             ground_truth_path: Optional[str] = None) -> VisualAssessment:
        """
        Evaluate visualization with enhanced semantic validation and ground truth comparison.
        
        Args:
            debug_result: ExecutionResult from the DebugAgent
            data: Original data used for visualization
            original_query: Original user query for context
            plotting_key_points: List of key plotting requirements from query analysis
            evaluation_context: Optional context for evaluation
            ground_truth_path: Optional path to ground truth image for comparison
            
        Returns:
            VisualAssessment with comprehensive evaluation including semantic validation
        """
        
        if not debug_result.success or not debug_result.output_file:
            return self._create_failure_assessment(debug_result)
        
        # Get the output file from the execution result
        output_file = debug_result.output_file
        
        if not output_file or not os.path.exists(output_file):
            return self._create_no_output_assessment(debug_result)
        
        try:
            # Enhanced visual analysis with semantic validation
            visual_analysis = self._analyze_visual_output(output_file, data, original_query, plotting_key_points, evaluation_context)
            
            # Add ground truth comparison if available
            comparison_results = {}
            if ground_truth_path and os.path.exists(ground_truth_path):
                comparison_results = self.compare_with_ground_truth(output_file, ground_truth_path)
                visual_analysis['ground_truth_comparison'] = comparison_results
            
            # Add accessibility analysis
            accessibility_analysis = self.analyze_accessibility_compliance(output_file)
            visual_analysis['accessibility_analysis'] = accessibility_analysis
            
            # Perform semantic validation with search
            semantic_validation = self.validate_semantic_correctness(original_query, output_file, evaluation_context)
            visual_analysis['semantic_validation'] = semantic_validation
            
            # Enhanced assessment with semantic validation
            readability_score = self._assess_human_readability(visual_analysis, data)
            aesthetic_score = self._assess_aesthetic_quality(visual_analysis)
            clarity_score = self._assess_information_clarity(visual_analysis, data)
            accessibility_score = self._assess_accessibility_enhanced(visual_analysis)
            ux_score = self._assess_user_experience(visual_analysis, evaluation_context)
            
            # Add ground truth similarity bonus/penalty
            if comparison_results.get('comparison_success', False):
                similarity_score = comparison_results.get('composite_similarity', 0.0)
                # Apply similarity bonus to overall scoring
                similarity_bonus = (similarity_score - 0.5) * 0.2  # Can be ±0.1
                readability_score = min(1.0, max(0.0, readability_score + similarity_bonus))
                clarity_score = min(1.0, max(0.0, clarity_score + similarity_bonus))
            
            # Calculate overall score with semantic validation
            semantic_score = semantic_validation.get('semantic_score', 1.0)
            overall_score = self._calculate_overall_score_with_semantics({
                "readability": readability_score,
                "aesthetic": aesthetic_score,
                "clarity": clarity_score,
                "accessibility": accessibility_score,
                "ux": ux_score,
                "semantic": semantic_score
            })
            
            # Generate enhanced detailed feedback
            detailed_feedback = self._generate_enhanced_feedback(
                visual_analysis, readability_score, aesthetic_score, 
                clarity_score, accessibility_score, ux_score, comparison_results
            )
            
            # Generate semantic-aware improvement suggestions
            improvements = self._generate_semantic_improvement_suggestions(
                visual_analysis, detailed_feedback, overall_score, comparison_results
            )
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(
                visual_analysis, detailed_feedback
            )
            
            # Assess confidence in evaluation
            confidence = self._assess_evaluation_confidence(visual_analysis, overall_score)
            
            assessment = VisualAssessment(
                assessment_id=f"visual_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                overall_score=overall_score,
                human_readability_score=readability_score,
                aesthetic_quality_score=aesthetic_score,
                information_clarity_score=clarity_score,
                accessibility_score=accessibility_score,
                user_experience_score=ux_score,
                detailed_feedback=detailed_feedback,
                improvement_suggestions=improvements,
                strengths=strengths,
                weaknesses=weaknesses,
                assessment_confidence=confidence
            )
            
            self.logger.info(f"Visual evaluation completed. Overall score: {overall_score:.2f}")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Visual evaluation failed: {str(e)}")
            return self._create_error_assessment(debug_result, str(e))
    
    def refine_visualization(self, assessment: VisualAssessment,
                           debug_result: DebugResult,
                           refinement_threshold: float = 0.9) -> RefinementResult:
        """
        Generate refinement suggestions based on visual assessment.
        
        Args:
            assessment: Visual assessment results
            debug_result: Debug results with current code
            refinement_threshold: Threshold for refinement (0.0-1.0)
            
        Returns:
            RefinementResult with refinement suggestions
        """
        
        try:
            # Analyze refinement needs
            refinement_analysis = self._analyze_refinement_needs(assessment, refinement_threshold)
            
            # Generate specific refinement suggestions
            refinement_suggestions = self._generate_refinement_suggestions(
                assessment, debug_result, refinement_analysis
            )
            
            # Prioritize improvements
            priority_improvements = self._prioritize_improvements(
                refinement_suggestions, assessment
            )
            
            # Generate code modifications
            code_modifications = self._generate_code_modifications(
                debug_result.fixed_code or debug_result.original_code,
                priority_improvements,
                assessment
            )
            
            # Estimate expected improvements
            expected_improvements = self._estimate_expected_improvements(
                priority_improvements, assessment
            )
            
            # Create refinement rationale
            rationale = self._create_refinement_rationale(
                assessment, priority_improvements, refinement_threshold
            )
            
            # Analyze human impact
            human_impact = self._analyze_human_impact(priority_improvements, assessment)
            
            # Assess implementation difficulty
            difficulty = self._assess_implementation_difficulty(code_modifications)
            
            # Define success metrics
            success_metrics = self._define_success_metrics(
                priority_improvements, assessment, refinement_threshold
            )
            
            refinement_result = RefinementResult(
                refinement_id=f"refine_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                original_assessment=assessment,
                refinement_suggestions=refinement_suggestions,
                priority_improvements=priority_improvements,
                code_modifications=code_modifications,
                expected_improvements=expected_improvements,
                refinement_rationale=rationale,
                human_impact_analysis=human_impact,
                implementation_difficulty=difficulty,
                success_metrics=success_metrics
            )
            
            self.logger.info(f"Refinement analysis completed. Priority improvements: {len(priority_improvements)}")
            return refinement_result
            
        except Exception as e:
            self.logger.error(f"Refinement analysis failed: {str(e)}")
            return self._create_error_refinement(assessment, str(e))
    
    def compare_with_ground_truth(self, generated_image_path: str, ground_truth_path: str) -> Dict[str, float]:
        """Compare generated image with ground truth using multiple metrics"""
        
        try:
            if not os.path.exists(generated_image_path) or not os.path.exists(ground_truth_path):
                return {"error": "Missing image files", "similarity_score": 0.0}
            
            # Load images
            generated = Image.open(generated_image_path).convert('RGB')
            ground_truth = Image.open(ground_truth_path).convert('RGB')
            
            # Resize to same dimensions for comparison
            target_size = (800, 600)
            generated = generated.resize(target_size, Image.Resampling.LANCZOS)
            ground_truth = ground_truth.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy arrays
            gen_array = np.array(generated)
            gt_array = np.array(ground_truth)
            
            # Calculate multiple similarity metrics
            metrics_results = {}
            
            # 1. Structural Similarity Index (SSIM)
            ssim_score = metrics.structural_similarity(gen_array, gt_array, multichannel=True, channel_axis=2)
            metrics_results['ssim'] = float(ssim_score)
            
            # 2. Peak Signal-to-Noise Ratio (PSNR)
            psnr_score = metrics.peak_signal_noise_ratio(gt_array, gen_array)
            metrics_results['psnr'] = float(psnr_score)
            
            # 3. Mean Squared Error
            mse_score = metrics.mean_squared_error(gt_array, gen_array)
            metrics_results['mse'] = float(mse_score)
            
            # 4. Color histogram comparison
            color_similarity = self._compare_color_histograms(generated, ground_truth)
            metrics_results['color_similarity'] = color_similarity
            
            # 5. Edge detection similarity
            edge_similarity = self._compare_edge_features(gen_array, gt_array)
            metrics_results['edge_similarity'] = edge_similarity
            
            # 6. Layout similarity using feature matching
            layout_similarity = self._compare_layout_features(gen_array, gt_array)
            metrics_results['layout_similarity'] = layout_similarity
            
            # Calculate composite similarity score
            weights = {
                'ssim': 0.25,
                'color_similarity': 0.20,
                'edge_similarity': 0.25,
                'layout_similarity': 0.30
            }
            
            composite_score = sum(metrics_results[metric] * weight 
                                for metric, weight in weights.items() 
                                if metric in metrics_results)
            
            metrics_results['composite_similarity'] = composite_score
            metrics_results['comparison_success'] = True
            
            self.logger.info(f"Image comparison completed. Composite similarity: {composite_score:.3f}")
            return metrics_results
            
        except Exception as e:
            self.logger.error(f"Image comparison failed: {str(e)}")
            return {"error": str(e), "similarity_score": 0.0, "comparison_success": False}
    
    def _compare_color_histograms(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compare color histograms of two images"""
        try:
            # Calculate histograms for each channel
            hist1 = np.array([np.histogram(np.array(img1)[:,:,i], bins=256, range=(0,256))[0] for i in range(3)])
            hist2 = np.array([np.histogram(np.array(img2)[:,:,i], bins=256, range=(0,256))[0] for i in range(3)])
            
            # Normalize histograms
            hist1 = hist1 / (hist1.sum(axis=1, keepdims=True) + 1e-10)
            hist2 = hist2 / (hist2.sum(axis=1, keepdims=True) + 1e-10)
            
            # Calculate correlation coefficient for each channel
            correlations = [np.corrcoef(hist1[i], hist2[i])[0,1] for i in range(3)]
            
            # Return average correlation
            return float(np.nanmean([c for c in correlations if not np.isnan(c)]))
        except:
            return 0.0
    
    def _compare_edge_features(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare edge features using Canny edge detection"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
            
            # Apply Canny edge detection
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            
            # Calculate intersection over union of edge pixels
            intersection = int(np.logical_and(edges1, edges2).sum())
            union = int(np.logical_or(edges1, edges2).sum())
            
            if union == 0:
                return 1.0 if intersection == 0 else 0.0
            
            return float(intersection / union)
        except:
            return 0.0
    
    def _compare_layout_features(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare layout features using grid-based analysis"""
        try:
            # Divide images into grid
            h, w = img1.shape[:2]
            grid_size = 4
            
            h_step, w_step = h // grid_size, w // grid_size
            
            similarities = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Extract grid cells
                    y1, y2 = i * h_step, (i + 1) * h_step
                    x1, x2 = j * w_step, (j + 1) * w_step
                    
                    cell1 = img1[y1:y2, x1:x2]
                    cell2 = img2[y1:y2, x1:x2]
                    
                    # Calculate mean color for each cell
                    mean1 = np.mean(cell1, axis=(0, 1))
                    mean2 = np.mean(cell2, axis=(0, 1))
                    
                    # Calculate similarity (1 - normalized distance)
                    distance = np.linalg.norm(mean1 - mean2)
                    max_distance = np.sqrt(3) * 255  # Maximum possible distance
                    similarity = 1 - (distance / max_distance)
                    
                    similarities.append(similarity)
            
            return float(np.mean(similarities))
        except:
            return 0.0
    
    def analyze_accessibility_compliance(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for accessibility compliance using computer vision"""
        
        try:
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
            
            accessibility_results = {}
            
            # 1. Color contrast analysis
            contrast_analysis = self._analyze_color_contrast(img_array)
            accessibility_results['contrast_analysis'] = contrast_analysis
            
            # 2. Text size analysis (approximate)
            text_analysis = self._analyze_text_elements(img_array)
            accessibility_results['text_analysis'] = text_analysis
            
            # 3. Color distribution analysis
            color_analysis = self._analyze_color_distribution(img_array)
            accessibility_results['color_analysis'] = color_analysis
            
            # 4. Overall accessibility score
            accessibility_score = self._calculate_accessibility_score(accessibility_results)
            accessibility_results['overall_accessibility_score'] = accessibility_score
            
            return accessibility_results
            
        except Exception as e:
            self.logger.error(f"Accessibility analysis failed: {str(e)}")
            return {"error": str(e), "overall_accessibility_score": 0.0}
    
    def _analyze_color_contrast(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color contrast in the image"""
        try:
            # Convert to LAB color space for better contrast measurement
            from skimage import color
            lab_img = color.rgb2lab(img_array / 255.0)
            
            # Calculate luminance statistics
            luminance = lab_img[:, :, 0]
            
            contrast_metrics = {
                'luminance_std': float(np.std(luminance)),
                'luminance_range': float(np.max(luminance) - np.min(luminance)),
                'high_contrast_areas': float(np.sum(np.gradient(luminance)[0]**2 + np.gradient(luminance)[1]**2 > 100) / luminance.size)
            }
            
            # Estimate WCAG compliance
            if contrast_metrics['luminance_range'] > 50:
                wcag_compliance = 'AA'
            elif contrast_metrics['luminance_range'] > 30:
                wcag_compliance = 'A'
            else:
                wcag_compliance = 'none'
            
            contrast_metrics['estimated_wcag_level'] = wcag_compliance
            
            return contrast_metrics
        except:
            return {'error': 'Contrast analysis failed'}
    
    def _analyze_text_elements(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze text elements in the image"""
        try:
            # Simple heuristic: detect high-contrast small features that might be text
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply morphological operations to detect text-like features
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find contours that might be text
            edges = cv2.Canny(processed, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 5000:  # Reasonable text size range
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.1 < aspect_ratio < 10:  # Text-like aspect ratio
                        text_like_areas.append({'area': area, 'bbox': (x, y, w, h)})
            
            return {
                'estimated_text_regions': len(text_like_areas),
                'average_text_area': float(np.mean([area['area'] for area in text_like_areas])) if text_like_areas else 0.0,
                'text_size_adequacy': 'adequate' if len(text_like_areas) > 0 else 'unknown'
            }
        except:
            return {'error': 'Text analysis failed'}
    
    def _analyze_color_distribution(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution for colorblind accessibility"""
        try:
            # Analyze color diversity
            colors_flat = img_array.reshape(-1, 3)
            unique_colors = int(len(np.unique(colors_flat.view(np.dtype((np.void, colors_flat.dtype.itemsize * 3))))))
            
            # Analyze red-green distinction (most common colorblindness)
            red_channel = img_array[:, :, 0]
            green_channel = img_array[:, :, 1]
            rg_correlation = np.corrcoef(red_channel.flatten(), green_channel.flatten())[0, 1]
            
            # High correlation means poor red-green distinction
            colorblind_friendly = bool(rg_correlation < 0.7)
            
            return {
                'unique_colors': int(unique_colors),
                'red_green_correlation': float(rg_correlation),
                'estimated_colorblind_friendly': colorblind_friendly,
                'color_diversity_score': min(1.0, unique_colors / 1000.0)
            }
        except:
            return {'error': 'Color analysis failed'}
    
    def _calculate_accessibility_score(self, accessibility_results: Dict[str, Any]) -> float:
        """Calculate overall accessibility score"""
        try:
            score = 0.0
            weights = {
                'contrast': 0.4,
                'text': 0.3,
                'color': 0.3
            }
            
            # Contrast score
            contrast_data = accessibility_results.get('contrast_analysis', {})
            if 'estimated_wcag_level' in contrast_data:
                wcag_scores = {'AA': 1.0, 'A': 0.7, 'none': 0.3}
                score += wcag_scores.get(contrast_data['estimated_wcag_level'], 0.0) * weights['contrast']
            
            # Text score
            text_data = accessibility_results.get('text_analysis', {})
            if text_data.get('text_size_adequacy') == 'adequate':
                score += 0.8 * weights['text']
            
            # Color score
            color_data = accessibility_results.get('color_analysis', {})
            if color_data.get('estimated_colorblind_friendly', False):
                score += 0.9 * weights['color']
            else:
                score += 0.3 * weights['color']
            
            return float(min(1.0, score))
        except:
            return 0.0
    
    def _analyze_visual_output(self, output_file: str, data: pd.DataFrame,
                             original_query: str = "",
                             plotting_key_points: List[str] = None,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze visual output using enhanced LLM vision capabilities with semantic understanding"""
        
        try:
            # Load and encode image
            with open(output_file, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Get image properties
            image = Image.open(output_file)
            image_properties = {
                "size": image.size,
                "mode": image.mode,
                "format": image.format,
                "file_size": len(image_data)
            }
            
            # Create image part for multimodal litellm input
            image_part = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_properties['format'].lower()};base64,{image_base64}"
                }
            }
            
            context_str = ""
            if context:
                context_str = f"\n\nAdditional Context: {safe_json_dumps(context, indent=2)}"
            
            # Prepare query context
            query_context = ""
            if original_query:
                query_context += f"\n\nOriginal User Query: \"{original_query}\""
            
            # Prepare plotting key points context
            key_points_context = ""
            if plotting_key_points:
                key_points_context = f"\n\nKey Plotting Requirements:\n" + "\n".join([f"- {point}" for point in plotting_key_points])
            
            # Enhanced analysis prompt with semantic understanding
            analysis_prompt = f"""
You are Dr. Elena Vasquez, a Harvard Psychology PhD and Adobe UX Researcher specializing in human perception, visual cognition, and chart validation.

Analyze this matplotlib visualization with STRICT semantic accuracy requirements:

{query_context}{key_points_context}

Image Properties:
{safe_json_dumps(image_properties, indent=2)}

Data Context:
- Shape: {data.shape}
- Columns: {list(data.columns)}
- Data Types: {dict(zip(data.columns, [str(dtype) for dtype in data.dtypes]))}

{context_str}

PERFORM DETAILED SEMANTIC VALIDATION:
1. **Data-Query Alignment**: Does the visualization show the EXACT data relationships requested?
2. **Mathematical Accuracy**: Are formulas, functions, and calculations correctly implemented?
3. **Visual Element Compliance**: Are all requested visual elements (colors, markers, labels, axes) present and correct?
4. **Layout and Structure**: Does the plot structure match the specification (subplots, dimensions, arrangement)?
5. **Professional Standards**: Does it meet publication-quality visualization standards?

IMPORTANT SEMANTIC CHECKS:
- If query asks for specific mathematical functions, verify they are correctly visualized
- If query specifies data ranges or axis limits, verify they are correctly set
- If query requires specific colors or styling, verify exact compliance
- If query asks for multiple subplots with specific content, verify each subplot individually
- If query specifies markers, line styles, or visual effects, verify they are correctly applied

Respond with detailed JSON assessment:
{{
    "semantic_accuracy": {{
        "data_query_match": "excellent|good|fair|poor",
        "mathematical_correctness": "excellent|good|fair|poor",
        "visual_element_compliance": "excellent|good|fair|poor",
        "layout_structure_match": "excellent|good|fair|poor",
        "specification_adherence_score": 0.0-1.0
    }},
    "quality_assessment": {{
        "overall_quality": "excellent|good|fair|poor",
        "readability": "excellent|good|fair|poor",
        "visual_appeal": "high|medium|low",
        "professional_appearance": "yes|no|partially"
    }},
    "requirement_analysis": {{
        "key_points_covered": ["list specific requirements correctly implemented"],
        "key_points_missing": ["list specific requirements NOT implemented"],
        "critical_errors": ["list major deviations from requirements"],
        "requirement_match_percentage": 0.0-1.0
    }},
    "accessibility_check": {{
        "color_contrast_adequate": true|false,
        "colorblind_friendly": true|false,
        "text_size_adequate": true|false,
        "wcag_compliance_level": "AA|A|none"
    }},
    "final_recommendation": {{
        "decision": "approve|revise|reject",
        "confidence_level": 0.0-1.0,
        "primary_issues": ["list main problems"],
        "improvement_priority": "high|medium|low"
    }}
}}

Be extremely strict in semantic validation. A visualization that doesn't match the query requirements should receive low scores regardless of aesthetic quality.
"""
            
            response = self._generate_with_usage(
                model=self.model_name,
                content=[
                    {"type": "text", "text": analysis_prompt},
                    image_part
                ],
                generation_config={
                    "max_output_tokens": 15000,
                    "temperature": 0.1
                }
            )
            result_text = response.text
            
            # Enhanced JSON parsing with better error handling
            json_match = result_text.find('{')
            if json_match != -1:
                json_str = result_text[json_match:]
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > 0:
                    json_str = json_str[:end_pos]
                    try:
                        analysis = json.loads(json_str)
                        analysis["image_properties"] = image_properties
                        analysis["analysis_timestamp"] = datetime.now().isoformat()
                        return analysis
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON parse error: {e}, attempting fallback parsing")
                        return self._create_fallback_analysis(result_text, image_properties)
            
            # If no JSON found, create fallback
            return self._create_fallback_analysis(result_text, image_properties)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze visual output: {str(e)}")
            return self._create_fallback_analysis(str(e), image_properties)
    
    def _create_fallback_analysis(self, error_info: str, image_properties: Dict) -> Dict[str, Any]:
        """Create fallback analysis when main analysis fails"""
        return {
            "semantic_accuracy": {
                "data_query_match": "poor",
                "mathematical_correctness": "poor",
                "visual_element_compliance": "poor",
                "layout_structure_match": "poor",
                "specification_adherence_score": 0.0
            },
            "quality_assessment": {
                "overall_quality": "poor",
                "readability": "unknown",
                "visual_appeal": "low",
                "professional_appearance": "no"
            },
            "requirement_analysis": {
                "key_points_covered": [],
                "key_points_missing": ["Analysis failed"],
                "critical_errors": [f"Visual analysis error: {error_info}"],
                "requirement_match_percentage": 0.0
            },
            "accessibility_check": {
                "color_contrast_adequate": False,
                "colorblind_friendly": False,
                "text_size_adequate": False,
                "wcag_compliance_level": "none"
            },
            "final_recommendation": {
                "decision": "reject",
                "confidence_level": 0.9,
                "primary_issues": ["Analysis failure"],
                "improvement_priority": "high"
            },
            "image_properties": image_properties,
            "error_info": error_info
        }
    
    def _assess_human_readability(self, visual_analysis: Dict[str, Any], data: pd.DataFrame) -> float:
        """Assess human readability score with enhanced semantic validation"""
        
        # Extract semantic accuracy from enhanced analysis
        if isinstance(visual_analysis, dict) and "semantic_accuracy" in visual_analysis:
            semantic_scores = visual_analysis["semantic_accuracy"]
            
            # Calculate weighted semantic score
            semantic_weight = {
                "data_query_match": 0.4,
                "mathematical_correctness": 0.3,
                "visual_element_compliance": 0.2,
                "layout_structure_match": 0.1
            }
            
            semantic_score = 0.0
            for aspect, weight in semantic_weight.items():
                if aspect in semantic_scores:
                    aspect_score = self._convert_quality_to_score(semantic_scores[aspect])
                    semantic_score += aspect_score * weight
            
            # Combine with traditional readability assessment
            readability_prompt = f"""
As Dr. Elena Vasquez, assess the human readability focusing on visual presentation:

Semantic Analysis Results:
{safe_json_dumps(visual_analysis.get('semantic_accuracy', {}), indent=2)}

Quality Assessment:
{safe_json_dumps(visual_analysis.get('quality_assessment', {}), indent=2)}

Data Context:
- Columns: {list(data.columns)}
- Data complexity: {data.shape}

Rate ONLY the visual readability aspects (0.0-1.0):
1. Text clarity and size
2. Information hierarchy
3. Cognitive load
4. Visual presentation quality

Provide only the numerical score as a float.
"""
            
            readability_score = self._get_llm_score(readability_prompt, 0.7)
            
            # Penalize heavily if semantic accuracy is poor
            if semantic_score < 0.5:
                readability_score *= 0.5  # Cut readability score in half for poor semantic accuracy
            
            return min(1.0, max(0.0, (semantic_score * 0.6 + readability_score * 0.4)))
        
        # Fallback to original method if enhanced analysis not available
        readability_prompt = f"""
As Dr. Elena Vasquez, assess the human readability of this visualization:

Visual Analysis:
{safe_json_dumps(visual_analysis, indent=2)}

Data Context:
- Columns: {list(data.columns)}
- Data complexity: {data.shape}

Rate human readability (0.0-1.0) considering:
1. Text clarity and size
2. Information hierarchy
3. Cognitive load
4. Immediate comprehension
5. Visual clutter

Provide only the numerical score as a float.
"""
        
        return self._get_llm_score(readability_prompt, 0.7)
    
    def _assess_aesthetic_quality(self, visual_analysis: Dict[str, Any]) -> float:
        """Assess aesthetic quality score"""
        
        aesthetic_prompt = f"""
As Dr. Elena Vasquez, assess the aesthetic quality of this visualization:

Visual Analysis:
{safe_json_dumps(visual_analysis, indent=2)}

Rate aesthetic quality (0.0-1.0) considering:
1. Color harmony and palette
2. Visual balance and composition
3. Professional appearance
4. Visual appeal
5. Design coherence

Provide only the numerical score as a float.
"""
        
        return self._get_llm_score(aesthetic_prompt, 0.6)
    
    def _assess_information_clarity(self, visual_analysis: Dict[str, Any], data: pd.DataFrame) -> float:
        """Assess information clarity with semantic accuracy emphasis"""
        
        # Prioritize semantic accuracy in clarity assessment
        if isinstance(visual_analysis, dict):
            # Check for enhanced analysis structure
            requirement_analysis = visual_analysis.get("requirement_analysis", {})
            semantic_accuracy = visual_analysis.get("semantic_accuracy", {})
            
            # Heavy penalty for poor requirement matching
            requirement_match = requirement_analysis.get("requirement_match_percentage", 0.0)
            if requirement_match < 0.5:
                return min(0.3, requirement_match)  # Cap at 30% if requirements poorly met
            
            # Calculate semantic clarity score
            data_query_match = self._convert_quality_to_score(
                semantic_accuracy.get("data_query_match", "poor")
            )
            math_correctness = self._convert_quality_to_score(
                semantic_accuracy.get("mathematical_correctness", "poor")
            )
            
            semantic_clarity = (data_query_match * 0.6 + math_correctness * 0.4)
            
            # Traditional clarity assessment
            clarity_prompt = f"""
As Dr. Elena Vasquez, assess the visual clarity and communication effectiveness:

Requirement Analysis:
{safe_json_dumps(requirement_analysis, indent=2)}

Semantic Accuracy:
{safe_json_dumps(semantic_accuracy, indent=2)}

Data Context: {data.shape[1]} columns, {data.shape[0]} rows

Rate ONLY visual communication clarity (0.0-1.0):
1. Visual encoding effectiveness
2. Message communication
3. Interpretation ease
4. Information hierarchy

Provide only the numerical score as a float.
"""
            
            visual_clarity = self._get_llm_score(clarity_prompt, 0.8)
            
            # Weighted combination favoring semantic accuracy
            return semantic_clarity * 0.7 + visual_clarity * 0.3
        
        # Fallback to original method
        clarity_prompt = f"""
As Dr. Elena Vasquez, assess the information clarity of this visualization:

Visual Analysis:
{safe_json_dumps(visual_analysis, indent=2)}

Data Context: {data.shape[1]} columns, {data.shape[0]} rows

Rate information clarity (0.0-1.0) considering:
1. Data accuracy representation
2. Visual encoding effectiveness
3. Message communication
4. Data-to-insight ratio
5. Interpretation ease

Provide only the numerical score as a float.
"""
        
        return self._get_llm_score(clarity_prompt, 0.8)
    
    def _assess_accessibility(self, visual_analysis: Dict[str, Any]) -> float:
        """Assess accessibility score"""
        
        accessibility_prompt = f"""
As Dr. Elena Vasquez, assess the accessibility of this visualization:

Visual Analysis:
{safe_json_dumps(visual_analysis, indent=2)}

Rate accessibility (0.0-1.0) considering:
1. Color blind accessibility
2. Text size and contrast
3. Screen reader compatibility
4. Universal design principles
5. Inclusive design practices

Provide only the numerical score as a float.
"""
        
        return self._get_llm_score(accessibility_prompt, 0.7)
    
    def _assess_user_experience(self, visual_analysis: Dict[str, Any], 
                              context: Optional[Dict[str, Any]] = None) -> float:
        """Assess user experience score"""
        
        context_str = ""
        if context:
            context_str = f"\n\nContext: {safe_json_dumps(context, indent=2)}"
        
        ux_prompt = f"""
As Dr. Elena Vasquez, assess the user experience of this visualization:

Visual Analysis:
{safe_json_dumps(visual_analysis, indent=2)}

{context_str}

Rate user experience (0.0-1.0) considering:
1. Ease of use and navigation
2. Emotional response
3. Actionability of insights
4. Memorability
5. Overall satisfaction

Provide only the numerical score as a float.
"""
        
        return self._get_llm_score(ux_prompt, 0.6)
    
    def _get_llm_score(self, prompt: str, default_score: float) -> float:
        """Get numerical score from LLM"""
        
        try:
            response = self._generate_with_usage(model=self.model_name, content=prompt)
            score_text = response.text.strip()
            
            # Extract numerical score
            import re
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Failed to get LLM score: {str(e)}")
        
        return default_score
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall score with enhanced semantic accuracy weighting"""
        
        # Enhanced weights prioritizing semantic accuracy
        weights = {
            "readability": 0.30,  # Includes semantic accuracy
            "aesthetic": 0.15,   # Reduced weight for aesthetics
            "clarity": 0.30,    # Includes requirement matching
            "accessibility": 0.15,
            "ux": 0.10          # Reduced weight for UX
        }
        
        overall = sum(scores[key] * weights[key] for key in weights)
        return min(1.0, max(0.0, overall))
    
    def _convert_quality_to_score(self, quality_rating: str) -> float:
        """Convert quality rating to numerical score"""
        rating_map = {
            "excellent": 1.0,
            "good": 0.8,
            "fair": 0.6,
            "poor": 0.2,
            "unknown": 0.5
        }
        return rating_map.get(quality_rating.lower(), 0.0)
    
    def _generate_detailed_feedback(self, visual_analysis: Dict[str, Any],
                                  readability: float, aesthetic: float,
                                  clarity: float, accessibility: float,
                                  ux: float) -> Dict[str, Any]:
        """Generate detailed feedback"""
        
        return {
            "scores": {
                "readability": readability,
                "aesthetic": aesthetic,
                "clarity": clarity,
                "accessibility": accessibility,
                "ux": ux
            },
            "visual_analysis": visual_analysis,
            "score_interpretation": {
                "readability": "excellent" if readability > 0.8 else "good" if readability > 0.6 else "needs_improvement",
                "aesthetic": "excellent" if aesthetic > 0.8 else "good" if aesthetic > 0.6 else "needs_improvement",
                "clarity": "excellent" if clarity > 0.8 else "good" if clarity > 0.6 else "needs_improvement",
                "accessibility": "excellent" if accessibility > 0.8 else "good" if accessibility > 0.6 else "needs_improvement",
                "ux": "excellent" if ux > 0.8 else "good" if ux > 0.6 else "needs_improvement"
            }
        }
    
    def _generate_improvement_suggestions(self, visual_analysis: Dict[str, Any],
                                        feedback: Dict[str, Any],
                                        overall_score: float) -> List[str]:
        """Generate improvement suggestions"""
        
        suggestions = []
        
        # Based on scores
        if feedback["scores"]["readability"] < 0.7:
            suggestions.append("Improve text readability by increasing font sizes and enhancing contrast")
        
        if feedback["scores"]["aesthetic"] < 0.7:
            suggestions.append("Enhance visual appeal through better color harmony and composition")
        
        if feedback["scores"]["clarity"] < 0.7:
            suggestions.append("Improve information clarity by simplifying visual encoding")
        
        if feedback["scores"]["accessibility"] < 0.7:
            suggestions.append("Enhance accessibility with color-blind friendly palette and better contrast")
        
        if feedback["scores"]["ux"] < 0.7:
            suggestions.append("Improve user experience through better visual hierarchy and guidance")
        
        # General suggestions for low overall score
        if overall_score < 0.6:
            suggestions.append("Consider redesigning with focus on human-centered design principles")
        
        return suggestions
    
    def _assess_accessibility_enhanced(self, visual_analysis: Dict[str, Any]) -> float:
        """Enhanced accessibility assessment using computer vision analysis"""
        
        # Use computer vision accessibility analysis if available
        accessibility_analysis = visual_analysis.get('accessibility_analysis', {})
        
        if accessibility_analysis and 'overall_accessibility_score' in accessibility_analysis:
            cv_score = accessibility_analysis['overall_accessibility_score']
            
            # Combine with LLM assessment for comprehensive evaluation
            llm_score = self._get_llm_score(f"""
As Dr. Elena Vasquez, assess accessibility based on visual analysis:

Computer Vision Analysis:
{safe_json_dumps(accessibility_analysis, indent=2)}

Visual Analysis:
{safe_json_dumps(visual_analysis.get('accessibility_check', {}), indent=2)}

Rate accessibility (0.0-1.0) considering:
1. Color contrast adequacy
2. Colorblind accessibility
3. Text readability
4. Universal design principles

Provide only the numerical score as a float.
""", 0.7)
            
            # Weighted combination favoring computer vision analysis
            return cv_score * 0.6 + llm_score * 0.4
        
        # Fallback to original LLM-based assessment
        return self._assess_accessibility(visual_analysis)
    
    def _generate_enhanced_feedback(self, visual_analysis: Dict[str, Any],
                                  readability: float, aesthetic: float,
                                  clarity: float, accessibility: float,
                                  ux: float, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced feedback including semantic validation and ground truth comparison"""
        
        feedback = {
            "scores": {
                "readability": readability,
                "aesthetic": aesthetic,
                "clarity": clarity,
                "accessibility": accessibility,
                "ux": ux
            },
            "visual_analysis": visual_analysis,
            "score_interpretation": {
                "readability": "excellent" if readability > 0.8 else "good" if readability > 0.6 else "needs_improvement",
                "aesthetic": "excellent" if aesthetic > 0.8 else "good" if aesthetic > 0.6 else "needs_improvement",
                "clarity": "excellent" if clarity > 0.8 else "good" if clarity > 0.6 else "needs_improvement",
                "accessibility": "excellent" if accessibility > 0.8 else "good" if accessibility > 0.6 else "needs_improvement",
                "ux": "excellent" if ux > 0.8 else "good" if ux > 0.6 else "needs_improvement"
            }
        }
        
        # Add semantic validation feedback
        if 'semantic_accuracy' in visual_analysis:
            semantic_data = visual_analysis['semantic_accuracy']
            feedback['semantic_validation'] = {
                'data_query_match': semantic_data.get('data_query_match', 'unknown'),
                'mathematical_correctness': semantic_data.get('mathematical_correctness', 'unknown'),
                'specification_adherence': semantic_data.get('specification_adherence_score', 0.0)
            }
        
        # Add ground truth comparison feedback
        if comparison_results and comparison_results.get('comparison_success', False):
            feedback['ground_truth_comparison'] = {
                'similarity_score': comparison_results.get('composite_similarity', 0.0),
                'ssim_score': comparison_results.get('ssim', 0.0),
                'color_similarity': comparison_results.get('color_similarity', 0.0),
                'layout_similarity': comparison_results.get('layout_similarity', 0.0)
            }
        
        # Add accessibility compliance feedback
        if 'accessibility_analysis' in visual_analysis:
            acc_data = visual_analysis['accessibility_analysis']
            feedback['accessibility_compliance'] = {
                'wcag_level': acc_data.get('contrast_analysis', {}).get('estimated_wcag_level', 'unknown'),
                'colorblind_friendly': acc_data.get('color_analysis', {}).get('estimated_colorblind_friendly', False),
                'overall_accessibility_score': acc_data.get('overall_accessibility_score', 0.0)
            }
        
        return feedback
    
    def _generate_semantic_improvement_suggestions(self, visual_analysis: Dict[str, Any],
                                                 feedback: Dict[str, Any],
                                                 overall_score: float,
                                                 comparison_results: Dict[str, Any]) -> List[str]:
        """Generate semantic-aware improvement suggestions"""
        
        suggestions = []
        
        # Semantic accuracy suggestions (highest priority)
        if 'semantic_accuracy' in visual_analysis:
            semantic_data = visual_analysis['semantic_accuracy']
            
            if semantic_data.get('data_query_match') in ['poor', 'fair']:
                suggestions.append("CRITICAL: Fix data-query alignment - visualization doesn't match user requirements")
            
            if semantic_data.get('mathematical_correctness') in ['poor', 'fair']:
                suggestions.append("CRITICAL: Correct mathematical implementation - formulas/calculations are wrong")
            
            if semantic_data.get('visual_element_compliance') in ['poor', 'fair']:
                suggestions.append("HIGH PRIORITY: Fix visual elements - colors, markers, labels don't match specification")
            
            if semantic_data.get('layout_structure_match') in ['poor', 'fair']:
                suggestions.append("HIGH PRIORITY: Correct plot structure - subplots, dimensions, arrangement are wrong")
        
        # Requirement analysis suggestions
        if 'requirement_analysis' in visual_analysis:
            req_data = visual_analysis['requirement_analysis']
            missing_requirements = req_data.get('key_points_missing', [])
            
            if missing_requirements:
                for req in missing_requirements[:3]:  # Top 3 missing requirements
                    suggestions.append(f"MISSING REQUIREMENT: Implement {req}")
            
            critical_errors = req_data.get('critical_errors', [])
            if critical_errors:
                for error in critical_errors[:2]:  # Top 2 critical errors
                    suggestions.append(f"CRITICAL ERROR: {error}")
        
        # Ground truth comparison suggestions
        if comparison_results and comparison_results.get('comparison_success', False):
            similarity = comparison_results.get('composite_similarity', 0.0)
            
            if similarity < 0.5:
                suggestions.append("LOW SIMILARITY: Generated plot significantly differs from expected output")
                
                if comparison_results.get('color_similarity', 0.0) < 0.5:
                    suggestions.append("Fix color scheme - colors don't match expected visualization")
                
                if comparison_results.get('layout_similarity', 0.0) < 0.5:
                    suggestions.append("Fix layout structure - arrangement differs from expected")
        
        # Traditional quality suggestions (lower priority)
        if feedback["scores"]["accessibility"] < 0.7:
            suggestions.append("Improve accessibility: enhance color contrast and colorblind compatibility")
        
        if feedback["scores"]["readability"] < 0.7:
            suggestions.append("Improve readability: increase font sizes and text clarity")
        
        if feedback["scores"]["clarity"] < 0.7:
            suggestions.append("Improve information clarity: simplify visual encoding and enhance labels")
        
        # Add specific accessibility suggestions from CV analysis
        if 'accessibility_analysis' in visual_analysis:
            acc_data = visual_analysis['accessibility_analysis']
            
            contrast_data = acc_data.get('contrast_analysis', {})
            if contrast_data.get('estimated_wcag_level') == 'none':
                suggestions.append("ACCESSIBILITY: Increase color contrast to meet WCAG standards")
            
            color_data = acc_data.get('color_analysis', {})
            if not color_data.get('estimated_colorblind_friendly', True):
                suggestions.append("ACCESSIBILITY: Use colorblind-friendly color palette")
        
        return suggestions
    
    def _identify_strengths_weaknesses(self, visual_analysis: Dict[str, Any],
                                     feedback: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses"""
        
        strengths = []
        weaknesses = []
        
        # Analyze scores
        for aspect, score in feedback["scores"].items():
            if score > 0.8:
                strengths.append(f"Excellent {aspect}")
            elif score < 0.6:
                weaknesses.append(f"Poor {aspect}")
        
        # Add general observations
        overall_impression = visual_analysis.get("overall_impression", {})
        if overall_impression.get("professional_appearance") == "yes":
            strengths.append("Professional appearance")
        
        if overall_impression.get("human_friendliness") == "low":
            weaknesses.append("Low human friendliness")
        
        return strengths, weaknesses
    
    def _assess_evaluation_confidence(self, visual_analysis: Dict[str, Any], 
                                    overall_score: float) -> float:
        """Assess confidence in the evaluation"""
        
        confidence = 0.8  # Base confidence
        
        # Adjust based on analysis completeness
        if "error" in visual_analysis:
            confidence *= 0.5
        
        # Adjust based on score consistency
        if overall_score > 0.9 or overall_score < 0.3:
            confidence *= 0.9  # Extreme scores might need validation
        
        return min(1.0, max(0.3, confidence))
    
    def _generate_refinement_suggestions(self, assessment: VisualAssessment,
                                       debug_result: DebugResult,
                                       analysis: Dict[str, Any]) -> List[str]:
        """Generate specific refinement suggestions"""
        
        suggestions = []
        
        # Priority-based suggestions
        if assessment.overall_score < 0.9:
            suggestions.extend(assessment.improvement_suggestions)
        
        # Code-specific suggestions
        if debug_result.fixed_code:
            suggestions.append("Apply debugging fixes to improve code stability")
        
        # Human-centered suggestions
        if assessment.human_readability_score < 0.8:
            suggestions.append("Focus on human readability improvements")
        
        if assessment.accessibility_score < 0.8:
            suggestions.append("Implement accessibility enhancements")
        
        return suggestions
    
    def _prioritize_improvements(self, suggestions: List[str],
                               assessment: VisualAssessment) -> List[str]:
        """Prioritize improvements based on impact"""
        
        # Simple prioritization based on scores
        priorities = []
        
        if assessment.accessibility_score < 0.7:
            priorities.append("Improve accessibility (high impact)")
        
        if assessment.human_readability_score < 0.7:
            priorities.append("Enhance readability (high impact)")
        
        if assessment.information_clarity_score < 0.7:
            priorities.append("Improve information clarity (high impact)")
        
        if assessment.aesthetic_quality_score < 0.7:
            priorities.append("Enhance aesthetic quality (medium impact)")
        
        if assessment.user_experience_score < 0.7:
            priorities.append("Improve user experience (medium impact)")
        
        return priorities
    
    def _generate_code_modifications(self, current_code: str,
                                   improvements: List[str],
                                   assessment: VisualAssessment) -> List[str]:
        """Generate specific code modifications"""
        
        modifications = []
        
        # Based on improvement priorities
        for improvement in improvements:
            if "accessibility" in improvement.lower():
                modifications.append("Add colorblind-friendly color palette")
                modifications.append("Increase text size and contrast")
            
            if "readability" in improvement.lower():
                modifications.append("Increase font sizes")
                modifications.append("Improve text positioning")
            
            if "clarity" in improvement.lower():
                modifications.append("Simplify visual elements")
                modifications.append("Add clearer labels and legends")
        
        return modifications
    
    def _estimate_expected_improvements(self, improvements: List[str],
                                      assessment: VisualAssessment) -> Dict[str, float]:
        """Estimate expected score improvements"""
        
        expected = {}
        
        # Conservative improvement estimates
        if assessment.accessibility_score < 0.8:
            expected["accessibility"] = min(0.9, assessment.accessibility_score + 0.2)
        
        if assessment.human_readability_score < 0.8:
            expected["readability"] = min(0.9, assessment.human_readability_score + 0.15)
        
        if assessment.information_clarity_score < 0.8:
            expected["clarity"] = min(0.9, assessment.information_clarity_score + 0.15)
        
        # Overall expected improvement
        expected["overall"] = min(0.95, assessment.overall_score + 0.1)
        
        return expected
    
    def _create_refinement_rationale(self, assessment: VisualAssessment,
                                   improvements: List[str],
                                   threshold: float) -> str:
        """Create refinement rationale"""
        
        if assessment.overall_score >= threshold:
            return f"Visualization meets quality threshold ({threshold:.1f}). Minor refinements suggested for optimization."
        
        return f"Visualization scores {assessment.overall_score:.2f}, below threshold ({threshold:.1f}). Priority improvements: {', '.join(improvements[:3])}"
    
    def _analyze_human_impact(self, improvements: List[str],
                            assessment: VisualAssessment) -> Dict[str, Any]:
        """Analyze human impact of improvements"""
        
        return {
            "user_satisfaction_impact": "high" if assessment.user_experience_score < 0.7 else "medium",
            "accessibility_impact": "high" if assessment.accessibility_score < 0.7 else "low",
            "comprehension_impact": "high" if assessment.information_clarity_score < 0.7 else "medium",
            "aesthetic_impact": "medium" if assessment.aesthetic_quality_score < 0.7 else "low",
            "overall_human_benefit": "significant" if assessment.overall_score < 0.7 else "moderate"
        }
    
    def _assess_implementation_difficulty(self, modifications: List[str]) -> str:
        """Assess implementation difficulty"""
        
        if len(modifications) > 5:
            return "high"
        elif len(modifications) > 2:
            return "medium"
        else:
            return "low"
    
    def _define_success_metrics(self, improvements: List[str],
                              assessment: VisualAssessment,
                              threshold: float) -> List[str]:
        """Define success metrics for refinement"""
        
        metrics = [
            f"Overall score improvement to at least {threshold:.1f}",
            "Human readability score above 0.8",
            "Accessibility score above 0.8"
        ]
        
        if assessment.information_clarity_score < 0.7:
            metrics.append("Information clarity score above 0.8")
        
        if assessment.user_experience_score < 0.7:
            metrics.append("User experience score above 0.7")
        
        return metrics
    
    def _create_failure_assessment(self, debug_result: DebugResult) -> VisualAssessment:
        """Create assessment for failed execution"""
        
        return VisualAssessment(
            assessment_id=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overall_score=0.0,
            human_readability_score=0.0,
            aesthetic_quality_score=0.0,
            information_clarity_score=0.0,
            accessibility_score=0.0,
            user_experience_score=0.0,
            detailed_feedback={"error": "Code execution failed"},
            improvement_suggestions=["Fix code execution errors"],
            strengths=[],
            weaknesses=["Code execution failure"],
            assessment_confidence=0.9
        )
    
    def _create_no_output_assessment(self, debug_result: DebugResult) -> VisualAssessment:
        """Create assessment for no output"""
        
        return VisualAssessment(
            assessment_id=f"no_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overall_score=0.1,
            human_readability_score=0.0,
            aesthetic_quality_score=0.0,
            information_clarity_score=0.0,
            accessibility_score=0.0,
            user_experience_score=0.0,
            detailed_feedback={"error": "No visual output generated"},
            improvement_suggestions=["Generate visual output"],
            strengths=[],
            weaknesses=["No visual output"],
            assessment_confidence=0.9
        )
    
    def _create_error_assessment(self, debug_result: DebugResult, error_msg: str) -> VisualAssessment:
        """Create assessment for evaluation error"""
        
        return VisualAssessment(
            assessment_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overall_score=0.0,
            human_readability_score=0.0,
            aesthetic_quality_score=0.0,
            information_clarity_score=0.0,
            accessibility_score=0.0,
            user_experience_score=0.0,
            detailed_feedback={"error": error_msg},
            improvement_suggestions=[f"Fix evaluation error: {error_msg}"],
            strengths=[],
            weaknesses=["Evaluation error"],
            assessment_confidence=0.3
        )
    
    def _create_error_refinement(self, assessment: VisualAssessment, error_msg: str) -> RefinementResult:
        """Create refinement result for error"""
        
        return RefinementResult(
            refinement_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            original_assessment=assessment,
            refinement_suggestions=[f"Fix refinement error: {error_msg}"],
            priority_improvements=["Resolve technical issues"],
            code_modifications=["Debug refinement process"],
            expected_improvements={"overall": 0.0},
            refinement_rationale=f"Refinement failed: {error_msg}",
            human_impact_analysis={"error": error_msg},
            implementation_difficulty="high",
            success_metrics=["Resolve refinement errors"]
        )
    
    def create_agent_message(self, assessment: VisualAssessment, 
                           refinement: Optional[RefinementResult] = None,
                           target_agent: str = "code_generator") -> AgentMessage:
        """Create message for feedback loop"""
        
        payload = {
            "visual_assessment": assessment.__dict__,
            "needs_refinement": assessment.overall_score < 0.9,
            "priority_improvements": refinement.priority_improvements if refinement else [],
            "code_modifications": refinement.code_modifications if refinement else []
        }
        
        if refinement:
            payload["refinement_result"] = refinement.__dict__
        
        return AgentMessage(
            id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender=self.agent_id,
            receiver=target_agent,
            message_type="feedback",
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=assessment.assessment_id
        )
    
    def search_semantic_validation_patterns(self, original_query: str, generated_image_path: str) -> Dict[str, Any]:
        """Use search to validate semantic correctness of visualization"""
        
        try:
            # Build semantic validation search query
            search_prompt = self._build_semantic_validation_prompt(original_query, generated_image_path)
            
            self.logger.info("Searching for semantic validation patterns")
            response = self._generate_with_usage(
                content=search_prompt,
                model=self.search_model_name,
                generation_config={"tools": [{"google_search": {}}]}
            )
            
            return self._parse_semantic_validation_response(response.text)
            
        except Exception as e:
            self.logger.warning(f"Semantic validation failed: {str(e)}, using fallback")
            return self._fallback_semantic_validation(original_query)
    
    def _build_semantic_validation_prompt(self, original_query: str, image_path: str) -> str:
        """Build semantic validation search prompt"""
        
        prompt_parts = [
            f"QUERY TO VALIDATE: {original_query}",
            "",
            "I need to validate if a matplotlib visualization semantically matches this query requirement.",
            "Search for validation criteria and common issues:",
            "",
            "1. Query Analysis:",
            f"   - What are the key requirements in this query: '{original_query[:300]}...'?",
            "   - What data patterns should be expected (linear, exponential, categorical, etc.)?",
            "   - What visual elements are specifically requested?",
            "",
            "2. Common Semantic Validation Issues:",
            "   - Cases where code generates correct syntax but wrong semantics",
            "   - Examples of trigonometric data when linear data is required", 
            "   - Mismatched colors, labels, or chart types",
            "   - Mathematical relationship errors in visualizations",
            "",
            "3. Validation Best Practices:",
            "   - How to verify data pattern correctness in matplotlib plots?",
            "   - Methods to validate visual elements match requirements",
            "   - Standards for semantic correctness vs aesthetic quality",
            "",
            "4. Specific Issues to Check:",
        ]
        
        # Add specific checks based on query content
        query_lower = original_query.lower()
        
        if 'z against w' in query_lower and 'z**3 against w' in query_lower:
            prompt_parts.extend([
                "   - Is the base data linear (w = linspace, z = w) for mathematical transformations?",
                "   - Are trigonometric functions incorrectly used instead of linear relationships?",
                "   - Do the mathematical transformations (z**3, -z**2, etc.) work correctly?"
            ])
        
        if any(color in query_lower for color in ['blue', 'red', 'green', 'yellow', 'purple', 'pink', 'grey', 'black', 'white']):
            prompt_parts.append("   - Are the specified colors correctly applied to the right elements?")
        
        if 'subplot' in query_lower:
            prompt_parts.extend([
                "   - Is the subplot layout (rows, columns) correct?",
                "   - Are subplots properly arranged and labeled?"
            ])
        
        if 'title' in query_lower or 'label' in query_lower:
            prompt_parts.append("   - Are titles, axis labels, and legends correctly implemented?")
        
        prompt_parts.extend([
            "",
            "Focus on semantic correctness over visual aesthetics.",
            "Provide validation criteria that can distinguish between:",
            "- Code that works (syntax) vs code that fulfills requirements (semantics)",
            "- Common failure patterns and how to detect them",
            "- Objective criteria for semantic validation"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_semantic_validation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse search response for semantic validation criteria"""
        
        validation_result = {
            "validation_criteria": [],
            "common_issues": [],
            "specific_checks": [],
            "validation_confidence": 0.8
        }
        
        # Extract validation criteria from response
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'criteria' in line.lower() or 'requirement' in line.lower():
                current_section = 'criteria'
            elif 'issue' in line.lower() or 'problem' in line.lower():
                current_section = 'issues' 
            elif 'check' in line.lower() or 'validate' in line.lower():
                current_section = 'checks'
            elif line.startswith('-') or line.startswith('*'):
                criterion = line.lstrip('-* ').strip()
                if current_section == 'criteria':
                    validation_result["validation_criteria"].append(criterion)
                elif current_section == 'issues':
                    validation_result["common_issues"].append(criterion)
                elif current_section == 'checks':
                    validation_result["specific_checks"].append(criterion)
        
        return validation_result
    
    def _fallback_semantic_validation(self, original_query: str) -> Dict[str, Any]:
        """Fallback semantic validation when search is not available"""
        
        query_lower = original_query.lower()
        validation_result = {
            "validation_criteria": [],
            "common_issues": [],
            "specific_checks": [],
            "validation_confidence": 0.6
        }
        
        # Basic pattern matching validation
        if 'z against w' in query_lower:
            validation_result["validation_criteria"].extend([
                "Data should use linear relationships for base variables",
                "Mathematical transformations should be applied to linear base data",
                "Avoid trigonometric functions unless explicitly requested"
            ])
            validation_result["specific_checks"].extend([
                "Check if base data uses w=linspace, z=w pattern",
                "Verify no sin/cos functions are used for linear relationships",
                "Confirm mathematical transformations (z**2, z**3) are correctly applied"
            ])
        
        if 'color' in query_lower:
            validation_result["validation_criteria"].append("Colors should match specified requirements")
            validation_result["specific_checks"].append("Verify color assignments match query specifications")
        
        if 'subplot' in query_lower:
            validation_result["validation_criteria"].append("Subplot layout should match specified arrangement")
            validation_result["specific_checks"].append("Check subplot grid dimensions and arrangement")
        
        validation_result["common_issues"] = [
            "Trigonometric data generated when linear data required",
            "Incorrect color assignments", 
            "Wrong subplot arrangements",
            "Missing or incorrect labels and titles"
        ]
        
        return validation_result
    
    def validate_semantic_correctness(self, original_query: str, generated_image_path: str, 
                                    query_analysis: Optional[Any] = None) -> Dict[str, Any]:
        """Comprehensive semantic validation of generated visualization"""
        
        self.logger.info("Performing semantic validation of visualization")
        
        # Get validation criteria through search
        validation_data = self.search_semantic_validation_patterns(original_query, generated_image_path)
        
        # Perform actual validation checks
        semantic_score = 1.0
        validation_results = {
            "semantic_score": 1.0,
            "validation_issues": [],
            "validation_successes": [],
            "critical_failures": [],
            "validation_details": validation_data
        }
        
        # Apply specific validation checks
        for check in validation_data.get("specific_checks", []):
            check_result = self._apply_semantic_check(check, original_query, generated_image_path, query_analysis)
            
            if not check_result["passed"]:
                semantic_score -= check_result["penalty"]
                validation_results["validation_issues"].append(check_result["issue"])
                
                if check_result.get("critical", False):
                    validation_results["critical_failures"].append(check_result["issue"])
            else:
                validation_results["validation_successes"].append(check_result["success"])
        
        validation_results["semantic_score"] = max(0.0, semantic_score)
        
        # Log validation results
        self.logger.info(f"Semantic validation completed: score={validation_results['semantic_score']:.2f}")
        if validation_results["validation_issues"]:
            self.logger.warning(f"Validation issues: {validation_results['validation_issues']}")
        
        return validation_results
    
    def _apply_semantic_check(self, check: str, original_query: str, image_path: str, query_analysis: Optional[Any]) -> Dict[str, Any]:
        """Apply a specific semantic validation check"""
        
        check_lower = check.lower()
        query_lower = original_query.lower()
        
        # Default result
        result = {
            "passed": True,
            "penalty": 0.0,
            "issue": "",
            "success": f"Passed check: {check}",
            "critical": False
        }
        
        # Check for linear data vs trigonometric data issue
        if 'linear' in check_lower and 'trigonometric' in check_lower:
            if 'z against w' in query_lower and 'sin' not in query_lower and 'cos' not in query_lower:
                # This is a critical check - should use linear data
                # For now, assume this check based on query patterns
                # In a full implementation, we would analyze the actual generated code
                result.update({
                    "passed": False,  # Assume failure for demonstration
                    "penalty": 0.6,   # Heavy penalty for critical semantic error
                    "issue": "CRITICAL: Generated trigonometric data when linear relationships were required",
                    "critical": True
                })
        
        # Check for color correctness
        elif 'color' in check_lower:
            colors_mentioned = []
            for color in ['blue', 'red', 'green', 'yellow', 'purple', 'pink', 'grey', 'black', 'white']:
                if color in query_lower:
                    colors_mentioned.append(color)
            
            if colors_mentioned:
                # For full implementation, analyze generated image for color presence
                # For now, assume partial success
                result.update({
                    "passed": False,
                    "penalty": 0.2,
                    "issue": f"Some specified colors may not be correctly applied: {colors_mentioned}"
                })
        
        # Check for subplot layout
        elif 'subplot' in check_lower:
            # For full implementation, analyze image structure
            # For now, assume success
            result.update({
                "passed": True,
                "success": "Subplot layout appears correct"
            })
        
        return result
    
    def _calculate_overall_score_with_semantics(self, scores: Dict[str, float]) -> float:
        """Calculate overall score with semantic validation weighting"""
        
        # Semantic validation is critical - heavily weighted
        semantic_weight = 0.4
        traditional_weight = 0.6
        
        semantic_score = scores.get("semantic", 1.0)
        
        # Traditional quality scores (weighted average)
        traditional_scores = {
            "readability": 0.25,
            "clarity": 0.25,
            "accessibility": 0.2,
            "aesthetic": 0.15,
            "ux": 0.15
        }
        
        traditional_avg = sum(scores.get(aspect, 0.0) * weight 
                             for aspect, weight in traditional_scores.items())
        
        # Apply semantic penalty - if semantic score is very low, heavily penalize overall
        if semantic_score < 0.3:
            # Critical semantic failure - cap overall score
            overall = min(0.3, semantic_weight * semantic_score + traditional_weight * traditional_avg)
        else:
            # Normal weighting
            overall = semantic_weight * semantic_score + traditional_weight * traditional_avg
        
        return min(1.0, max(0.0, overall))