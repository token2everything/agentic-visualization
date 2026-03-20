"""
LLM Evaluator for agentic_vis_core - Simplified standalone version
"""

import os
import base64
import json
import logging
from typing import Dict, Any, Optional
import litellm


class LLMEvaluator:
    """
    LLM-based evaluator for benchmark evaluation
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Get configuration from config or environment
        project_id = self.config.get('project_id', os.getenv('GOOGLE_PROJECT_ID'))
        location = self.config.get('location', os.getenv('GOOGLE_LOCATION'))
        model_name = self.config.get('model_name')

        self.text_model_name = model_name
        self.vision_model_name = self.config.get('vision_model_name')

        self.logger.info(f"LLMEvaluator initialized using LiteLLM (vision uses {self.vision_model_name})")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def find_image_in_folder(self, folder_path: str) -> str:
        """
        Find final output image with priority order:
        1. final_result.png
        2. result.png
        3. plot.png
        """
        priority_files = ['final_result.png', 'result.png', 'plot.png']

        self.logger.info(f"Looking for images in folder: {folder_path}")
        if not os.path.exists(folder_path):
            self.logger.warning(f"Folder does not exist: {folder_path}")
            return None

        try:
            files = os.listdir(folder_path)
            self.logger.info(f"Files in {folder_path}: {files}")
        except Exception as e:
            self.logger.error(f"Error listing files in {folder_path}: {e}")
            return None

        for filename in priority_files:
            image_path = os.path.join(folder_path, filename)
            if os.path.exists(image_path):
                self.logger.info(f"Found image: {image_path}")
                return image_path

        self.logger.warning(f"No image files found in {folder_path}")
        return None

    def llm_evaluate(self, code: str, query: str, folder_path: str) -> str:
        """
        Evaluate code quality

        Args:
            code: Generated code to evaluate
            query: Original user query
            folder_path: Path to folder containing generated image

        Returns:
            LLM evaluation response with final score
        """
        image_path = self.find_image_in_folder(folder_path)
        executable = 'True' if image_path else 'False'

        prompt = f'''
        You are evaluating visualization code quality. Be generous in scoring - we want to recognize effort and reasonable implementations.

        **User Query**: {query}

        **Code**:
        ```
        {code}
        ```

        **Has Output**: {executable}

        Scoring guidelines (be lenient and allow reasonable interpretations):
        - 80-100: Code successfully creates a visualization that reasonably addresses the query (even if different from expected)
        - 60-80: Code creates a visualization with some relation to the query
        - 40-60: Code creates any meaningful visualization, even if not directly related
        - 20-40: Code attempts visualization but has issues
        - 0-20: Code fails to produce output or has major errors

        Important: If Has Output is True and code creates ANY visualization, give at least 40 points.
        Allow creative interpretations and different approaches. Focus on whether the code works and produces visual output.

        Give a brief assessment and final score.
        Format: [FINAL SCORE]: X
        '''

        try:
            response = litellm.completion(
                model=self.text_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=12000
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Code evaluation failed: {str(e)}")
            return f"[FINAL SCORE]: 0\nEvaluation failed due to error: {str(e)}"

    def vision_evaluate(
        self,
        ground_truth_path: str,
        folder_path: str,
        rollback_path: str = None,
        has_csv_data: bool = True
    ) -> str:
        """
        Evaluate visual output against ground truth

        Args:
            ground_truth_path: Path to ground truth image
            folder_path: Path to folder containing generated image
            rollback_path: Path to rollback image (optional)
            has_csv_data: Whether query uses CSV data

        Returns:
            LLM vision evaluation response with final score
        """
        image_path = self.find_image_in_folder(folder_path)

        if not image_path:
            query_id = folder_path.split('/')[-2] if '/' in folder_path else 'unknown'
            self.logger.warning(f"Query {query_id} has no image in folder {folder_path}")
            return f"No image generated - Score: 0\n\n[FINAL SCORE]: 0"

        generated_image_path = image_path
        ground_truth_image_path = ground_truth_path

        try:
            self.logger.info(f"Reading generated image: {generated_image_path}")
            with open(generated_image_path, "rb") as f:
                generated_image_data = f.read()

            self.logger.info(f"Reading ground truth image: {ground_truth_image_path}")
            with open(ground_truth_image_path, "rb") as f:
                ground_truth_image_data = f.read()

            if has_csv_data:
                prompt = '''
                You are an excellent judge at evaluating visualization plots between a model generated plot and the ground truth. You will be giving scores on how well it matches the ground truth plot.

                The generated plot will be given to you as the first figure. If the first figure is blank, that means the code failed to generate a figure.
                Another plot will be given to you as the second figure, which is the desired outcome of the user query, meaning it is the ground truth for you to reference.
                Please compare the two figures head to head and rate them.
                Suppose the second figure has a score of 100, rate the first figure on a scale from 0 to 100.
                Scoring should be carried out in the following aspect:
                1. Plot correctness:
                Compare closely between the generated plot and the ground truth, the more resemblance the generated plot has compared to the ground truth, the higher the score. The score should be proportionate to the resemblance between the two plots.
                In some rare occurrence, see if the data points are generated randomly according to the query, if so, the generated plot may not perfectly match the ground truth, but it is correct nonetheless.
                Only rate the first figure, the second figure is only for reference.
                If the first figure is blank, that means the code failed to generate a figure. Give a score of 0 on the Plot correctness.
                After scoring from the above aspect, please give a brief reason (1-2 sentences) and then a final score.
                Format: Brief reasoning. [FINAL SCORE]: 85
                '''
            else:
                prompt = '''
                You are an excellent judge at evaluating visualization plots between a model generated plot and the ground truth. You will be giving scores on how well it matches the ground truth plot.

                The generated plot will be given to you as the first figure. If the first figure is blank, that means the code failed to generate a figure.
                Another plot will be given to you as the second figure, which is the desired outcome of the user query, meaning it is the ground truth for you to reference.
                Please compare the two figures head to head and rate them.
                Suppose the second figure has a score of 100, rate the first figure on a scale from 0 to 100.
                Scoring should be carried out in the following aspect:
                1. Plot correctness:
                Compare closely between the generated plot and the ground truth only on semantic meaning. Ignore data/ploting shape and stylistic differences (e.g., aspect ratio, smoothing, point density, layout, fonts, colors, marker styles). The closer the generated plot's semantics match the ground truth (variables mapped, categories present, relationships, and label/legend correspondence), the higher the score. The score must be proportional to semantic resemblance between the two plots.
                Only rate the first figure, the second figure is only for reference.
                If the first figure is blank, that means the code failed to generate a figure. Give a score of 0 on the Plot correctness.
                After scoring from the above aspect, please give a brief reason (1-2 sentences) and then a final score.
                Format: Brief reasoning. [FINAL SCORE]: 85
                '''

            content = [{"type": "text", "text": prompt}]

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(generated_image_data).decode('utf-8')}"
                }
            })

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(ground_truth_image_data).decode('utf-8')}"
                }
            })

            self.logger.info("Generating evaluation using vision model...")
            response = litellm.completion(
                model=self.vision_model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.7,
                max_tokens=12000
            )
            self.logger.info("Vision evaluation completed successfully")
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Vision evaluation failed: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Vision evaluation failed: {str(e)} - Score: 0\n\n[FINAL SCORE]: 0"

    def extract_final_score(self, evaluation_response: str) -> int:
        """
        Extract the final score from LLM evaluation response

        Args:
            evaluation_response: Raw LLM response text

        Returns:
            Extracted score as integer (0 if not found)
        """
        try:
            import re
            match = re.search(r'\[FINAL SCORE\]:\s*(\d+)', evaluation_response)
            if match:
                return int(match.group(1))
            else:
                self.logger.warning("No [FINAL SCORE] found in response")
                return 0
        except Exception as e:
            self.logger.error(f"Score extraction failed: {str(e)}")
            return 0

    def evaluate_query_result(self, query_id: str, query_text: str, code: str,
                            folder_path: str, ground_truth_path: str,
                            rollback_path: str = None, has_csv_data: bool = True) -> Dict[str, Any]:
        """
        Complete evaluation of a query result including both code and vision evaluation

        Args:
            query_id: Unique identifier for the query
            query_text: Original user query text
            code: Generated code to evaluate
            folder_path: Path to folder containing generated image
            ground_truth_path: Path to ground truth image
            rollback_path: Optional path to rollback image
            has_csv_data: Whether query uses CSV data

        Returns:
            Complete evaluation results dictionary
        """
        self.logger.info(f"Starting evaluation for query {query_id}")

        try:
            image_path = self.find_image_in_folder(folder_path)

            # Code evaluation
            code_evaluation = self.llm_evaluate(code, query_text, folder_path)
            code_score = self.extract_final_score(code_evaluation)

            # Vision evaluation
            vision_evaluation = self.vision_evaluate(ground_truth_path, folder_path, rollback_path, has_csv_data)
            vision_score = self.extract_final_score(vision_evaluation)

            results = {
                'query_id': query_id,
                'query_text': query_text,
                'code_evaluation': {
                    'response': code_evaluation,
                    'score': code_score
                },
                'vision_evaluation': {
                    'response': vision_evaluation,
                    'score': vision_score
                },
                'overall_scores': {
                    'code_score': code_score,
                    'vision_score': vision_score,
                    'average_score': (code_score + vision_score) / 2
                },
                'image_exists': image_path is not None and os.path.exists(image_path),
                'image_path': image_path,
                'ground_truth_exists': os.path.exists(ground_truth_path)
            }

            self.logger.info(f"Evaluation completed for query {query_id}: code={code_score}, vision={vision_score}")
            return results

        except Exception as e:
            self.logger.error(f"Error in evaluate_query_result: {str(e)}")
            return {
                'query_id': query_id,
                'query_text': query_text,
                'code_evaluation': {'response': f"Evaluation failed: {str(e)}", 'score': 0},
                'vision_evaluation': {'response': f"Evaluation failed: {str(e)}", 'score': 0},
                'overall_scores': {'code_score': 0, 'vision_score': 0, 'average_score': 0},
                'image_exists': False,
                'image_path': None,
                'ground_truth_exists': os.path.exists(ground_truth_path) if ground_truth_path else False
            }
