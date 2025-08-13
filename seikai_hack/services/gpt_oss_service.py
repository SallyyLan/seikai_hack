import os
from openai import OpenAI
from typing import Dict, List, Any, Optional
import json
import asyncio

class GPTOSSService:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.gpt_oss_model = os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-20b:fireworks-ai")
        if not self.hf_token:
            raise ValueError("Hugging Face token not found in environment variables")
        
        # Initialize OpenAI client with Hugging Face base URL
        try:
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.hf_token,
            )
        except Exception as e:
            # Fallback initialization if there are compatibility issues
            print(f"Warning: OpenAI client initialization failed: {e}")
            print("GPT-OSS service will be disabled")
            self.client = None

    async def validate_latex_code(self, latex_code: str) -> Dict[str, Any]:
        """
        Validate LaTeX code and identify any syntax errors or issues
        """
        if not self.client:
            return {
                "is_valid": False,
                "errors": ["GPT-OSS service not available"],
                "warnings": [],
                "suggestions": ["Check API configuration"],
                "confidence": 0.0
            }
        
        try:
            prompt = f"""
            You are a LaTeX expert. Validate the following LaTeX code and identify any errors.
            
            LaTeX Code:
            {latex_code}
            
            Return a JSON response with:
            {{
                "is_valid": true/false,
                "errors": ["error1", "error2"],
                "warnings": ["warning1", "warning2"],
                "suggestions": ["suggestion1", "suggestion2"],
                "confidence": 0.95
            }}
            
            If the LaTeX is valid, return empty arrays for errors and warnings.
            """
            
            response = await self._generate_content(prompt)
            return self._parse_json_response(response)
            
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "suggestions": ["Check API connection"],
                "confidence": 0.0
            }

    async def classify_topics_from_latex(self, latex_code: str) -> Dict[str, Any]:
        """
        Classify topics from LaTeX code and determine importance
        """
        if not self.client:
            return {
                "topics": [],
                "total_questions": 0,
                "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
                "error": "GPT-OSS service not available"
            }
        
        try:
            prompt = f"""
            Analyze this LaTeX code from a practice exam and classify the topics covered.
            
            LaTeX Code:
            {latex_code}
            
            Return a JSON response with:
            {{
                "topics": [
                    {{
                        "name": "topic_name",
                        "frequency": 3,
                        "importance": "high/medium/low",
                        "subtopics": ["subtopic1", "subtopic2"]
                    }}
                ],
                "total_questions": 10,
                "difficulty_distribution": {{
                    "easy": 3,
                    "medium": 4,
                    "hard": 3
                }}
            }}
            
            Focus on mathematical topics like: calculus, algebra, probability, statistics, geometry, etc.
            """
            
            response = await self._generate_content(prompt)
            return self._parse_json_response(response)
            
        except Exception as e:
            return {
                "topics": [],
                "total_questions": 0,
                "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
                "error": str(e)
            }

    async def generate_practice_exam(self, topics: List[str], exam_format: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a practice exam based on identified topics and format
        """
        if not self.client:
            return {
                "exam_title": "Practice Exam",
                "questions": [],
                "total_time": "0 minutes",
                "instructions": "GPT-OSS service not available",
                "error": "Service disabled"
            }
        
        try:
            topics_str = ", ".join(topics)
            format_str = json.dumps(exam_format, indent=2)
            
            prompt = f"""
            Generate a practice exam based on these topics: {topics_str}
            
            Exam Format Requirements:
            {format_str}
            
            Return a JSON response with:
            {{
                "exam_title": "Practice Exam",
                "questions": [
                    {{
                        "question_number": 1,
                        "topic": "topic_name",
                        "question_latex": "\\\\question ...",
                        "solution_latex": "\\\\begin{{solution}} ... \\\\end{{solution}}",
                        "difficulty": "easy/medium/hard",
                        "estimated_time": "5 minutes"
                    }}
                ],
                "total_time": "60 minutes",
                "instructions": "Exam instructions here"
            }}
            
            Make sure the LaTeX code is valid and follows proper syntax.
            """
            
            response = await self._generate_content(prompt)
            return self._parse_json_response(response)
            
        except Exception as e:
            return {
                "exam_title": "Practice Exam",
                "questions": [],
                "total_time": "0 minutes",
                "instructions": "Generation failed",
                "error": str(e)
            }

    async def analyze_student_work(self, student_latex: str, correct_solution: str) -> Dict[str, Any]:
        """
        Analyze student's work and provide feedback
        """
        if not self.client:
            return {
                "is_correct": False,
                "score": 0.0,
                "feedback": "GPT-OSS service not available",
                "common_mistakes": [],
                "improvement_suggestions": ["Check API configuration"],
                "concept_understanding": "unknown"
            }
        
        try:
            prompt = f"""
            Analyze the student's work compared to the correct solution.
            
            Student's Work:
            {student_latex}
            
            Correct Solution:
            {correct_solution}
            
            Return a JSON response with:
            {{
                "is_correct": true/false,
                "score": 0.85,
                "feedback": "Detailed feedback on the work",
                "common_mistakes": ["mistake1", "mistake2"],
                "improvement_suggestions": ["suggestion1", "suggestion2"],
                "concept_understanding": "good/fair/poor"
            }}
            
            Be constructive and specific in your feedback.
            """
            
            response = await self._generate_content(prompt)
            return self._parse_json_response(response)
            
        except Exception as e:
            return {
                "is_correct": False,
                "score": 0.0,
                "feedback": f"Analysis failed: {str(e)}",
                "common_mistakes": [],
                "improvement_suggestions": ["Check your work carefully"],
                "concept_understanding": "unknown"
            }

    async def _generate_content(self, prompt: str) -> str:
        """Generate content using GPT-OSS via Hugging Face"""
        if not self.client:
            raise Exception("GPT-OSS client not available")
        
        try:
            response = self.client.chat.completions.create(
                model=self.gpt_oss_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"GPT-OSS API error: {str(e)}")

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from GPT-OSS"""
        try:
            # Try to extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_content = response_text[json_start:json_end].strip()
            else:
                # Look for JSON in the response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_content = response_text[json_start:json_end]
            
            return json.loads(json_content)
            
        except (json.JSONDecodeError, ValueError):
            # Fallback: return structured text response
            return {
                "raw_response": response_text,
                "parse_error": "Failed to parse JSON response"
            }

    def is_available(self) -> bool:
        """Check if GPT-OSS service is available"""
        return bool(self.hf_token and self.client)
