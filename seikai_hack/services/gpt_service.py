import openai
import os
from typing import Dict, List, Any
import json

class GPTService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        openai.api_key = self.api_key
        self.model = "gpt-4"  # Using GPT-4 for better reasoning
    
    async def analyze_work(self, extracted_text: str, course_context: str = "") -> Dict[str, Any]:
        """Analyze student work for correctness and provide feedback"""
        try:
            # Create a comprehensive prompt for analysis
            prompt = f"""
            You are an expert tutor analyzing a student's handwritten work. 
            
            Student's work (extracted from image):
            {extracted_text}
            
            Course context: {course_context}
            
            Please analyze this work and provide:
            1. Is the answer correct? (true/false)
            2. Detailed feedback explaining what's right/wrong
            3. List of topics/concepts this question covers
            4. Confidence level in your assessment (0.0-1.0)
            5. Specific suggestions for improvement
            
            Respond in JSON format:
            {{
                "is_correct": boolean,
                "feedback": "detailed explanation",
                "topics": ["topic1", "topic2"],
                "confidence": 0.95,
                "suggestions": ["suggestion1", "suggestion2"]
            }}
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert academic tutor. Provide clear, constructive feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            try:
                # Try to extract JSON from the response
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                else:
                    # Look for JSON in the response
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    json_content = content[json_start:json_end]
                
                analysis = json.loads(json_content)
                
                # Ensure all required fields are present
                required_fields = ["is_correct", "feedback", "topics", "confidence", "suggestions"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = self._get_default_value(field)
                
                return analysis
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return self._create_fallback_analysis(content)
                
        except Exception as e:
            raise Exception(f"GPT analysis failed: {str(e)}")
    
    async def identify_topics(self, question_text: str, course_materials: Dict[str, str]) -> List[str]:
        """Identify relevant topics from course materials"""
        try:
            materials_summary = "\n".join([f"{key}: {value[:500]}..." for key, value in course_materials.items()])
            
            prompt = f"""
            Based on the following course materials, identify the main topics/concepts that this question covers:
            
            Question: {question_text}
            
            Course Materials:
            {materials_summary}
            
            List the top 3-5 most relevant topics. Respond with just a comma-separated list.
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying academic topics and concepts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            topics_text = response.choices[0].message.content.strip()
            topics = [topic.strip() for topic in topics_text.split(",")]
            return topics
            
        except Exception as e:
            return ["General Problem Solving"]
    
    def _get_default_value(self, field: str) -> Any:
        """Get default values for missing fields"""
        defaults = {
            "is_correct": False,
            "feedback": "Unable to analyze work completely",
            "topics": ["Unknown Topic"],
            "confidence": 0.5,
            "suggestions": ["Please review the problem carefully"]
        }
        return defaults.get(field, "")
    
    def _create_fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Create fallback analysis when JSON parsing fails"""
        return {
            "is_correct": False,
            "feedback": f"Analysis completed but format unclear: {content[:200]}...",
            "topics": ["General Problem Solving"],
            "confidence": 0.3,
            "suggestions": ["Please review your work and try again"]
        }
