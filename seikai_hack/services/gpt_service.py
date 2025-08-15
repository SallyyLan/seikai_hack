import os
from typing import Dict, List, Any
import json
import requests

class GPTService:
    def __init__(self):
        self.model = os.getenv("GPT_OSS_MODEL")
        self.hf_token = os.getenv("HF_TOKEN")
        
        if not self.model:
            raise ValueError("GPT_OSS_MODEL not found in environment variables")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        # Set up headers for Hugging Face API
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
    
    async def _call_gpt_oss(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """Call GPT OSS model via Hugging Face API"""
        try:
            print(f"🔍 GPT Service: Calling model {self.model}")
            print(f"🔍 GPT Service: Messages count: {len(messages)}")
            print(f"🔍 GPT Service: Max tokens: {max_tokens}")
            
            # Prepare the request payload
            payload = {
                "inputs": messages,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "do_sample": True
                }
            }
            
            print(f"🔍 GPT Service: Making request to Hugging Face API...")
            
            # Make request to Hugging Face API
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=self.headers,
                json=payload
            )
            
            print(f"🔍 GPT Service: Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"🔍 GPT Service: Response received successfully")
                # Extract the generated text from the response
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    return result.get("generated_text", "")
                else:
                    return str(result)
            else:
                print(f"❌ GPT Service: API call failed with status {response.status_code}")
                print(f"❌ GPT Service: Response text: {response.text[:200]}...")
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ GPT Service: Exception in _call_gpt_oss: {str(e)}")
            raise Exception(f"Failed to call GPT OSS: {str(e)}")
    
    async def test_connection(self) -> bool:
        """Test if the GPT service is working with a simple prompt"""
        try:
            print(f"🧪 Testing GPT service connection...")
            print(f"🧪 Model: {self.model}")
            print(f"🧪 Token: {self.hf_token[:10]}..." if self.hf_token else "🧪 No token")
            
            # Simple test prompt
            test_messages = [
                {"role": "user", "content": "Say 'Hello, GPT service is working!' and nothing else."}
            ]
            
            result = await self._call_gpt_oss(test_messages, max_tokens=50)
            print(f"🧪 Test result: {result}")
            
            if "Hello" in result or "working" in result:
                print("✅ GPT service test successful!")
                return True
            else:
                print("⚠️ GPT service test returned unexpected result")
                return False
                
        except Exception as e:
            print(f"❌ GPT service test failed: {str(e)}")
            return False

    async def generate_practice_exam(
        self, 
        prioritized_topics: List[Dict[str, Any]], 
        course_name: str,
        difficulty_level: str = "medium"
    ) -> str:
        """Generate a comprehensive practice exam in LaTeX format based on prioritized topics"""
        try:
            # Create topic list for the prompt
            topic_list = []
            for topic in prioritized_topics:
                topic_info = f"- {topic['name']} (Priority: {topic['study_priority']}, Score Value: {topic.get('score_value', 0)})"
                topic_list.append(topic_info)
            
            prompt = f"""
            Create a comprehensive practice exam for {course_name} covering these topics:
            
            {chr(10).join(topic_list)}
            
            Requirements:
            - Mix of question types: definitions, problem-solving, analysis, applications
            - Include algorithm tracing, complexity analysis, and real-world scenarios
            - Total: ~100 points, 90-120 minutes
            - Use proper LaTeX with \\documentclass, math mode, clear structure
            - Make questions challenging and application-focused, not just definitions
            - Include step-by-step problem solving and critical thinking
            
            Return ONLY LaTeX code, no explanations.
            """
            
            # Prepare messages for the Hugging Face API
            messages = [
                {"role": "system", "content": "You are an expert at creating comprehensive academic practice exams in LaTeX format. Focus on challenging, application-based questions that test deep understanding."},
                {"role": "user", "content": prompt}
            ]
            
            latex_code = await self._call_gpt_oss(messages)
            
            # Clean up the response to ensure it's valid LaTeX
            if not latex_code.startswith("\\documentclass"):
                # Try to find LaTeX content in the response
                start = latex_code.find("\\documentclass")
                if start != -1:
                    latex_code = latex_code[start:]
            
            return latex_code
            
        except Exception as e:
            raise Exception(f"Failed to generate practice exam: {str(e)}")
    
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
            
            # Prepare messages for the Hugging Face API
            messages = [
                {"role": "system", "content": "You are an expert academic tutor. Provide clear, constructive feedback."},
                {"role": "user", "content": prompt}
            ]
            
            analysis_json = await self._call_gpt_oss(messages)
            
            # Parse the response
            try:
                # Try to extract JSON from the response
                if "```json" in analysis_json:
                    json_start = analysis_json.find("```json") + 7
                    json_end = analysis_json.find("```", json_start)
                    json_content = analysis_json[json_start:json_end].strip()
                else:
                    # Look for JSON in the response
                    json_start = analysis_json.find("{")
                    json_end = analysis_json.rfind("}") + 1
                    json_content = analysis_json[json_start:json_end]
                
                analysis = json.loads(json_content)
                
                # Ensure all required fields are present
                required_fields = ["is_correct", "feedback", "topics", "confidence", "suggestions"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = self._get_default_value(field)
                
                return analysis
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return self._create_fallback_analysis(analysis_json)
                
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
            
            # Prepare messages for the Hugging Face API
            messages = [
                {"role": "system", "content": "You are an expert at identifying academic topics and concepts."},
                {"role": "user", "content": prompt}
            ]
            
            topics_text = await self._call_gpt_oss(messages)
            
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
