import google.generativeai as genai
import os
from typing import Dict, List, Any, Tuple
import json
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        
        # Try to use the latest available models
        try:
            # List available models to see what's supported
            models = genai.list_models()
            print(f"Available Gemini models: {[model.name for model in models]}")
            
            # Define model priority list (newest first)
            vision_models = [
                'gemini-2.0-flash-exp',
                'gemini-2.0-flash',
                'gemini-1.5-pro-latest',
                'gemini-1.5-pro',
                'gemini-1.5-flash-latest', 
                'gemini-1.5-flash',
                'gemini-pro-vision',
                'gemini-pro'
            ]
            
            text_models = [
                'gemini-2.0-flash-exp',
                'gemini-2.0-flash',
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash', 
                'gemini-1.5-pro-latest',
                'gemini-1.5-pro',
                'gemini-pro'
            ]
            
            # Try to initialize vision model (for images)
            self.model = None
            for model_name in vision_models:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    print(f"✅ Using {model_name} for vision tasks")
                    break
                except Exception:
                    continue
            
            if not self.model:
                raise Exception("No vision models available")
            
            # Try to initialize text model
            self.text_model = None
            for model_name in text_models:
                try:
                    self.text_model = genai.GenerativeModel(model_name)
                    print(f"✅ Using {model_name} for text tasks")
                    break
                except Exception:
                    continue
            
            if not self.text_model:
                # Use the same model for both if text model fails
                self.text_model = self.model
                print(f"⚠️  Using {self.model.model_name} for both vision and text tasks")
                    
        except Exception as e:
            print(f"⚠️  Error with model selection, using fallback: {e}")
            # Fallback to basic models
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                self.text_model = genai.GenerativeModel('gemini-pro')
                print("✅ Using fallback models")
            except Exception as fallback_error:
                raise ValueError(f"Failed to initialize Gemini models: {fallback_error}")
    
    async def test_model_availability(self) -> Dict[str, Any]:
        """Test if the Gemini models are working correctly"""
        try:
            # Test text model with a simple prompt
            test_prompt = "Hello, this is a test. Please respond with 'Test successful'."
            response = self.text_model.generate_content(test_prompt)
            
            return {
                "status": "success",
                "text_model": "working",
                "response": response.text,
                "models_available": True
            }
        except Exception as e:
            return {
                "status": "error",
                "text_model": "failed",
                "error": str(e),
                "models_available": False
            }
    
    async def process_exam_coverage_text(self, text_content: str) -> Dict[str, Any]:
        """Process exam coverage text directly and extract topics"""
        try:
            # Use Gemini to identify topics from exam coverage text
            topics = await self._identify_exam_topics(text_content)
            
            return {
                "topics": topics,
                "raw_text": text_content,
                "processed": True
            }
        except Exception as e:
            raise Exception(f"Failed to process exam coverage text: {str(e)}")
    
    async def process_exam_coverage(self, file_content: bytes, file_type: str) -> Dict[str, Any]:
        """Process exam coverage document and extract topics"""
        try:
            if file_type == "application/pdf":
                text_content = await self._extract_pdf_text(file_content)
            else:
                text_content = await self._extract_image_text(file_content)
            
            # Use Gemini to identify topics from exam coverage
            topics = await self._identify_exam_topics(text_content)
            
            return {
                "topics": topics,
                "raw_text": text_content,
                "processed": True
            }
        except Exception as e:
            raise Exception(f"Failed to process exam coverage: {str(e)}")
    
    async def process_practice_exam(self, file_content: bytes, file_type: str, exam_topics: List[str]) -> Dict[str, Any]:
        """Process practice exam and extract questions with topics and scores"""
        try:
            if file_type == "application/pdf":
                text_content = await self._extract_pdf_text(file_content)
            else:
                text_content = await self._extract_image_text(file_content)
            
            # Use Gemini to analyze practice exam and extract questions with topics and scores
            exam_analysis = await self._analyze_practice_exam(text_content, exam_topics)
            
            return {
                "questions": exam_analysis["questions"],
                "total_score": exam_analysis["total_score"],
                "raw_text": text_content,
                "processed": True
            }
        except Exception as e:
            raise Exception(f"Failed to process practice exam: {str(e)}")
    
    async def convert_to_latex(self, file_content: bytes, file_type: str) -> str:
        """Convert handwritten or PDF content to LaTeX code"""
        try:
            if file_type == "application/pdf":
                text_content = await self._extract_pdf_text(file_content)
            else:
                text_content = await self._extract_image_text(file_content)
            
            # Use Gemini to convert to LaTeX
            latex_code = await self._text_to_latex(text_content)
            
            return latex_code
        except Exception as e:
            raise Exception(f"Failed to convert to LaTeX: {str(e)}")
    
    async def extract_text(self, file_content: bytes, file_type: str = None) -> str:
        """Extract text from uploaded file (PDF or image)"""
        try:
            if file_type == "application/pdf":
                return await self._extract_pdf_text(file_content)
            else:
                return await self._extract_image_text(file_content)
        except Exception as e:
            raise Exception(f"Failed to extract text: {str(e)}")
    
    async def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"PDF text extraction failed: {str(e)}")
    
    async def _extract_image_text(self, image_content: bytes) -> str:
        """Extract text from image using Gemini Vision"""
        try:
            image = Image.open(io.BytesIO(image_content))
            
            prompt = """
            Please extract all the text from this image. If this is handwritten work, 
            transcribe it exactly as written. If this is printed text, extract it accurately.
            Return only the extracted text without any additional commentary.
            """
            
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            raise Exception(f"Image text extraction failed: {str(e)}")
    
    async def _identify_exam_topics(self, text_content: str) -> List[str]:
        """Use Gemini to identify topics from exam coverage"""
        prompt = f"""
        You are an expert at analyzing exam coverage documents. 
        
        Please analyze the following exam coverage text and identify the main topics/concepts that will be covered on the exam.
        
        IMPORTANT: Consolidate related subtopics into main topics. For example:
        - "Hashtables: Load Factor", "Hashtables: Runtimes", "Hashtables: Deletion" → combine into "Hashtables"
        - "Binary Trees: Traversal", "Binary Trees: Insertion" → combine into "Binary Trees"
        - "Sorting: QuickSort", "Sorting: MergeSort" → combine into "Sorting Algorithms"
        
        Exam Coverage Text:
        {text_content}
        
        Please return a JSON array of consolidated topic names. Each topic should be a clear, main concept that students need to study.
        
        Example format:
        ["Big-O Notation", "Data Structures", "Sorting Algorithms", "Graph Theory", "Dynamic Programming"]
        
        Guidelines:
        - Combine related subtopics under main topic names
        - Avoid too many specific subtopics
        - Aim for 5-15 main topics total
        - Use clear, broad topic names
        
        Return only the JSON array, no additional text.
        """
        
        response = self.text_model.generate_content(prompt)
        try:
            # Try to extract JSON from response
            content = response.text.strip()
            if content.startswith('[') and content.endswith(']'):
                topics = json.loads(content)
            else:
                # Look for JSON in the response
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end != 0:
                    topics = json.loads(content[start:end])
                else:
                    # Fallback: split by common delimiters and consolidate
                    raw_topics = [topic.strip() for topic in content.replace('\n', ',').split(',') if topic.strip()]
                    topics = self._consolidate_topics(raw_topics)
            
            return topics if isinstance(topics, list) else []
        except json.JSONDecodeError:
            # Fallback parsing with consolidation
            lines = response.text.strip().split('\n')
            raw_topics = [line.strip().strip('- ').strip('* ').strip() for line in lines if line.strip()]
            return self._consolidate_topics(raw_topics)
    
    def _consolidate_topics(self, raw_topics: List[str]) -> List[str]:
        """Consolidate related subtopics into main topics"""
        consolidated = {}
        
        for topic in raw_topics:
            if not topic:
                continue
                
            # Check if this is a subtopic (contains ':')
            if ':' in topic:
                main_topic = topic.split(':')[0].strip()
                if main_topic not in consolidated:
                    consolidated[main_topic] = []
                consolidated[main_topic].append(topic)
            else:
                # This is already a main topic
                if topic not in consolidated:
                    consolidated[topic] = []
        
        # Return main topic names
        return list(consolidated.keys())
    
    async def _analyze_practice_exam(self, text_content: str, exam_topics: List[str]) -> Dict[str, Any]:
        """Use Gemini to analyze practice exam and extract questions with topics and scores"""
        prompt = f"""
        You are analyzing a practice exam. Your task is simple:

        FIRST: Find the total possible score for this exam. Look for:
        - "Total: X points" or "Total Points: X"
        - "Exam worth X points" or "X points total"
        - Sum of all individual question scores
        - Any number followed by "points" or "pts"

        SECOND: Group questions by topic and assign scores.

        Exam Topics to Consider (only these):
        {', '.join(exam_topics)}

        Practice Exam Text:
        {text_content}

        Return ONLY this JSON format (no other text):
        {{
            "topics": [
                {{
                    "name": "topic_name",
                    "score": total_points_for_this_topic,
                    "questions": [
                        {{
                            "question_text": "text of the question",
                            "score": individual_question_score,
                            "question_number": question_number
                        }}
                    ]
                }}
            ],
            "total_score": total_exam_score
        }}

        CRITICAL: 
        1. The total_score must be the actual total from the exam, not a calculation.
        2. Each topic MUST have a score > 0. If you can't determine the exact score for a topic, estimate it based on the number of questions covering that topic.
        3. The sum of all topic scores should equal the total_score.
        4. If a question covers multiple topics, split the score proportionally between those topics.
        5. IMPORTANT: Only include topics that actually appear in the practice exam. Do NOT create topics that don't exist in the exam content.
        6. If a topic from the exam coverage list is not covered in the practice exam, simply omit it from the topics array.
        """
        
        response = self.text_model.generate_content(prompt)
        
        # Debug: Log the raw response
        print(f"🤖 Gemini raw response: {response.text[:500]}...")
        
        try:
            content = response.text.strip()
            if content.startswith('{') and content.endswith('}'):
                analysis = json.loads(content)
                print(f"✅ Parsed JSON directly")
            else:
                # Look for JSON in the response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    json_content = content[start:end]
                    print(f"✅ Extracted JSON from response: {json_content[:200]}...")
                    analysis = json.loads(json_content)
                else:
                    raise json.JSONDecodeError("No JSON found in response")
            
            # Validate structure
            if "topics" not in analysis or "total_score" not in analysis:
                raise ValueError("Invalid analysis structure")
            
            print(f"✅ Analysis structure valid. Total score: {analysis.get('total_score')}")
            
            # Convert to the format expected by the priority queue
            questions = []
            for topic_data in analysis["topics"]:
                topic_name = topic_data["name"]
                for question in topic_data["questions"]:
                    questions.append({
                        "question_text": question["question_text"],
                        "topics": [topic_name],
                        "score": question["score"],
                        "question_number": question.get("question_number", len(questions) + 1)
                    })
            
            result = {
                "questions": questions,
                "topics": analysis["topics"],
                "total_score": analysis["total_score"]
            }
            
            print(f"✅ Returning analysis with {len(questions)} questions and total score: {result['total_score']}")
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Gemini analysis failed, using fallback: {str(e)}")
            print(f"⚠️ Raw response: {response.text[:500]}...")
            
            # Fallback: try to extract score from text content
            fallback_score = self._extract_fallback_score(text_content)
            
            return {
                "questions": [
                    {
                        "question_text": "Practice exam question (fallback mode)",
                        "topics": exam_topics[:1] if exam_topics else ["General"],
                        "score": fallback_score,
                        "question_number": 1
                    }
                ],
                "topics": [
                    {
                        "name": exam_topics[0] if exam_topics else "General",
                        "score": fallback_score,
                        "questions": []
                    }
                ],
                "total_score": fallback_score
            }
    
    def _extract_fallback_score(self, text_content: str) -> float:
        """Extract total score from text content when Gemini analysis fails"""
        try:
            import re
            
            print(f"🔍 Fallback: Analyzing text for score patterns...")
            print(f"🔍 Text preview: {text_content[:200]}...")
            
            # Pattern 1: Look for total score patterns (highest priority)
            total_patterns = [
                r'total[:\s]+(\d+(?:\.\d+)?)\s*points?',
                r'total\s+points?[:\s]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*points?\s*total',
                r'(\d+(?:\.\d+)?)\s*total\s*points?',
                r'exam\s+worth\s+(\d+(?:\.\d+)?)\s*points?',
                r'(\d+(?:\.\d+)?)\s*points?\s*exam',
                r'(\d+(?:\.\d+)?)\s*points?\s*possible',
                r'(\d+(?:\.\d+)?)\s*possible\s*points?'
            ]
            
            for pattern in total_patterns:
                match = re.search(pattern, text_content.lower())
                if match:
                    score = float(match.group(1))
                    print(f"✅ Found total score pattern: {score} points")
                    return score
            
            # Pattern 2: Look for individual scores and sum them
            score_patterns = [
                r'(\d+(?:\.\d+)?)\s*points?',
                r'(\d+(?:\.\d+)?)\s*pts?',
                r'\((\d+(?:\.\d+)?)\)',
                r'(\d+(?:\.\d+)?)\s*pt'
            ]
            
            scores = []
            for pattern in score_patterns:
                matches = re.findall(pattern, text_content.lower())
                for match in matches:
                    try:
                        score = float(match)
                        if score > 0 and score <= 100:  # Reasonable score range
                            scores.append(score)
                    except ValueError:
                        continue
            
            if scores:
                total = sum(scores)
                print(f"✅ Summed individual scores: {total} points")
                return total
            
            # Pattern 3: Count questions and estimate
            question_patterns = [
                r'question\s+(\d+)',
                r'problem\s+(\d+)',
                r'(\d+)\.\s*[a-z]',
                r'(\d+)\)\s*[a-z]',
                r'(\d+)\s*[a-z]\.',
                r'(\d+)\s*[a-z]\)'
            ]
            
            question_numbers = set()
            for pattern in question_patterns:
                matches = re.findall(pattern, text_content.lower())
                for match in matches:
                    try:
                        question_numbers.add(int(match))
                    except ValueError:
                        continue
            
            if question_numbers:
                # Assume 4-6 points per question (typical range)
                estimated_score = len(question_numbers) * 5.0
                print(f"✅ Estimated from {len(question_numbers)} questions: {estimated_score} points")
                return estimated_score
            
            # Pattern 4: Look for any number that might be a score
            number_pattern = r'(\d{2,3})\s*(?:points?|pts?|total|exam)'
            match = re.search(number_pattern, text_content.lower())
            if match:
                score = float(match.group(1))
                if 20 <= score <= 200:  # Reasonable exam score range
                    print(f"✅ Found potential score: {score} points")
                    return score
            
            print(f"⚠️ No score patterns found, using default: 50 points")
            return 50.0
            
        except Exception as e:
            print(f"⚠️ Fallback score extraction failed: {str(e)}")
            return 50.0
    
    async def _text_to_latex(self, text_content: str) -> str:
        """Convert text content to LaTeX code using Gemini"""
        prompt = f"""
        You are an expert at converting mathematical text and handwritten work to LaTeX code.
        
        Please convert the following text to proper LaTeX code:
        
        {text_content}
        
        If this contains mathematical expressions, use appropriate LaTeX math mode.
        If this contains handwritten work, convert it to clean LaTeX.
        If this contains diagrams or figures, add appropriate LaTeX comments.
        
        Return only the LaTeX code, no additional text or explanations.
        """
        
        response = self.text_model.generate_content(prompt)
        return response.text.strip()
