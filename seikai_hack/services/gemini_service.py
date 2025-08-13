import os
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from fastapi import UploadFile
import base64
import json
import PyPDF2
import io
import asyncio

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        if not self.api_key:
            raise ValueError("Gemini API key not found in environment variables")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    async def extract_latex_from_pdf(self, pdf_file: UploadFile) -> Dict[str, Any]:
        """
        Extract LaTeX code from uploaded PDF using Gemini
        """
        try:
            # Read PDF content
            pdf_content = await pdf_file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            
            # Extract text from all pages
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
            
            # Use Gemini to convert to LaTeX
            prompt = f"""
            Convert the following exam content to clean LaTeX code. 
            Focus on mathematical expressions, equations, and proper formatting.
            
            Content:
            {full_text[:4000]}  # Limit to avoid token limits
            
            Return ONLY the LaTeX code, no explanations. Use proper LaTeX syntax.
            """
            
            response = await self._generate_content(prompt)
            
            return {
                "success": True,
                "latex_code": response,
                "original_text": full_text[:1000],  # First 1000 chars for reference
                "page_count": len(pdf_reader.pages)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latex_code": "",
                "original_text": ""
            }

    async def extract_latex_from_image(self, image_file: UploadFile) -> Dict[str, Any]:
        """
        Extract LaTeX code from handwritten image using Gemini
        """
        try:
            # Read and encode image
            image_content = await image_file.read()
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            
            prompt = """
            Analyze this handwritten work and convert it to LaTeX code.
            Focus on mathematical expressions, equations, and proper formatting.
            
            Return ONLY the LaTeX code, no explanations. Use proper LaTeX syntax.
            """
            
            # Generate content with image
            response = await self._generate_content_with_image(image_base64, prompt)
            
            return {
                "success": True,
                "latex_code": response,
                "file_type": image_file.content_type
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latex_code": "",
                "file_type": ""
            }

    async def _generate_content(self, prompt: str) -> str:
        """Generate text content using Gemini"""
        try:
            # Run the synchronous Gemini call in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, prompt)
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    async def _generate_content_with_image(self, image_base64: str, prompt: str) -> str:
        """Generate content with image input using Gemini"""
        try:
            # Run the synchronous Gemini call in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self.model.generate_content,
                [
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {"text": prompt}
                ]
            )
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Gemini API error with image: {str(e)}")

    def is_available(self) -> bool:
        """Check if Gemini service is available"""
        return bool(self.api_key)
