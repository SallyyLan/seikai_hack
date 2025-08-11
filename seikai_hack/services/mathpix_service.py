import requests
import base64
import os
from typing import Optional
from fastapi import UploadFile
import json

class MathpixService:
    def __init__(self):
        self.app_id = os.getenv("MATHPIX_APP_ID")
        self.app_key = os.getenv("MATHPIX_APP_KEY")
        self.base_url = "https://api.mathpix.com/v3"
        
        if not self.app_id or not self.app_key:
            raise ValueError("Mathpix credentials not found in environment variables")
    
    async def extract_text(self, image_file: UploadFile) -> str:
        """Extract text from handwritten work image using Mathpix"""
        try:
            # Read image file
            image_content = await image_file.read()
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            
            # Prepare request payload
            payload = {
                "src": f"data:image/{image_file.content_type};base64,{image_base64}",
                "formats": ["text", "mathml"],
                "ocr_options": {
                    "math_inline_delimiters": ["$", "$"],
                    "math_display_delimiters": ["$$", "$$"],
                    "rm_spaces": True
                }
            }
            
            # Make request to Mathpix API
            headers = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.base_url}/text",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract the text content
                extracted_text = result.get("text", "")
                return extracted_text
            else:
                raise Exception(f"Mathpix API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Failed to extract text: {str(e)}")
    
    async def extract_math(self, image_file: UploadFile) -> dict:
        """Extract mathematical expressions from image"""
        try:
            # Read image file
            image_content = await image_file.read()
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            
            # Prepare request payload for math extraction
            payload = {
                "src": f"data:image/{image_file.content_type};base64,{image_base64}",
                "formats": ["mathml", "latex"],
                "ocr_options": {
                    "math_inline_delimiters": ["$", "$"],
                    "math_display_delimiters": ["$$", "$$"]
                }
            }
            
            headers = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.base_url}/text",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "latex": result.get("latex", ""),
                    "mathml": result.get("mathml", ""),
                    "text": result.get("text", "")
                }
            else:
                raise Exception(f"Mathpix API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Failed to extract math: {str(e)}")
