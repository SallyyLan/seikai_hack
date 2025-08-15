import os
import aiofiles
from typing import Dict, Any, Optional
from fastapi import UploadFile
import uuid
from datetime import datetime

class FileProcessor:
    def __init__(self):
        self.upload_dir = "uploads"
        self.processed_dir = "processed"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure upload and processed directories exist"""
        for directory in [self.upload_dir, self.processed_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    async def process_file(self, file: UploadFile, file_type: str) -> Dict[str, Any]:
        """Process uploaded file and extract relevant information"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = self._get_file_extension(file.filename)
            filename = f"{file_id}{file_extension}"
            
            # Save file
            file_path = os.path.join(self.upload_dir, filename)
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process based on file type
            processed_info = await self._extract_file_info(file_path, file_type, content)
            
            # Store processed information
            processed_path = os.path.join(self.processed_dir, f"{file_id}.json")
            async with aiofiles.open(processed_path, 'w') as f:
                await f.write(str(processed_info))
            
            return {
                "file_id": file_id,
                "original_filename": file.filename,
                "file_type": file_type,
                "file_path": file_path,
                "processed_path": processed_path,
                "content_summary": processed_info.get("summary", ""),
                "uploaded_at": datetime.utcnow().isoformat(),
                "file_size": len(content)
            }
            
        except Exception as e:
            raise Exception(f"File processing failed: {str(e)}")
    
    async def _extract_file_info(self, file_path: str, file_type: str, content: bytes) -> Dict[str, Any]:
        """Extract relevant information from different file types"""
        try:
            if file_type == "textbook":
                return await self._process_textbook(file_path, content)
            elif file_type == "slides":
                return await self._process_slides(file_path, content)
            elif file_type == "homework":
                return await self._process_homework(file_path, content)
            elif file_type == "past_exams":
                return await self._process_past_exams(file_path, content)
            elif file_type == "syllabus":
                return await self._process_syllabus(file_path, content)
            else:
                return await self._process_generic(file_path, content)
                
        except Exception as e:
            return {
                "error": f"Failed to process {file_type}: {str(e)}",
                "summary": f"File type: {file_type}, Size: {len(content)} bytes"
            }
    
    async def _process_textbook(self, file_path: str, content: bytes) -> Dict[str, Any]:
        """Process textbook files"""
        # In a real implementation, this would use OCR or PDF parsing
        # For now, return basic info
        return {
            "type": "textbook",
            "summary": f"Textbook content ({len(content)} bytes)",
            "chapters": [],
            "key_concepts": [],
            "difficulty_level": "intermediate"
        }
    
    async def _process_slides(self, file_path: str, content: bytes) -> Dict[str, Any]:
        """Process lecture slides"""
        return {
            "type": "slides",
            "summary": f"Lecture slides ({len(content)} bytes)",
            "topics": [],
            "key_points": [],
            "lecture_count": 0
        }
    
    async def _process_homework(self, file_path: str, content: bytes) -> Dict[str, Any]:
        """Process homework assignments"""
        return {
            "type": "homework",
            "summary": f"Homework assignment ({len(content)} bytes)",
            "problems": [],
            "solutions": [],
            "difficulty": "medium"
        }
    
    async def _process_past_exams(self, file_path: str, content: bytes) -> Dict[str, Any]:
        """Process past exam materials"""
        return {
            "type": "past_exams",
            "summary": f"Past exam materials ({len(content)} bytes)",
            "exam_count": 0,
            "topics_covered": [],
            "difficulty_trends": []
        }
    
    async def _process_syllabus(self, file_path: str, content: bytes) -> Dict[str, Any]:
        """Process course syllabus"""
        return {
            "type": "syllabus",
            "summary": f"Course syllabus ({len(content)} bytes)",
            "course_info": {},
            "learning_objectives": [],
            "assessment_methods": []
        }
    
    async def _process_generic(self, file_path: str, content: bytes) -> Dict[str, Any]:
        """Process generic files"""
        return {
            "type": "generic",
            "summary": f"Generic file ({len(content)} bytes)",
            "file_size": len(content),
            "extension": self._get_file_extension(file_path)
        }
    
    async def process_practice_work(self, file: UploadFile) -> Dict[str, Any]:
        """Process practice work files (PDF, images, etc.)"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = self._get_file_extension(file.filename)
            filename = f"{file_id}{file_extension}"
            
            # Save file
            file_path = os.path.join(self.upload_dir, filename)
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process based on content type
            processed_info = await self._extract_practice_work_info(file_path, file.content_type, content)
            
            # Store processed information
            processed_path = os.path.join(self.processed_dir, f"{file_id}.json")
            async with aiofiles.open(processed_path, 'w') as f:
                await f.write(str(processed_info))
            
            return {
                "file_id": file_id,
                "original_filename": file.filename,
                "file_type": "practice_work",
                "content_type": file.content_type,
                "file_path": file_path,
                "processed_path": processed_path,
                "content_summary": processed_info.get("summary", ""),
                "uploaded_at": datetime.utcnow().isoformat(),
                "file_size": len(content)
            }
            
        except Exception as e:
            raise Exception(f"Practice work processing failed: {str(e)}")
    
    async def _extract_practice_work_info(self, file_path: str, content_type: str, content: bytes) -> Dict[str, Any]:
        """Extract information from practice work files"""
        try:
            if content_type == "application/pdf":
                return await self._process_practice_pdf(file_path, content)
            elif content_type.startswith("image/"):
                return await self._process_practice_image(file_path, content, content_type)
            else:
                return await self._process_generic(file_path, content)
                
        except Exception as e:
            return {
                "error": f"Failed to process practice work: {str(e)}",
                "content_type": content_type,
                "summary": f"Practice work file ({len(content)} bytes)"
            }
    
    async def _process_practice_pdf(self, file_path: str, content: bytes) -> Dict[str, Any]:
        """Process practice work PDF files"""
        try:
            import PyPDF2
            import io
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            page_count = len(pdf_reader.pages)
            
            # Extract text from first few pages for summary
            sample_text = ""
            for i in range(min(3, page_count)):
                sample_text += pdf_reader.pages[i].extract_text()[:500] + " "
            
            return {
                "type": "practice_work_pdf",
                "summary": f"Practice work PDF with {page_count} pages",
                "page_count": page_count,
                "sample_text": sample_text.strip(),
                "file_size": len(content),
                "estimated_questions": max(1, page_count // 2)  # Rough estimate
            }
        except ImportError:
            return {
                "type": "practice_work_pdf",
                "summary": f"Practice work PDF ({len(content)} bytes) - PDF processing not available",
                "file_size": len(content),
                "note": "Install PyPDF2 for better PDF processing"
            }
        except Exception as e:
            return {
                "type": "practice_work_pdf",
                "summary": f"Practice work PDF ({len(content)} bytes)",
                "error": str(e),
                "file_size": len(content)
            }
    
    async def _process_practice_image(self, file_path: str, content: bytes, content_type: str) -> Dict[str, Any]:
        """Process practice work image files"""
        try:
            from PIL import Image
            import io
            
            # Open image to get metadata
            image = Image.open(io.BytesIO(content))
            width, height = image.size
            format_name = image.format
            mode = image.mode
            
            return {
                "type": "practice_work_image",
                "summary": f"Practice work image: {format_name} ({width}x{height})",
                "dimensions": {"width": width, "height": height},
                "format": format_name,
                "color_mode": mode,
                "file_size": len(content),
                "content_type": content_type,
                "estimated_questions": 1  # Single image typically contains one question
            }
        except ImportError:
            return {
                "type": "practice_work_image",
                "summary": f"Practice work image ({len(content)} bytes)",
                "content_type": content_type,
                "file_size": len(content),
                "note": "Install Pillow for better image processing"
            }
        except Exception as e:
            return {
                "type": "practice_work_image",
                "summary": f"Practice work image ({len(content)} bytes)",
                "content_type": content_type,
                "error": str(e),
                "file_size": len(content)
            }
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        if filename and '.' in filename:
            return '.' + filename.split('.')[-1]
        return ''
    
    async def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old uploaded files"""
        try:
            current_time = datetime.utcnow()
            for directory in [self.upload_dir, self.processed_dir]:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)
                        file_age = current_time - datetime.fromtimestamp(os.path.getctime(file_path))
                        
                        if file_age.total_seconds() > max_age_hours * 3600:
                            os.remove(file_path)
                            
        except Exception as e:
            print(f"Cleanup failed: {str(e)}")
    
    async def get_file_summary(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of processed file"""
        try:
            processed_path = os.path.join(self.processed_dir, f"{file_id}.json")
            if os.path.exists(processed_path):
                async with aiofiles.open(processed_path, 'r') as f:
                    content = await f.read()
                    return eval(content)  # In production, use proper JSON parsing
            return None
        except Exception:
            return None
