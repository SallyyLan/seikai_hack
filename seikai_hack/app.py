from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import json
import uuid
from datetime import datetime
import asyncio

# Import our dual-AI services
from services.gemini_service import GeminiService
from services.gpt_oss_service import GPTOSSService
from services.priority_queue import PriorityQueueService
from services.file_processor import FileProcessor

app = FastAPI(title="Exam Prep AI - Dual AI System", version="2.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize services
gemini_service = GeminiService()
gpt_oss_service = GPTOSSService()
priority_queue = PriorityQueueService()
file_processor = FileProcessor()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with dual-AI exam prep features"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_ai_status():
    """Check the availability of AI services"""
    return {
        "gemini": gemini_service.is_available(),
        "gpt_oss": gpt_oss_service.is_available(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload-practice-exam")
async def upload_practice_exam(
    exam_file: UploadFile = File(...),
    exam_type: str = Form("practice_exam")
):
    """
    Upload a practice exam PDF and extract LaTeX code using Gemini
    """
    try:
        # Validate file type
        if not exam_file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        print(f"Processing PDF: {exam_file.filename}, Size: {exam_file.size} bytes")
        
        # Extract LaTeX from PDF using Gemini
        print("Calling Gemini service for LaTeX extraction...")
        latex_result = await gemini_service.extract_latex_from_pdf(exam_file)
        print(f"Gemini result: {latex_result}")
        
        if not latex_result["success"]:
            raise HTTPException(status_code=500, detail=f"LaTeX extraction failed: {latex_result['error']}")
        
        # Validate LaTeX code using GPT-OSS
        print("Calling GPT-OSS service for LaTeX validation...")
        validation_result = await gpt_oss_service.validate_latex_code(latex_result["latex_code"])
        print(f"Validation result: {validation_result}")
        
        # Classify topics from LaTeX using GPT-OSS
        print("Calling GPT-OSS service for topic classification...")
        topic_result = await gpt_oss_service.classify_topics_from_latex(latex_result["latex_code"])
        print(f"Topic result: {topic_result}")
        
        # Save to processed directory
        session_id = str(uuid.uuid4())
        result_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "file_name": exam_file.filename,
            "latex_code": latex_result["latex_code"],
            "validation": validation_result,
            "topics": topic_result,
            "original_text": latex_result["original_text"],
            "page_count": latex_result["page_count"]
        }
        
        print(f"Saving result data for session: {session_id}")
        await file_processor.save_processed_result(session_id, result_data)
        
        return {
            "success": True,
            "session_id": session_id,
            "latex_code": latex_result["latex_code"],
            "validation": validation_result,
            "topics": topic_result,
            "message": "Practice exam uploaded and analyzed successfully"
        }
        
    except Exception as e:
        print(f"Error in upload_practice_exam: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/upload-student-work")
async def upload_student_work(
    work_file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Upload student's handwritten work and extract LaTeX code using Gemini
    """
    try:
        # Validate file type
        if not work_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are supported")
        
        # Extract LaTeX from image using Gemini
        latex_result = await gemini_service.extract_latex_from_image(work_file)
        
        if not latex_result["success"]:
            raise HTTPException(status_code=500, detail=f"LaTeX extraction failed: {latex_result['error']}")
        
        # Validate LaTeX code using GPT-OSS
        validation_result = await gpt_oss_service.validate_latex_code(latex_result["latex_code"])
        
        # Save student work
        work_id = str(uuid.uuid4())
        work_data = {
            "work_id": work_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "file_name": work_file.filename,
            "latex_code": latex_result["latex_code"],
            "validation": validation_result,
            "file_type": latex_result["file_type"]
        }
        
        await file_processor.save_processed_result(work_id, work_data)
        
        return {
            "success": True,
            "work_id": work_id,
            "latex_code": latex_result["latex_code"],
            "validation": validation_result,
            "message": "Student work uploaded and analyzed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-practice-exam")
async def generate_practice_exam(
    session_id: str = Form(...),
    target_topics: str = Form(...),
    difficulty: str = Form("medium")
):
    """
    Generate a practice exam based on identified topics using GPT-OSS
    """
    try:
        # Parse target topics
        topics = json.loads(target_topics) if isinstance(target_topics, str) else target_topics
        
        # Get exam format from original practice exam
        session_data = await file_processor.load_processed_result(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Create exam format based on original
        exam_format = {
            "style": "practice_exam",
            "difficulty": difficulty,
            "question_count": len(topics) * 2,  # 2 questions per topic
            "time_limit": "60 minutes"
        }
        
        # Generate practice exam using GPT-OSS
        practice_exam = await gpt_oss_service.generate_practice_exam(topics, exam_format)
        
        return {
            "success": True,
            "practice_exam": practice_exam,
            "message": "Practice exam generated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-student-work")
async def analyze_student_work(
    student_latex: str = Form(...),
    correct_solution: str = Form(...)
):
    """
    Analyze student's work and provide feedback using GPT-OSS
    """
    try:
        # Analyze student work using GPT-OSS
        analysis_result = await gpt_oss_service.analyze_student_work(student_latex, correct_solution)
        
        return {
            "success": True,
            "analysis": analysis_result,
            "message": "Student work analyzed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session_data(session_id: str):
    """Get session data and analysis results"""
    try:
        session_data = await file_processor.load_processed_result(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session_data": session_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "gemini": gemini_service.is_available(),
            "gpt_oss": gpt_oss_service.is_available()
        }
    }

@app.get("/api/test-gemini")
async def test_gemini():
    """Test Gemini service with a simple prompt"""
    try:
        test_prompt = "Say 'Hello, Gemini is working!' in exactly 5 words."
        response = await gemini_service._generate_content(test_prompt)
        return {
            "success": True,
            "response": response,
            "message": "Gemini service is working correctly"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Gemini service test failed"
        }

@app.get("/api/test-gpt-oss")
async def test_gpt_oss():
    """Test GPT-OSS service with a simple prompt"""
    try:
        test_prompt = "Say 'Hello, GPT-OSS is working!' in exactly 5 words."
        response = await gpt_oss_service._generate_content(test_prompt)
        return {
            "success": True,
            "response": response,
            "message": "GPT-OSS service is working correctly"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "GPT-OSS service test failed"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
