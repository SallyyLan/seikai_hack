from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import os
import json
from typing import List, Optional
import asyncio
from datetime import datetime
import uuid

from models import ExamSession, Topic, Question, User
from services.mathpix_service import MathpixService
from services.gpt_service import GPTService
from services.priority_queue import PriorityQueueService
from services.file_processor import FileProcessor
from database import get_db, engine
import models

app = FastAPI(title="LAST MINUTE Exam Prep AI", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize services
mathpix_service = MathpixService()
gpt_service = GPTService()
priority_queue = PriorityQueueService()
file_processor = FileProcessor()

# Create database tables
models.Base.metadata.create_all(bind=engine)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main landing page - URGENT MODE interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start-exam")
async def start_exam(
    course_name: str = Form(...),
    exam_date: str = Form(...),
    db = Depends(get_db)
):
    """Start a new exam session"""
    session_id = str(uuid.uuid4())
    session = ExamSession(
        id=session_id,
        course_name=course_name,
        exam_date=exam_date,
        created_at=datetime.utcnow()
    )
    db.add(session)
    db.commit()
    return {"session_id": session_id, "message": "Exam session started!"}

@app.post("/upload-materials")
async def upload_materials(
    session_id: str = Form(...),
    textbook: Optional[UploadFile] = File(None),
    slides: Optional[UploadFile] = File(None),
    homework: Optional[UploadFile] = File(None),
    past_exams: Optional[UploadFile] = File(None),
    syllabus: Optional[UploadFile] = File(None),
    db = Depends(get_db)
):
    """Upload course materials for AI analysis"""
    try:
        # Process uploaded files
        materials = {}
        if textbook:
            materials["textbook"] = await file_processor.process_file(textbook, "textbook")
        if slides:
            materials["slides"] = await file_processor.process_file(slides, "slides")
        if homework:
            materials["homework"] = await file_processor.process_file(homework, "homework")
        if past_exams:
            materials["past_exams"] = await file_processor.process_file(past_exams, "past_exams")
        if syllabus:
            materials["syllabus"] = await file_processor.process_file(syllabus, "syllabus")
        
        # Store materials in database
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if session:
            session.materials = materials
            db.commit()
        
        return {"message": "Materials uploaded successfully!", "materials": list(materials.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload-practice-work")
async def upload_practice_work(
    session_id: str = Form(...),
    work_files: List[UploadFile] = File(...),
    db = Depends(get_db)
):
    """Upload practice work files (PDF, images, handwritten work) for AI analysis"""
    try:
        results = []
        for work_file in work_files:
            # Validate file type
            if not _is_valid_practice_file(work_file):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {work_file.content_type}. Supported: PDF, PNG, JPG, JPEG, GIF, BMP"
                )
            
            # Process file based on type
            if work_file.content_type == "application/pdf":
                # Handle PDF files
                extracted_text = await _process_pdf_file(work_file)
            else:
                # Handle image files (PNG, JPG, etc.)
                extracted_text = await mathpix_service.extract_text(work_file)
            
            # Analyze correctness using GPT
            analysis = await gpt_service.analyze_work(extracted_text)
            
            # Store question and analysis
            question = Question(
                session_id=session_id,
                extracted_text=extracted_text,
                is_correct=analysis["is_correct"],
                feedback=analysis["feedback"],
                topics=analysis["topics"],
                confidence=analysis["confidence"]
            )
            db.add(question)
            
            results.append({
                "question_id": question.id,
                "filename": work_file.filename,
                "file_type": work_file.content_type,
                "extracted_text": extracted_text,
                "analysis": analysis
            })
        
        db.commit()
        
        # Update priority queue based on results
        await priority_queue.update_priorities(session_id, results)
        
        return {"message": "Practice work analyzed!", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def _is_valid_practice_file(file: UploadFile) -> bool:
    """Check if uploaded file is a valid practice work format"""
    valid_types = [
        "application/pdf",  # PDF files
        "image/png",        # PNG images
        "image/jpeg",       # JPEG images
        "image/jpg",        # JPG images
        "image/gif",        # GIF images
        "image/bmp",        # BMP images
        "image/webp",       # WebP images
        "image/tiff",       # TIFF images
    ]
    return file.content_type in valid_types

async def _process_pdf_file(pdf_file: UploadFile) -> str:
    """Extract text from PDF files"""
    try:
        # For now, we'll use a simple approach
        # In production, you might want to use PyPDF2 or pdfplumber for better extraction
        import PyPDF2
        import io
        
        # Read PDF content
        pdf_content = await pdf_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        # Extract text from all pages
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() + "\n"
        
        return extracted_text.strip()
    except ImportError:
        # Fallback if PyPDF2 is not available
        return f"PDF file uploaded: {pdf_file.filename} (PDF processing not available)"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@app.get("/get-priority-queue/{session_id}")
async def get_priority_queue(session_id: str, db = Depends(get_db)):
    """Get prioritized topics for study focus"""
    try:
        priorities = await priority_queue.get_priorities(session_id)
        return {"priorities": priorities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get priorities: {str(e)}")

@app.get("/session/{session_id}")
async def get_session_summary(session_id: str, request: Request, db = Depends(get_db)):
    """Get comprehensive session summary"""
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    questions = db.query(Question).filter(Question.session_id == session_id).all()
    priorities = await priority_queue.get_priorities(session_id)
    
    return templates.TemplateResponse("session.html", {
        "request": request,
        "session": session,
        "questions": questions,
        "priorities": priorities
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
