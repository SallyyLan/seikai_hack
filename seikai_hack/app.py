from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import uuid
import os
import json
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import database and models
from database import get_db, engine
from models import ExamSession, Topic, Question, User
import models

# Import services
from services.gpt_service import GPTService
from services.gemini_service import GeminiService
from services.priority_queue import PriorityQueueService
from services.file_processor import FileProcessor

app = FastAPI(title="LAST MINUTE Exam Prep AI", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize services
try:
    print("Initializing services...")
    
    # Initialize GPT service
    try:
        gpt_service = GPTService()
        print("✅ GPT service initialized")
    except Exception as e:
        print(f"⚠️  GPT service initialization failed: {str(e)}")
        gpt_service = None
    
    # Initialize Gemini service
    try:
        gemini_service = GeminiService()
        print("✅ Gemini service initialized")
    except Exception as e:
        print(f"⚠️  Gemini service initialization failed: {str(e)}")
        gemini_service = None
    
    # Initialize other services
    try:
        priority_queue = PriorityQueueService()
        print("✅ Priority queue service initialized")
    except Exception as e:
        print(f"⚠️  Priority queue service initialization failed: {str(e)}")
        priority_queue = None
    
    try:
        file_processor = FileProcessor()
        print("✅ File processor service initialized")
    except Exception as e:
        print(f"⚠️  File processor service initialization failed: {str(e)}")
        file_processor = None
    
    # Check which services are available
    available_services = []
    if gpt_service: available_services.append("GPT")
    if gemini_service: available_services.append("Gemini")
    if priority_queue: available_services.append("Priority Queue")
    if file_processor: available_services.append("File Processor")
    
    if available_services:
        print(f"✅ Services available: {', '.join(available_services)}")
    else:
        print("❌ No services available - check your API keys in .env file")
        
except Exception as e:
    print(f"❌ Service initialization failed: {str(e)}")
    gpt_service = None
    gemini_service = None
    priority_queue = None
    file_processor = None

# Create database tables
try:
    print("Creating database tables...")
    # Check if database file exists and has old schema
    import os
    db_file = "exam_prep.db"
    
    if os.path.exists(db_file):
        print(f"Existing database file found: {db_file}")
        # Check if the table has the new columns by trying to query them
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                # Try to query the new columns to see if they exist
                result = conn.execute(text("SELECT exam_coverage, practice_exam, minimum_score, maximum_score FROM exam_sessions LIMIT 1"))
                print("✅ Database schema is up to date!")
        except Exception as e:
            print(f"Schema is outdated, recreating database: {str(e)}")
            # Remove old database and recreate
            os.remove(db_file)
            print("Old database removed, creating new one...")
            models.Base.metadata.create_all(bind=engine)
            print("✅ New database created successfully!")
    else:
        # Create new database
        models.Base.metadata.create_all(bind=engine)
        print("✅ New database created successfully!")
        
except Exception as e:
    print(f"Error creating database tables: {str(e)}")
    # Try to drop and recreate tables
    try:
        print("Attempting to recreate database tables...")
        models.Base.metadata.drop_all(bind=engine)
        models.Base.metadata.create_all(bind=engine)
        print("✅ Database tables recreated successfully!")
    except Exception as e2:
        print(f"Failed to recreate database tables: {str(e2)}")
        raise e2

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main landing page - URGENT MODE interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        from database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return {
            "status": "healthy",
            "database": "connected",
            "services": {
                "gpt": gpt_service is not None,
                "gemini": gemini_service is not None,
                "priority_queue": priority_queue is not None,
                "file_processor": file_processor is not None
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/reset-database")
async def reset_database():
    """Reset database - removes all data and recreates tables"""
    try:
        import os
        db_file = "exam_prep.db"
        
        if os.path.exists(db_file):
            os.remove(db_file)
            print("Old database removed")
        
        # Recreate tables
        models.Base.metadata.create_all(bind=engine)
        print("New database created")
        
        return {"message": "Database reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")

@app.post("/start-exam")
async def start_exam(
    course_name: str = Form(...),
    db = Depends(get_db)
):
    """Start a new exam session"""
    try:
        session_id = str(uuid.uuid4())
        session = ExamSession(
            id=session_id,
            course_name=course_name,
            created_at=datetime.now(timezone.utc)
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return {"session_id": session_id, "message": "Exam session started!"}
    except Exception as e:
        # Log the error for debugging
        error_msg = str(e)
        print(f"Error starting exam session: {error_msg}")
        
        # Check if it's a schema issue
        if "no column named" in error_msg.lower():
            print("Database schema issue detected. Please reset the database.")
            raise HTTPException(
                status_code=500, 
                detail="Database schema is outdated. Please contact support or reset the database."
            )
        
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to start exam session: {error_msg}")

@app.post("/upload-exam-coverage")
async def upload_exam_coverage(
    session_id: str = Form(...),
    exam_coverage_text: str = Form(...),
    db = Depends(get_db)
):
    """Process exam coverage text and extract topics"""
    try:
        if gemini_service:
            # Use Gemini service if available
            coverage_data = await gemini_service.process_exam_coverage_text(exam_coverage_text)
        else:
            # Fallback: extract topics manually from text
            print("⚠️  Gemini service not available, using fallback topic extraction")
            coverage_data = await _extract_topics_fallback(exam_coverage_text)
        
        # Update session with exam coverage
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if session:
            session.exam_coverage = coverage_data
            db.commit()
        
        return {
            "message": "Exam coverage processed successfully!",
            "topics": coverage_data["topics"],
            "topics_count": len(coverage_data["topics"])
        }
    except Exception as e:
        print(f"Error processing exam coverage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process exam coverage: {str(e)}")

async def _extract_topics_fallback(text_content: str) -> Dict[str, Any]:
    """Fallback method to extract topics when Gemini service is not available"""
    try:
        # Simple text parsing to extract topics
        lines = text_content.strip().split('\n')
        topics = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove common bullet points and numbering
                line = line.lstrip('•').lstrip('*').lstrip('-').lstrip('0123456789.').strip()
                if line and len(line) > 2:  # Only add non-empty lines with meaningful content
                    topics.append(line)
        
        # If no topics found, create a default one
        if not topics:
            topics = ["General Topics"]
        
        return {
            "topics": topics,
            "raw_text": text_content,
            "processed": True,
            "method": "fallback"
        }
    except Exception as e:
        # Ultimate fallback
        return {
            "topics": ["General Topics"],
            "raw_text": text_content,
            "processed": True,
            "method": "fallback_error"
        }

@app.post("/upload-practice-exam")
async def upload_practice_exam(
    session_id: str = Form(...),
    practice_exam_file: UploadFile = File(...),
    minimum_score: float = Form(...),
    db = Depends(get_db)
):
    """Upload practice exam and set minimum score requirement"""
    try:
        if not gemini_service:
            raise HTTPException(status_code=500, detail="Gemini service not available")
            
        # Validate file type
        if not _is_valid_document_file(practice_exam_file):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {practice_exam_file.content_type}. Supported: PDF, PNG, JPG, JPEG"
            )
        
        # Get exam coverage from session
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if not session or not session.exam_coverage:
            raise HTTPException(status_code=400, detail="Please upload exam coverage first")
        
        exam_topics = session.exam_coverage.get("topics", [])
        
        # Read file content
        file_content = await practice_exam_file.read()
        
        # Process with Gemini to extract questions and scores
        exam_data = await gemini_service.process_practice_exam(
            file_content,
            practice_exam_file.content_type,
            exam_topics
        )
        
        # Update session with practice exam data and minimum score
        session.practice_exam = exam_data
        session.minimum_score = minimum_score
        session.maximum_score = exam_data.get("total_score", 0.0)
        db.commit()
        
        return {
            "message": "Practice exam processed successfully!",
            "questions_count": len(exam_data["questions"]),
            "total_score": exam_data["total_score"],
            "minimum_score": minimum_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process practice exam: {str(e)}")

@app.post("/set-confidence-scores")
async def set_confidence_scores(
    session_id: str = Form(...),
    confidence_scores: str = Form(...),
    db = Depends(get_db)
):
    """Set confidence scores for topics"""
    try:
        # Parse confidence scores JSON
        topic_confidence = json.loads(confidence_scores)
        
        # Get the session
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Store confidence scores in session
        if not session.materials:
            session.materials = {}
        session.materials["topic_confidence"] = topic_confidence
        db.commit()
        
        return {
            "message": "Confidence scores saved successfully!",
            "topics_count": len(topic_confidence)
        }
        
    except Exception as e:
        print(f"Error setting confidence scores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save confidence scores: {str(e)}")

@app.post("/calculate-priorities")
async def calculate_priorities(
    session_id: str = Form(...),
    db = Depends(get_db)
):
    """Calculate study priorities based on topics and confidence scores"""
    try:
        print(f"🔍 Starting priority calculation for session: {session_id}")
        
        # Check if priority queue service is available
        if not priority_queue:
            print("❌ Priority queue service is None")
            raise HTTPException(status_code=500, detail="Priority queue service not available")
        
        print("✅ Priority queue service is available")
        
        # Get the session
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if not session:
            print("❌ Session not found")
            raise HTTPException(status_code=404, detail="Session not found")
        
        print(f"✅ Session found: {session.course_name}")
        
        # Get exam topics and confidence scores
        exam_topics = session.exam_coverage.get("topics", [])
        topic_confidence = session.materials.get("topic_confidence", {})
        practice_exam_data = session.practice_exam
        
        print(f"📋 Exam topics: {exam_topics}")
        print(f"🎯 Topic confidence: {topic_confidence}")
        print(f"📝 Practice exam data keys: {list(practice_exam_data.keys()) if practice_exam_data else 'None'}")
        
        if not exam_topics:
            print("❌ No exam topics found")
            raise HTTPException(status_code=400, detail="No exam topics found")
        
        if not topic_confidence:
            print("❌ No confidence scores found")
            raise HTTPException(status_code=400, detail="No confidence scores found")
        
        if not practice_exam_data:
            print("❌ No practice exam data found")
            raise HTTPException(status_code=400, detail="No practice exam data found")
        
        # Validate practice exam data structure
        if "questions" not in practice_exam_data:
            print("❌ Practice exam data missing 'questions' key")
            raise HTTPException(status_code=400, detail="Practice exam data is incomplete")
        
        if "total_score" not in practice_exam_data:
            print("❌ Practice exam data missing 'total_score' key")
            raise HTTPException(status_code=400, detail="Practice exam data is incomplete")
        
        # Get minimum score from session
        minimum_score = session.minimum_score or 0.0
        print(f"🎯 Minimum score target: {minimum_score}")
        print(f"📊 Practice exam total score: {practice_exam_data.get('total_score', 'Unknown')}")
        
        print("✅ All required data found, calculating priorities...")
        
        # Calculate priorities using topic-based approach
        priorities = await priority_queue.calculate_topic_based_priorities(
            session_id, exam_topics, topic_confidence, practice_exam_data
        )
        
        print(f"✅ Priorities calculated: {len(priorities)} topics")
        
        # Store priorities in session
        if not session.materials:
            session.materials = {}
        session.materials["study_priorities"] = priorities
        db.commit()
        
        print("✅ Priorities stored in database")
        
        return {
            "message": "Study priorities calculated successfully!",
            "priorities": priorities,
            "topics_count": len(priorities)
        }
        
    except Exception as e:
        print(f"❌ Error calculating priorities: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to calculate priorities: {str(e)}")

@app.post("/generate-practice-exam")
async def generate_practice_exam(
    session_id: str = Form(...),
    difficulty_level: str = Form(default="medium"),
    db = Depends(get_db)
):
    """Generate a new practice exam based on prioritized topics"""
    try:
        print(f"🎯 Starting practice exam generation for session: {session_id}")
        
        # Check if GPT service is available
        if not gpt_service:
            print("❌ GPT service not available")
            raise HTTPException(
                status_code=500, 
                detail="GPT service not available. Please check your API keys (HF_TOKEN and GPT_OSS_MODEL in .env file)."
            )
        
        # Get session and prioritized topics
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if not session:
            print("❌ Session not found")
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get prioritized topics from session materials
        priorities = session.materials.get("study_priorities", [])
        if not priorities:
            print("❌ No prioritized topics found")
            raise HTTPException(
                status_code=400, 
                detail="No prioritized topics found. Please complete the confidence scoring step first."
            )
        
        # Get exam topics and course info
        exam_topics = session.exam_coverage.get("topics", [])
        course_name = session.course_name
        
        print(f"🎯 Generating practice exam for course: {course_name}")
        print(f"📋 Topics to focus on: {[p['name'] for p in priorities[:5]]}")
        print(f"🎲 Difficulty level: {difficulty_level}")
        
        # Generate practice exam using GPT
        try:
            print("🤖 Calling GPT service to generate practice exam...")
            latex_code = await gpt_service.generate_practice_exam(
                priorities,
                course_name,
                difficulty_level
            )
            print("✅ GPT service returned LaTeX code")
            
        except Exception as gpt_error:
            print(f"❌ GPT service failed: {str(gpt_error)}")
            
            # Check if it's an authentication error
            if "401" in str(gpt_error) or "Invalid credentials" in str(gpt_error):
                print("🔑 GPT authentication failed - using fallback generator")
                latex_code = generate_fallback_practice_exam(priorities, course_name, difficulty_level)
            else:
                print("⚠️ GPT service error - using fallback generator")
                latex_code = generate_fallback_practice_exam(priorities, course_name, difficulty_level)
        
        # Validate LaTeX code
        if not latex_code or len(latex_code.strip()) < 100:
            print("❌ Generated LaTeX code is too short or empty")
            raise HTTPException(
                status_code=500, 
                detail="Generated LaTeX code is invalid. Please try again or check your API configuration."
            )
        
        print(f"✅ LaTeX code generated successfully, length: {len(latex_code)} characters")
        
        # Create temporary file with LaTeX content
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as f:
                f.write(latex_code)
                temp_file_path = f.name
            
            print(f"✅ Temporary file created: {temp_file_path}")
            
        except Exception as file_error:
            print(f"❌ Failed to create temporary file: {str(file_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create practice exam file: {str(file_error)}"
            )
        
        print(f"✅ Practice exam generated successfully: {temp_file_path}")
        
        # Return the LaTeX file for download
        return FileResponse(
            temp_file_path,
            media_type='application/x-tex',
            filename=f"{course_name}_Practice_Exam.tex"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"❌ Failed to generate practice exam: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate practice exam: {str(e)}"
        )

def generate_fallback_practice_exam(priorities: List[Dict[str, Any]], course_name: str, difficulty_level: str) -> str:
    """Generate a basic practice exam in LaTeX format when GPT service is unavailable"""
    try:
        print("📝 Generating fallback practice exam...")
        
        # Create topic list for the exam
        topic_sections = []
        for i, topic in enumerate(priorities[:8], 1):  # Limit to 8 topics
            topic_name = topic['name']
            score_value = topic.get('score_value', 10)
            priority = topic.get('study_priority', i)
            
            topic_sections.append(f"""
\\section*{{{topic_name}}}
\\textbf{{Points: {score_value}}} \\hfill \\textbf{{Priority: {priority}}}

\\begin{{enumerate}}
\\item (5 points) Define and explain the key concepts of {topic_name.lower()}.
\\item (5 points) Provide an example or application of {topic_name.lower()}.
\\end{{enumerate}}
""")
        
        # Generate the LaTeX document
        latex_code = f"""\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}

\\title{{{course_name} Practice Exam}}
\\subtitle{{Generated Practice Exam - {difficulty_level.title()} Difficulty}}
\\author{{AI Study Assistant}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{center}}
\\textbf{{Instructions:}} This practice exam covers the topics you need to study most. \\
Focus on higher priority topics for maximum effectiveness.
\\end{{center}}

\\vspace{{0.5cm}}

\\begin{{center}}
\\textbf{{Total Points: 100}} \\hfill \\textbf{{Time: 60 minutes}}
\\end{{center}}

\\vspace{{0.5cm}}

{chr(10).join(topic_sections)}

\\section*{{General Problem Solving}}
\\textbf{{Points: 20}}

\\begin{{enumerate}}
\\item (10 points) Choose one topic from above and explain how it relates to other concepts in {course_name.lower()}.
\\item (10 points) Describe a real-world application where the concepts from this course would be useful.
\\end{{enumerate}}

\\vspace{{1cm}}

\\begin{{center}}
\\textbf{{Good luck with your studies!}}
\\end{{center}}

\\end{{document}}"""
        
        print("✅ Fallback practice exam generated successfully")
        return latex_code
        
    except Exception as e:
        print(f"❌ Fallback exam generation failed: {str(e)}")
        # Return a minimal LaTeX document as ultimate fallback
        return f"""\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\title{{{course_name} Practice Exam}}
\\author{{Study Assistant}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle
\\section*{{Practice Questions}}
\\begin{{enumerate}}
\\item Study the topics: {', '.join([p['name'] for p in priorities[:5]])}
\\item Focus on your confidence levels and priority scores
\\item Review course materials and practice problems
\\end{{enumerate}}
\\end{{document}}"""

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
        print(f"🔍 Starting practice work analysis for session: {session_id}")
        print(f"📁 Files to process: {len(work_files)}")
        
        # Check if required services are available
        if not gemini_service:
            raise HTTPException(status_code=500, detail="Gemini service not available")
        
        if not gpt_service:
            print("⚠️ GPT service not available - will use fallback analysis")
            gpt_available = False
        else:
            gpt_available = True
        
        if not priority_queue:
            raise HTTPException(status_code=500, detail="Priority queue service not available")
        
        print("✅ All required services are available")
        print(f"🤖 GPT service: {'Available' if gpt_available else 'Not available'}")
        
        results = []
        for i, work_file in enumerate(work_files):
            print(f"📄 Processing file {i+1}/{len(work_files)}: {work_file.filename}")
            
            # Validate file type
            if not _is_valid_practice_file(work_file):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {work_file.content_type}. Supported: PDF, PNG, JPG, JPEG, GIF, BMP"
                )
            
            # Process file based on type
            try:
                if work_file.content_type == "application/pdf":
                    print(f"📄 Processing PDF file: {work_file.filename}")
                    file_content = await work_file.read()
                    extracted_text = await gemini_service.extract_text(file_content, work_file.content_type)
                else:
                    print(f"🖼️ Processing image file: {work_file.filename}")
                    file_content = await work_file.read()
                    extracted_text = await gemini_service.extract_text(file_content, work_file.content_type)
                
                print(f"✅ Text extracted successfully, length: {len(extracted_text)}")
                
            except Exception as e:
                print(f"❌ Text extraction failed for {work_file.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
            
            # Analyze correctness using GPT
            if gpt_available:
                try:
                    print(f"🤖 Analyzing work with GPT...")
                    
                    # Get course context from session
                    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
                    course_context = session.course_name if session else "General Course"
                    
                    analysis = await gpt_service.analyze_work(extracted_text, course_context)
                    print(f"✅ GPT analysis completed")
                    
                except Exception as e:
                    print(f"❌ GPT analysis failed: {str(e)}")
                    
                    # Check if it's an authentication error
                    if "401" in str(e) or "Invalid credentials" in str(e):
                        print("🔑 GPT authentication failed - using fallback analysis")
                        analysis = {
                            "is_correct": True,
                            "feedback": "GPT analysis unavailable due to authentication error. Please check your HF_TOKEN in .env file.",
                            "topics": [],
                            "confidence": 0.5,
                            "suggestions": ["Check your work carefully", "Verify your Hugging Face API key"]
                        }
                    else:
                        # Create a general fallback analysis
                        analysis = {
                            "is_correct": True,
                            "feedback": "Analysis unavailable - check GPT service",
                            "topics": [],
                            "confidence": 0.5,
                            "suggestions": ["Check your work carefully"]
                        }
            else:
                print("⚠️ Using fallback analysis (GPT service not available)")
                analysis = {
                    "is_correct": True,
                    "feedback": "GPT analysis not available. Please check your HF_TOKEN in .env file for full analysis.",
                    "topics": [],
                    "confidence": 0.5,
                    "suggestions": ["Check your work carefully", "Set up Hugging Face API key for AI feedback"]
                }
            
            # Store question and analysis
            try:
                # Generate a unique ID for the question
                import uuid
                question_id = str(uuid.uuid4())
                
                question = Question(
                    id=question_id,
                    session_id=session_id,
                    extracted_text=extracted_text,
                    is_correct=analysis.get("is_correct", True),
                    feedback=analysis.get("feedback", "No feedback available"),
                    topics=analysis.get("topics", []),
                    confidence=analysis.get("confidence", 0.5)
                )
                db.add(question)
                print(f"✅ Question stored in database with ID: {question_id}")
                
            except Exception as e:
                print(f"❌ Failed to store question: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to store question: {str(e)}")
            
            results.append({
                "question_id": question.id,
                "filename": work_file.filename,
                "file_type": work_file.content_type,
                "extracted_text": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                "analysis": analysis
            })
        
        print(f"✅ All files processed, committing to database...")
        db.commit()
        print(f"✅ Database commit successful")
        
        # Update priority queue based on results
        try:
            print(f"🔄 Updating priority queue...")
            await priority_queue.update_priorities(session_id, results)
            print(f"✅ Priority queue updated")
        except Exception as e:
            print(f"⚠️ Priority queue update failed: {str(e)}")
            # Don't fail the entire request if priority queue update fails
        
        print(f"🎯 Practice work analysis completed successfully!")
        return {"message": "Practice work analyzed!", "results": results}
        
    except Exception as e:
        print(f"❌ Practice work analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def _is_valid_document_file(file: UploadFile) -> bool:
    """Check if uploaded file is a valid document format"""
    valid_types = [
        "application/pdf",  # PDF files
        "image/png",        # PNG images
        "image/jpeg",       # JPEG images
        "image/jpg",        # JPG images
    ]
    return file.content_type in valid_types

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

@app.get("/test-priority-queue")
async def test_priority_queue():
    """Test if priority queue service is working"""
    try:
        if not priority_queue:
            return {
                "status": "error",
                "message": "Priority queue service is None",
                "service_type": str(type(priority_queue)) if priority_queue else "None"
            }
        
        # Test basic functionality
        test_result = {
            "status": "success",
            "service_type": str(type(priority_queue)),
            "service_available": priority_queue is not None,
            "methods": [method for method in dir(priority_queue) if not method.startswith('_')]
        }
        
        return test_result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Priority queue test failed: {str(e)}",
            "service_type": str(type(priority_queue)) if priority_queue else "None"
        }

@app.get("/test-practice-work")
async def test_practice_work():
    """Test if practice work services are working"""
    try:
        test_results = {
            "gemini_service": gemini_service is not None,
            "gpt_service": gpt_service is not None,
            "priority_queue": priority_queue is not None,
            "services_available": []
        }
        
        if gemini_service:
            test_results["services_available"].append("Gemini")
        if gpt_service:
            test_results["services_available"].append("GPT")
        if priority_queue:
            test_results["services_available"].append("Priority Queue")
        
        return {
            "status": "success",
            "test_results": test_results,
            "message": f"Available services: {', '.join(test_results['services_available'])}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Test failed: {str(e)}"
        }

@app.get("/session/{session_id}")
async def get_session(session_id: str, db = Depends(get_db)):
    """Get session data including study priorities"""
    try:
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session.id,
            "course_name": session.course_name,
            "exam_coverage": session.exam_coverage,
            "practice_exam": session.practice_exam,
            "minimum_score": session.minimum_score,
            "maximum_score": session.maximum_score,
            "materials": session.materials
        }
        
    except Exception as e:
        print(f"Error getting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

# Removed old session summary endpoint - now using modal-based full report

@app.get("/services-status")
async def get_services_status():
    """Get the status of all services"""
    return {
        "services": {
            "gpt": gpt_service is not None,
            "gemini": gemini_service is not None,
            "priority_queue": priority_queue is not None,
            "file_processor": file_processor is not None
        },
        "recommendations": {
            "gpt": "Set GPT_OSS_MODEL and HF_TOKEN in .env" if gpt_service is None else "✅ Available",
            "gemini": "Set GEMINI_API_KEY in .env" if gemini_service is None else "✅ Available",
            "priority_queue": "✅ Available (no API key needed)",
            "file_processor": "✅ Available (no API key needed)"
        }
    }

@app.get("/test-gemini")
async def test_gemini():
    """Test Gemini service functionality"""
    try:
        if not gemini_service:
            return {
                "status": "error",
                "message": "Gemini service not available",
                "recommendation": "Check your GEMINI_API_KEY in .env file"
            }
        
        # Test the service
        test_result = await gemini_service.test_model_availability()
        
        return {
            "status": "success",
            "gemini_test": test_result,
            "message": "Gemini service test completed"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Gemini test failed: {str(e)}",
            "recommendation": "Check your API key and internet connection"
        }

@app.get("/list-gemini-models")
async def list_gemini_models():
    """List available Gemini models"""
    try:
        if not gemini_service:
            return {
                "status": "error",
                "message": "Gemini service not available"
            }
        
        import google.generativeai as genai
        models = genai.list_models()
        
        model_list = []
        for model in models:
            model_list.append({
                "name": model.name,
                "display_name": model.display_name,
                "description": model.description,
                "generation_methods": model.generation_methods
            })
        
        return {
            "status": "success",
            "models": model_list,
            "count": len(model_list)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list models: {str(e)}"
        }

@app.get("/debug-session/{session_id}")
async def debug_session(session_id: str, db = Depends(get_db)):
    """Debug endpoint to inspect session data"""
    try:
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if not session:
            return {"status": "error", "message": "Session not found"}
        
        # Convert session to dict for inspection
        session_data = {
            "id": session.id,
            "course_name": session.course_name,
            "exam_coverage": session.exam_coverage,
            "practice_exam": session.practice_exam,
            "minimum_score": session.minimum_score,
            "maximum_score": session.maximum_score,
            "materials": session.materials
        }
        
        return {
            "status": "success",
            "session_data": session_data,
            "data_types": {
                "exam_coverage_type": str(type(session.exam_coverage)),
                "practice_exam_type": str(type(session.practice_exam)),
                "materials_type": str(type(session.materials))
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Debug failed: {str(e)}",
            "traceback": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
