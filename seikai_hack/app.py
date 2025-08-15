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
        
        print(f"🔍 Session found: {session.course_name}")
        print(f"🔍 Session materials: {session.materials}")
        print(f"🔍 Session exam_coverage: {session.exam_coverage}")
        
        # Get prioritized topics from session materials
        priorities = session.materials.get("study_priorities", [])
        print(f"🔍 Retrieved priorities: {len(priorities) if priorities else 0} topics")
        
        if not priorities:
            print("❌ No prioritized topics found in session.materials")
            print(f"🔍 Available materials keys: {list(session.materials.keys()) if session.materials else 'None'}")
            
            # Check if we can calculate priorities on the fly
            if session.exam_coverage and session.practice_exam:
                print("🔄 Attempting to calculate priorities on the fly...")
                try:
                    exam_topics = session.exam_coverage.get("topics", [])
                    practice_exam_data = session.practice_exam
                    
                    # Get topic confidence from session (default to 1 for all topics)
                    topic_confidence = {}
                    for topic in exam_topics:
                        topic_confidence[topic] = 1  # Default confidence
                    
                    print(f"🔄 Calculating priorities for {len(exam_topics)} topics...")
                    priorities = await priority_queue.calculate_topic_based_priorities(
                        session_id, exam_topics, topic_confidence, practice_exam_data
                    )
                    
                    # Store the calculated priorities
                    if not session.materials:
                        session.materials = {}
                    session.materials["study_priorities"] = priorities
                    db.commit()
                    print(f"✅ Priorities calculated and stored: {len(priorities)} topics")
                    
                except Exception as calc_error:
                    print(f"❌ Failed to calculate priorities on the fly: {str(calc_error)}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"No prioritized topics found and failed to calculate them. Please complete the confidence scoring step first. Error: {str(calc_error)}"
                    )
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="No prioritized topics found. Please complete the confidence scoring step first. Required: exam coverage and practice exam data."
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
        
        # Generate solutions for the exam
        print("📝 Generating solutions for the practice exam...")
        solutions_code = generate_exam_solutions(priorities, course_name, difficulty_level)
        print(f"✅ Solutions generated successfully, length: {len(solutions_code)} characters")
        
        # Create temporary files with LaTeX content
        try:
            import zipfile
            import tempfile
            
            # Create exam file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as f:
                f.write(latex_code)
                exam_file_path = f.name
            
            # Create solutions file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as f:
                f.write(solutions_code)
                solutions_file_path = f.name
            
            # Create zip file containing both
            zip_file_path = tempfile.mktemp(suffix='.zip')
            with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                zipf.write(exam_file_path, f"{course_name}_Practice_Exam.tex")
                zipf.write(solutions_file_path, f"{course_name}_Practice_Exam_Solutions.tex")
            
            print(f"✅ Files created: exam={exam_file_path}, solutions={solutions_file_path}, zip={zip_file_path}")
            
        except Exception as file_error:
            print(f"❌ Failed to create files: {str(file_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create practice exam files: {str(file_error)}"
            )
        
        print(f"✅ Practice exam and solutions generated successfully")
        
        # Return the zip file for download
        return FileResponse(
            zip_file_path,
            media_type='application/zip',
            filename=f"{course_name}_Practice_Exam_With_Solutions.zip"
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

def generate_exam_solutions(priorities: List[Dict[str, Any]], course_name: str, difficulty_level: str) -> str:
    """Generate comprehensive solutions for the practice exam"""
    try:
        print("📝 Generating comprehensive exam solutions...")
        
        # Create solution sections for each topic
        solution_sections = []
        for i, topic in enumerate(priorities[:8], 1):  # Limit to 8 topics
            topic_name = topic['name']
            score_value = topic.get('score_value', 10)
            priority = topic.get('study_priority', i)
            
            # Generate different solution types based on topic
            if "big-o" in topic_name.lower() or "omega" in topic_name.lower() or "theta" in topic_name.lower():
                # Big-O notation solutions
                solution_sections.append(f"""
% =========================
% {i}) Big-O / Omega / Theta
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{$O$, $\\Omega$, and $\\Theta$}}
\\medskip

\\noindent For each statement, mark \\textbf{{Always}}, \\textbf{{Sometimes}}, or \\textbf{{Never}}.

\\begin{{enumerate}}[label=\\alph*)]
\\item A function that is $\\Theta(n^2 + n)$ is \\rule{{1.2in}}{{0.4pt}} $\\Theta(n^2)$.\\\\
\\(\\boxed{{\\textbf{{Always}}}}\\)

\\item A function that is $O(n\\log n)$ is \\rule{{1.2in}}{{0.4pt}} $O(n)$.\\\\
\\(\\boxed{{\\textbf{{Never}}}}\\)

\\item A function that is $O(\\log^{{30}} n)$ is \\rule{{1.2in}}{{0.4pt}} $O(n)$.\\\\
\\(\\boxed{{\\textbf{{Always}}}}\\)

\\item A function that is $\\Omega(n^2)$ is \\rule{{1.2in}}{{0.4pt}} $O(n^{{1.5}})$.\\\\
\\(\\boxed{{\\textbf{{Never}}}}\\)

\\item A function that is $\\Theta(2^n)$ is \\rule{{1.2in}}{{0.4pt}} $\\Omega(n^{{2025}})$.\\\\
\\(\\boxed{{\\textbf{{Always}}}}\\)

\\item If $f(n)\\in O(g(n))$ and $f(n)\\in \\Omega(h(n))$, then $h(n)$ is \\rule{{1.2in}}{{0.4pt}} $O(g(n))$.\\\\
\\(\\boxed{{\\textbf{{Always}}}}\\)

\\item If $f(n)\\in O(g(n))$, then $2^{{f(n)}}$ is \\rule{{1.2in}}{{0.4pt}} $O(2^{{g(n)}})$.\\\\
\\(\\boxed{{\\textbf{{Sometimes}}}}\\) \\textit{{(true if $f(n)\\le g(n)+O(1)$; false if $f(n)=c\\,g(n)$ with $c>1$)}}

\\item $f(n)+g(n)$ is \\rule{{1.2in}}{{0.4pt}} $\\Theta(\\max\\{{f(n),g(n)\\}})$.\\\\
\\(\\boxed{{\\textbf{{Always}}}}\\) \\textit{{(for eventually nonnegative $f,g$)}}
\\end{{enumerate}}

\\bigskip
""")
            elif "recurrence" in topic_name.lower():
                # Recurrence relations solutions
                solution_sections.append(f"""
% =========================
% {i}) Recurrences (Write + Tree)
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{ Write a Recurrence}}

\\medskip
\\noindent\\texttt{{public static int f(int n) \\{{}}\\\\
\\quad \\texttt{{if (n < 100) return 42;}}\\\\
\\quad \\texttt{{int m = 10 * n;}}\\\\
\\quad \\texttt{{return f(n/2) + f(m/100) + f(1);}} \\\\
\\texttt{{}}
\\medskip

\\noindent\\textbf{{Answer (do not solve):}}
\\[
\\boxed{{T(n)=c_0 \\text{{ for }} n<100\\quad\\text{{and}}\\quad T(n)=T(n/2)+T(n/10)+c_1 \\text{{ for }} n\\ge100.}}
\\]

\\bigskip
\\noindent\\textbf{{Tree Method}}

\\medskip
Given \\(T(0)=1,\\ T(N)=2T(N-2)+2^N\\) for even \\(N>0\\).

\\begin{{enumerate}}[label=\\alph*)]
\\item \\textit{{(Sketch)}} \\;[diagram placeholder]

\\item \\textbf{{Total work at level \\(i\\):}} \\(\\boxed{{2^{{N-i}}}}\\).

\\item \\textbf{{Base-case level:}} \\(\\boxed{{N/2}}\\).

\\item \\textbf{{Simplified bound:}} \\(\\boxed{{\\Theta(2^N)}}\\).
\\end{{enumerate}}

\\bigskip
""")
            elif "avl" in topic_name.lower() or "tree" in topic_name.lower():
                # AVL trees solutions
                solution_sections.append(f"""
% =========================
% {i}) AVL Trees
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{AVL}}

\\begin{{enumerate}}[label=\\alph*)]
\\item Insert \\(50,40,30,35,34,60,70,80\\). \\textit{{[Final tree drawing not included in uploaded solutions.]}}

\\item Then insert \\(90\\). \\textit{{[Final tree drawing not included in uploaded solutions.]}}

\\item The insertion from part (b) required: \\textit{{[Not marked in uploaded solutions.]}}
\\end{{enumerate}}

\\bigskip
""")
            elif "heap" in topic_name.lower():
                # Binary heaps solutions
                solution_sections.append(f"""
% =========================
% {i}) Binary Heaps
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Heaps}}

\\medskip
Array (0-indexed):
\\[
24\\ 26\\ 53\\ 84\\ 50\\ 63\\ 78\\ 99\\ 96\\ 71\\ 67
\\]

\\begin{{enumerate}}[label=\\alph*)]
\\item \\textbf{{Answer:}} This is a \\(\\boxed{{\\text{{min}}}}\\) heap.

\\item (Draw the heap.) \\textit{{[Diagram not provided in uploaded solutions.]}}

\\item \\textbf{{Insert}} \\texttt{{heap.insert(x)}} independently for each assertion:

\\begin{{enumerate}}[label=\\roman*)]
\\item Some node with value ``63'' has a child ``71'': \\(\\boxed{{x=71}}\\).

\\item No node with value ``78'' has a parent ``53'': \\(\\boxed{{x=25}}\\).

\\item Every node with value ``50'' has exactly one child: \\(\\boxed{{\\text{{Impossible.}}}}\\) \\\\
\\textit{{Reason: 50 already has two children; one insertion elsewhere will not change that.}}

\\item No leaf has a value of ``63'': \\(\\boxed{{x=100}}\\).
\\end{{enumerate}}
\\end{{enumerate}}

\\bigskip
""")
            elif "dictionary" in topic_name.lower() or "adt" in topic_name.lower() or "hash" in topic_name.lower():
                # Dictionary ADT solutions
                solution_sections.append(f"""
% =========================
% {i}) Dictionary ADT (via Hashing)
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Hashing}}

\\begin{{enumerate}}[label=\\alph*)]
\\item Linear probing with \\(h(x)=x\\bmod 10\\) inserting \\(23,5,3,66,43,19,79\\):

\\medskip
\\begin{{tabular}}{{|c|c|c|c|c|c|c|c|c|c|c|}}
\\hline
Index & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9\\\\
\\hline
Value & \\(\\boxed{{79}}\\) &  &  & \\(\\boxed{{23}}\\) & \\(\\boxed{{3}}\\) & \\(\\boxed{{5}}\\) & \\(\\boxed{{66}}\\) & \\(\\boxed{{43}}\\) &  & \\(\\boxed{{19}}\\)\\\\
\\hline
\\end{{tabular}}

\\medskip

\\item Quadratic probing table, \\(p\\) prime, \\(h(x)=0\\), load $<0.5$. Best-case runtime to insert $n$ elements:

\\medskip
\\noindent \\(\\boxed{{O(n^2)}}\\)

\\item Separate chaining, initial table size \\(12\\), \\(h(x)=4x\\), resizing doubles size, $n$ multiple of $p$. Minimum possible longest-chain length \\(L\\):

\\medskip
\\noindent \\(\\boxed{{L = 4n/p}}\\) \\textit{{[This is the minimum possible value when the hash function distributes elements evenly.]}}
\\end{{enumerate}}

\\bigskip
""")
            elif "complexity" in topic_name.lower() or "analysis" in topic_name.lower():
                # Complexity analysis solutions
                solution_sections.append(f"""
% =========================
% {i}) Time/Space Complexity (Code Analysis)
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Code Analysis}}

\\medskip
Describe the worst-case running time in Big-$O$ of the following.

\\medskip
\\noindent\\textbf{{(a)}}
\\begin{{verbatim}}
public static void mystery1(int[] arr) {{
    if (arr.length < 10000) {{
        int cnt = 0;
        for(int i = 0; i < arr.length; i++) {{
            for(int j = 0; j < arr.length; j++) {{
                if (arr[i] < arr[j]) {{ cnt++; }}
            }}
            arr[i] = cnt;
        }}
    }} else {{
        int mid = arr.length / 2;
        int[] newArr = new int[mid];
        for (int i = 0; i < mid; i++) {{ newArr[i] = arr[i]; }}
        mystery1(newArr);
        for (int i = 0; i < mid; i++) {{ arr[i] = newArr[i]; }}
    }}
}}
\\end{{verbatim}}

\\noindent \\textbf{{Answer:}} \\(\\boxed{{O(n)}}\\) \\textit{{[The recursive case dominates for large n, and T(n) = T(n/2) + O(n) = O(n).]}}

\\bigskip
\\noindent\\textbf{{(b)}}
\\begin{{verbatim}}
void mystery2(int n) {{
    int sum = 0;
    for (int i = 1; i < Math.pow(4, n); i *= 2) {{
        for (int j = 1; j < n; j++) {{
            if (i < n) {{
                sum += i + j;
            }}
        }}
    }}
}}
\\end{{verbatim}}

\\noindent \\textbf{{Answer:}} \\(\\boxed{{O(n^2)}}\\) \\textit{{[The outer loop runs O(n) times when i < n, and the inner loop is O(n).]}}

\\bigskip
\\noindent\\textbf{{(c)}}
\\begin{{verbatim}}
int mystery3(int n) {{
    while (n > 0) {{
        if (n % 5 == 0) {{ return n; }}
        n--;
    }}
    return 0;
}}
\\end{{verbatim}}

\\noindent \\textbf{{Answer:}} \\(\\boxed{{O(1)}}\\) \\textit{{[Best case: n is already divisible by 5. Worst case: O(n) but typically much faster.]}}

\\bigskip
\\noindent\\textbf{{(d)}}
\\begin{{verbatim}}
int mystery4(int n) {{
    int count = 0;
    for (int i = 0; i < n; i++) {{
        for (int j = 0; j < i; j += 5) {{
            count++;
        }}
    }}
    return count;
}}
\\end{{verbatim}}

\\noindent \\textbf{{Answer:}} \\(\\boxed{{O(n^2)}}\\) \\textit{{[The inner loop runs O(i) times for each i, and Σ(i) from 0 to n-1 is O(n²).]}}
\\end{{enumerate}}

\\bigskip
""")
            elif "stack" in topic_name.lower() or "queue" in topic_name.lower():
                # Stacks and queues solutions
                solution_sections.append(f"""
% =========================
% {i}) Stacks & Queues (Sequential numbering kept)
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Short-answer questions}}

\\medskip
\\begin{{enumerate}}[label=\\arabic*.]
\\item A binary min heap is a type of binary search tree. \\\\
\\textbf{{Answer:}} \\(\\boxed{{\\text{{False}}}}\\) \\textit{{[Heaps and BSTs have different structural properties.]}}

\\item A heap can contain duplicate priorities. \\\\
\\textbf{{Answer:}} \\(\\boxed{{\\text{{True}}}}\\) \\textit{{[Heaps allow duplicate values/priorities.]}}

\\item $T(N)=2T(N-1)+1$, $T(1)=1$. \\\\
\\textbf{{Answer:}} \\(\\boxed{{O(2^N)}}\\) \\textit{{[This is a linear recurrence relation.]}}

\\item \\(f(N)=N\\log(N^2)+\\log^2(N)+N^2\\). \\\\
\\textbf{{Answer:}} \\(\\boxed{{O(N^2)}}\\) \\textit{{[The N² term dominates.]}}

\\item \\(f(N)=(2^N+N)^5\\). \\\\
\\textbf{{Answer:}} \\(\\boxed{{O(32^N)}}\\) \\textit{{[The 2^N term dominates, and (2^N)^5 = 2^(5N) = 32^N.]}}

\\item For any constants $a,b>1$, $\\log_a(n)\\in \\Theta(\\log_b(n))$. \\\\
\\textbf{{Answer:}} \\(\\boxed{{\\text{{True}}}}\\) \\textit{{[Logarithms with different bases differ by a constant factor.]}}

\\item Best-case runtime to find an element in an AVL with $N$ elements. \\\\
\\textbf{{Answer:}} \\(\\boxed{{O(1)}}\\) \\textit{{[Best case: the element is at the root.]}}

\\item Worst-case runtime to find the smallest key in a separate chaining hash table. \\\\
\\textbf{{Answer:}} \\(\\boxed{{O(N)}}\\) \\textit{{[All elements could hash to the same bucket.]}}

\\item Worst-case runtime for pushing $2N$ elements into an initially empty \\texttt{{ArrayStack}} with initial capacity $1$. \\\\
\\textbf{{Answer:}} \\(\\boxed{{O(2N)}}\\) \\textit{{[Each resize operation is O(current size), total is O(N).]}}

\\item Minimum number of elements in a binary max heap of height $3$. \\\\
\\textbf{{Answer:}} \\(\\boxed{{8}}\\) \\textit{{[A complete binary tree of height 3 has 2³ = 8 nodes.]}}

\\item Exact number of leaves in a complete binary tree of $N$ nodes. \\\\
\\textbf{{Answer:}} \\(\\boxed{{\\lceil N/2 \\rceil}}\\) \\textit{{[In a complete binary tree, approximately half the nodes are leaves.]}}
\\end{{enumerate}}

\\bigskip
""")
            else:
                # General topics solutions
                solution_sections.append(f"""
% =========================
% {i}) {topic_name}
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Comprehensive Questions}}

\\begin{{enumerate}}[label=\\alph*)]
\\item (8 points) Provide a comprehensive explanation of {topic_name.lower()}, including its core concepts, properties, and applications.

\\textbf{{Answer:}} \\textit{{[This question requires a detailed explanation covering the fundamental concepts, mathematical properties, and practical applications of {topic_name.lower()}. Include examples and real-world use cases.]}}

\\vspace{{1in}}

\\item (6 points) Solve a practical problem using {topic_name.lower()}. Choose a scenario and walk through your solution step-by-step.

\\textbf{{Answer:}} \\textit{{[Choose a realistic problem that demonstrates the key concepts of {topic_name.lower()}. Show your work clearly, including any calculations, algorithms, or reasoning steps.]}}

\\vspace{{1in}}

\\item (4 points) Compare {topic_name.lower()} with related concepts or alternative approaches. What are the trade-offs?

\\textbf{{Answer:}} \\textit{{[Identify 2-3 related concepts and discuss their advantages and disadvantages compared to {topic_name.lower()}. Consider factors like efficiency, complexity, and applicability.]}}
\\end{{enumerate}}

\\bigskip
""")
        
        # Generate the LaTeX document with solutions
        latex_code = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{enumitem}}
\\usepackage{{fancyhdr}}
\\usepackage{{graphicx}}

\\pagestyle{{fancy}}
\\fancyhf{{}}
\\lhead{{{course_name} {difficulty_level.title()} Difficulty Practice Exam — Solutions}}
\\rhead{{Page \\thepage}}

\\begin{{document}}

\\begin{{center}}
{{\\LARGE \\textbf{{{course_name} {difficulty_level.title()} Difficulty Practice Exam — Solutions}}}}\\[6pt]
\\end{{center}}

\\noindent \\textbf{{Name:}} \\rule{{0.6\\linewidth}}{{0.4pt}}\\[6pt]
\\noindent \\textbf{{Email address:}} \\rule{{0.45\\linewidth}}{{0.4pt}}

\\section*{{Instructions}}
\\begin{{itemize}}[leftmargin=1.2em]
  \\item These are comprehensive solutions to the practice exam.
  \\item Use these solutions to check your work and learn from any mistakes.
  \\item Pay attention to the reasoning and explanations provided.
  \\item These solutions demonstrate the expected level of detail and rigor.
\\end{{itemize}}

\\bigskip

{chr(10).join(solution_sections)}

\\section*{{Logs / Scratch}}
\\noindent\\rule{{\\linewidth}}{{0.4pt}}

\\vfill
\\begin{{center}}
\\emph{{This page intentionally left blank. Good luck with your studies!}}
\\end{{center}}

\\end{{document}}"""
        
        print("✅ Comprehensive exam solutions generated successfully")
        return latex_code
        
    except Exception as e:
        print(f"❌ Solutions generation failed: {str(e)}")
        # Return a minimal solutions document as fallback
        return f"""\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\title{{{course_name} Practice Exam Solutions}}
\\author{{Study Assistant}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle
\\section*{{Solutions}}
\\begin{{enumerate}}
\\item Solutions for topics: {', '.join([p['name'] for p in priorities[:5]])}
\\item Review your work against these solutions
\\item Learn from any mistakes and understand the reasoning
\\end{{enumerate}}
\\end{{document}}"""

def generate_fallback_practice_exam(priorities: List[Dict[str, Any]], course_name: str, difficulty_level: str) -> str:
    """Generate a comprehensive CSE 332-style practice exam in LaTeX format when GPT service is unavailable"""
    try:
        print("📝 Generating comprehensive CSE 332-style fallback practice exam...")
        
        # Create topic sections with the CSE 332 exam format
        topic_sections = []
        for i, topic in enumerate(priorities[:8], 1):  # Limit to 8 topics
            topic_name = topic['name']
            score_value = topic.get('score_value', 10)
            priority = topic.get('study_priority', i)
            
            # Generate different question types based on topic
            if "big-o" in topic_name.lower() or "omega" in topic_name.lower() or "theta" in topic_name.lower():
                # Big-O notation gets multiple choice questions
                topic_sections.append(f"""
% =========================
% {i}) Big-O / Omega / Theta
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{$O$, $\\Omega$, and $\\Theta$}}
\\medskip

\\noindent For each of the following statements, indicate whether it is \\textbf{{always true}}, \\textbf{{sometimes true}}, or \\textbf{{never true}}. You do not need to include an explanation. Assume that the domain and codomain of all functions in this problem are natural numbers (1, 2, 3, \\dots).

\\begin{{enumerate}}[label=\\alph*)]
\\item A function that is $\\Theta(n^2 + n)$ is \\rule{{1.2in}}{{0.4pt}} $\\Theta(n^2)$.

\\noindent\\(\\square\\) Always \\quad \\(\\square\\) Never \\quad \\(\\square\\) Sometimes

\\item A function that is $O(n\\log n)$ is \\rule{{1.2in}}{{0.4pt}} $O(n)$.

\\noindent\\(\\square\\) Always \\quad \\(\\square\\) Never \\quad \\(\\square\\) Sometimes

\\item A function that is $O(\\log^{30} n)$ is \\rule{{1.2in}}{{0.4pt}} $O(n)$.

\\noindent\\(\\square\\) Always \\quad \\(\\square\\) Never \\quad \\(\\square\\) Sometimes

\\item A function that is $\\Omega(n^2)$ is \\rule{{1.2in}}{{0.4pt}} $O(n^{1.5})$.

\\noindent\\(\\square\\) Always \\quad \\(\\square\\) Never \\quad \\(\\square\\) Sometimes

\\item A function that is $\\Theta(2^n)$ is \\rule{{1.2in}}{{0.4pt}} $\\Omega(n^{2025})$.

\\noindent\\(\\square\\) Always \\quad \\(\\square\\) Never \\quad \\(\\square\\) Sometimes
\\end{{enumerate}}

\\bigskip
""")
            elif "recurrence" in topic_name.lower():
                # Recurrence relations get write + tree method
                topic_sections.append(f"""
% =========================
% {i}) Recurrences (Write + Tree)
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{ Write a Recurrence}}

\\medskip
Give a base case and a recurrence for the runtime of the following function. Use variables appropriately for constants (e.g., $c_1, c_2$, etc.) in your recurrence (you do not need to attempt to count the exact number of operations). \\textbf{{You do not need to solve}} this recurrence.
\\medskip

\\noindent\\texttt{{public static int f(int n) \\{{}}\\
\\quad \\texttt{{if (n < 100) \\{{}}\\
\\qquad \\texttt{{return 42;}}\\
\\quad \\texttt{{}}\\[2pt]
\\quad \\texttt{{int m = 10 * n;}}\\
\\quad \\texttt{{return f(n/2) + f(m/100) + f(1);}} \\\\
\\texttt{{}}
\\medskip

\\noindent\\rule{{\\linewidth}}{{0.4pt}}

\\noindent\\(\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\)\\(\\ \\ \\ \\ \\ \\)\\textbf{{For }} $T(n)=$ \\rule{{1.6in}}{{0.4pt}} \\textbf{{when }} $n<100$

\\medskip
\\noindent\\(\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\)\\(\\ \\ \\ \\ \\ \\)\\textbf{{For }} $T(n)=$ \\rule{{1.6in}}{{0.4pt}} \\textbf{{when }} $n\\ge 100$

\\bigskip
\\noindent\\textbf{{Tree Method}}

\\medskip
Suppose that the running time of an algorithm is expressed by the recurrence relation
\\[
T(0)=1,\\qquad T(N)=2T(N-2)+2^N\\ \\text{{ for integers }} N>0.
\\]
For the following questions, use the tree method to solve the recurrence relation. You may assume that $N$ is always even.

\\begin{{enumerate}}[label=\\alph*)]
\\item Sketch the tree in the space below. Include at least the first 3 levels of the tree.

\\vspace{{3.2in}}

\\item Indicate exactly the total amount of work done at level $i$ of the tree.

\\vspace{{1.2in}}

\\item Give a simplified $\\Theta$ bound on the solution.

\\medskip
$\\Theta($\\rule{{1.6in}}{{0.4pt}}$)$
\\end{{enumerate}}

\\bigskip
""")
            elif "avl" in topic_name.lower() or "tree" in topic_name.lower():
                # AVL trees get insertion and rotation questions
                topic_sections.append(f"""
% =========================
% {i}) AVL Trees
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{AVL}}

\\begin{{enumerate}}[label=\\alph*)]
\\item Start from an empty AVL tree, and do the insertions of elements in the following order:
\\[
50,\\ 40,\\ 30,\\ 35,\\ 34,\\ 60,\\ 70,\\ 80
\\]
\\textit{{Box the final tree.}}

\\vspace{{3in}}

\\item To your tree in part (a), add in the value 90. \\textit{{Box the final tree.}}

\\vspace{{2.6in}}

\\item The insertion from part (b) required: (\\textbf{{SELECT ONE}})

\\medskip
\\(\\square\\) One single rotation \\quad
\\(\\square\\) One double rotation \\quad
\\(\\square\\) One single rotation \\textbf{{AND}} one double rotation \\quad
\\(\\square\\) No rotations
\\end{{enumerate}}

\\bigskip
""")
            elif "heap" in topic_name.lower():
                # Binary heaps get array representation and operations
                topic_sections.append(f"""
% =========================
% {i}) Binary Heaps
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Heaps}}

\\medskip
Here's an array representation of a 0-indexed binary heap, named \\texttt{{heap}} (the priority of each element is the same as its value, just like in lecture):
\\[
24\\ 26\\ 53\\ 84\\ 50\\ 63\\ 78\\ 99\\ 96\\ 71\\ 67
\\]

\\begin{{enumerate}}[label=\\alph*)]
\\item This is a \\rule{{1.2in}}{{0.4pt}} (choose one: max/min) heap.

\\item Draw the visual representation of the given heap.

\\vspace{{3in}}

\\item Suppose we execute the code: \\texttt{{heap.insert(x)}}, where $x$ is some integer. For each assertion below, give a value of $x$ such that the assertion is true after the code is executed, or explain why no such $x$ exists.

\\begin{{enumerate}}[label=\\roman*)]
\\item Some node with value ``63'' has a child with value ``71''

\\vspace{{0.8in}}

\\item No node with value ``78'' has a parent with value ``53''

\\vspace{{0.8in}}
\\end{{enumerate}}
\\end{{enumerate}}

\\bigskip
""")
            elif "dictionary" in topic_name.lower() or "adt" in topic_name.lower() or "hash" in topic_name.lower():
                # Dictionary ADT gets hashing questions
                topic_sections.append(f"""
% =========================
% {i}) Dictionary ADT (via Hashing)
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Hashing}}

\\begin{{enumerate}}[label=\\alph*)]
\\item For the following hash table, insert the following elements in this order: 23, 5, 3, 66, 43, 19, 79, using the Linear Probing technique. Use the hash function $h(x)=x\\bmod 10$. In the table below, the first column holds the indices and the second holds the values.

\\medskip
\\begin{{tabular}}{{|c|c|c|c|c|c|c|c|c|c|c|}}
\\hline
Index & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9\\\\
\\hline
Value & & & & & & & & & & \\\\
\\hline
\\end{{tabular}}

\\medskip

\\item Consider an initially empty quadratic-probing hash table with table size $p$ and hash function $h(x)=0$. What is the best-case runtime of inserting $n$ elements into the table? Give a simplified tight big-$O$ bound in terms of $n$.

\\medskip
\\noindent $O($\\rule{{1.6in}}{{0.4pt}}$)$
\\end{{enumerate}}

\\bigskip
""")
            elif "complexity" in topic_name.lower() or "analysis" in topic_name.lower():
                # Complexity analysis gets code analysis
                topic_sections.append(f"""
% =========================
% {i}) Time/Space Complexity (Code Analysis)
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Code Analysis}}

\\medskip
Describe the \\textbf{{worst-case}} running time for the following pseudocode functions in Big-$O$ notation in terms of the variable $n$. Your answer MUST be tight and simplified.

\\medskip
\\noindent\\textbf{{(a)}}
\\begin{{verbatim}}
public static void mystery1(int[] arr) {{
    if (arr.length < 10000) {{
        int cnt = 0;
        for(int i = 0; i < arr.length; i++) {{
            for(int j = 0; j < arr.length; j++) {{
                if (arr[i] < arr[j]) {{
                    cnt++;
                }}
            }}
            arr[i] = cnt;
        }}
    }} else {{
        int mid = arr.length / 2;
        int[] newArr = new int[mid];
        for (int i = 0; i < mid; i++) {{
            newArr[i] = arr[i];
        }}
        mystery1(newArr);
        for (int i = 0; i < mid; i++) {{
            arr[i] = newArr[i];
        }}
    }}
}}
\\end{{verbatim}}

\\noindent $O($\\rule{{1.6in}}{{0.4pt}}$)$

\\bigskip
\\noindent\\textbf{{(b)}}
\\begin{{verbatim}}
void mystery2(int n) {{
    int sum = 0;
    for (int i = 1; i < Math.pow(4, n); i *= 2) {{
        for (int j = 1; j < n; j++) {{
            if (i < n) {{
                sum += i + j;
            }}
        }}
    }}
}}
\\end{{verbatim}}

\\noindent $O($\\rule{{1.6in}}{{0.4pt}}$)$
\\end{{enumerate}}

\\bigskip
""")
            elif "stack" in topic_name.lower() or "queue" in topic_name.lower():
                # Stacks and queues get short answer questions
                topic_sections.append(f"""
% =========================
% {i}) Stacks & Queues (Sequentially numbered items)
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Short-answer questions}}

\\medskip
Unless otherwise noted, assume all data structures are implemented as described in lecture. For questions asking you about runtime, give a simplified, tight Big-$O$ bound.

\\begin{{enumerate}}[label=\\arabic*.]
\\item A binary min heap is a type of binary search tree.

\\item A heap can contain duplicate priorities.

\\item Give a simplified, tight big-$O$ bound for: $T(N)=2T(N-1)+1$, given $T(1)=1$.

\\medskip

\\item Give a simplified, tight big-$O$ bound for:
\\[
f(N)=N\\log(N^2)+\\log^2(N)+N^2
\\]

\\medskip

\\item Give the best-case runtime for finding an element in an AVL tree containing $N$ elements.

\\medskip

\\item Give the worst-case runtime for finding the smallest key in a separate chaining hash table.

\\medskip
\\end{{enumerate}}

\\bigskip
""")
            else:
                # General topics get a mix of question types
                topic_sections.append(f"""
% =========================
% {i}) {topic_name}
% =========================
\\section*{{Topic {i}: {topic_name}}}
\\noindent\\textbf{{Comprehensive Questions}}

\\begin{{enumerate}}[label=\\alph*)]
\\item (8 points) Provide a comprehensive explanation of {topic_name.lower()}, including its core concepts, properties, and applications.

\\vspace{{2in}}

\\item (6 points) Solve a practical problem using {topic_name.lower()}. Choose a scenario and walk through your solution step-by-step.

\\vspace{{2in}}

\\item (4 points) Compare {topic_name.lower()} with related concepts or alternative approaches. What are the trade-offs?

\\vspace{{1.5in}}
\\end{{enumerate}}

\\bigskip
""")
        
        # Generate the LaTeX document with CSE 332 style
        latex_code = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{enumitem}}
\\usepackage{{fancyhdr}}
\\usepackage{{graphicx}}

\\pagestyle{{fancy}}
\\fancyhf{{}}
\\lhead{{{course_name} {difficulty_level.title()} Difficulty Practice Exam}}
\\rhead{{Page \\thepage}}

\\begin{{document}}

\\begin{{center}}
{{\\LARGE \\textbf{{{course_name} {difficulty_level.title()} Difficulty Practice Exam}}}}\\[6pt]
\\end{{center}}

\\noindent \\textbf{{Name:}} \\rule{{0.6\\linewidth}}{{0.4pt}}\\[6pt]
\\noindent \\textbf{{Email address:}} \\rule{{0.45\\linewidth}}{{0.4pt}}

\\section*{{Instructions}}
\\begin{{itemize}}[leftmargin=1.2em]
  \\item The allotted time is 90 minutes.
  \\item This is a closed-book and closed-notes practice exam.
  \\item Read the directions carefully, especially for problems that require you to show work or provide an explanation.
  \\item When provided, write your answers in the box or on the line provided.
  \\item Unless otherwise noted, every time we ask for an $O$, $\\Omega$, or $\\Theta$ bound, it must be \\emph{{simplified and tight}}.
  \\item Unless otherwise noted, assume all data structures are implemented as described in lecture.
  \\item If you run out of room on a page, indicate where the answer continues.
\\end{{itemize}}

\\subsection*{{Advice}}
\\begin{{itemize}}[leftmargin=1.2em]
  \\item If you feel like you're stuck on a problem, you may want to skip it and come back at the end if you have time.
  \\item Look at the question titles to see if you want to start somewhere other than problem 1.
  \\item Relax and take a few deep breaths. You've got this! :-)
\\end{{itemize}}

\\bigskip

{chr(10).join(topic_sections)}

\\section*{{Logs / Scratch}}
\\noindent\\rule{{\\linewidth}}{{0.4pt}}

\\vfill
\\begin{{center}}
\\emph{{This page intentionally left blank. Good luck with your studies!}}
\\end{{center}}

\\end{{document}}"""
        
        print("✅ Comprehensive CSE 332-style fallback practice exam generated successfully")
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

@app.get("/test-gpt")
async def test_gpt_service():
    """Test endpoint to check if GPT service is working"""
    try:
        if not gpt_service:
            return {"status": "error", "message": "GPT service not initialized"}
        
        # Test the connection
        is_working = await gpt_service.test_connection()
        
        if is_working:
            return {"status": "success", "message": "GPT service is working correctly"}
        else:
            return {"status": "warning", "message": "GPT service test failed"}
            
    except Exception as e:
        return {"status": "error", "message": f"GPT service test failed: {str(e)}"}

@app.get("/debug-session/{session_id}")
async def debug_session(session_id: str, db = Depends(get_db)):
    """Debug endpoint to check session data"""
    try:
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session.id,
            "course_name": session.course_name,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "materials_keys": list(session.materials.keys()) if session.materials else [],
            "exam_coverage_keys": list(session.exam_coverage.keys()) if session.exam_coverage else [],
            "practice_exam_keys": list(session.practice_exam.keys()) if session.practice_exam else [],
            "has_study_priorities": "study_priorities" in (session.materials or {}),
            "study_priorities_count": len(session.materials.get("study_priorities", [])) if session.materials else 0,
            "exam_topics_count": len(session.exam_coverage.get("topics", [])) if session.exam_coverage else 0,
            "practice_exam_questions_count": len(session.practice_exam.get("questions", [])) if session.practice_exam else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
