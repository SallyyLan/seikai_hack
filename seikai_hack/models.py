from sqlalchemy import Column, String, DateTime, Boolean, Float, Text, ForeignKey, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    exam_sessions = relationship("ExamSession", back_populates="user")

class ExamSession(Base):
    __tablename__ = "exam_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    course_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    materials = Column(JSON, default={})
    
    # New fields for exam preparation
    exam_coverage = Column(JSON, default={})  # Topics and content from exam coverage
    practice_exam = Column(JSON, default={})  # Practice exam questions and scores
    minimum_score = Column(Float, default=0.0)  # Minimum score user wants to achieve
    maximum_score = Column(Float, default=0.0)  # Maximum possible score from practice exam
    
    user = relationship("User", back_populates="exam_sessions")
    questions = relationship("Question", back_populates="session")
    topics = relationship("Topic", back_populates="session")

class Question(Base):
    __tablename__ = "questions"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("exam_sessions.id"))
    extracted_text = Column(Text, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    feedback = Column(Text)
    topics = Column(JSON, default=[])
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # New fields for practice exam questions
    question_text = Column(Text)  # Clean text of the question
    score_value = Column(Float, default=0.0)  # Points this question is worth
    question_number = Column(Integer, default=0)  # Question number in practice exam
    
    session = relationship("ExamSession", back_populates="questions")

class Topic(Base):
    __tablename__ = "topics"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("exam_sessions.id"))
    name = Column(String, nullable=False)
    priority_score = Column(Float, default=1.0)
    questions_attempted = Column(Integer, default=0)
    questions_correct = Column(Integer, default=0)
    last_practiced = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # New fields for confidence scoring
    user_confidence = Column(Integer, default=1)  # 1-6 scale confidence level
    calculated_score = Column(Float, default=0.0)  # Score based on confidence formula
    study_priority = Column(Integer, default=0)  # Priority order for studying
    
    session = relationship("ExamSession", back_populates="topics")
