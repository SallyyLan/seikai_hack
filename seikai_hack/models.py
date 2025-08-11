from sqlalchemy import Column, String, DateTime, Boolean, Float, Text, ForeignKey, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    exam_sessions = relationship("ExamSession", back_populates="user")

class ExamSession(Base):
    __tablename__ = "exam_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    course_name = Column(String, nullable=False)
    exam_date = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    materials = Column(JSON, default={})
    
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
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("ExamSession", back_populates="questions")

class Topic(Base):
    __tablename__ = "topics"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("exam_sessions.id"))
    name = Column(String, nullable=False)
    priority_score = Column(Float, default=1.0)
    questions_attempted = Column(Integer, default=0)
    questions_correct = Column(Integer, default=0)
    last_practiced = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("ExamSession", back_populates="topics")
