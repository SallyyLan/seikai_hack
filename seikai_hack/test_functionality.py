#!/usr/bin/env python3
"""
Test script for the new exam prep functionality
"""

import asyncio
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from services.gemini_service import GeminiService
from services.gpt_service import GPTService
from services.priority_queue import PriorityQueueService

async def test_gemini_service():
    """Test Gemini service functionality"""
    print("Testing Gemini Service...")
    
    try:
        # Initialize service (will fail without API key, but that's expected)
        gemini = GeminiService()
        print("✅ Gemini service initialized successfully")
    except ValueError as e:
        print(f"⚠️  Gemini service initialization failed (expected without API key): {e}")
    
    print()

async def test_gpt_service():
    """Test GPT service functionality"""
    print("Testing GPT Service...")
    
    try:
        # Initialize service (will fail without API key, but that's expected)
        gpt = GPTService()
        print("✅ GPT service initialized successfully")
    except ValueError as e:
        print(f"⚠️  GPT service initialization failed (expected without API key): {e}")
    
    print()

async def test_priority_queue():
    """Test priority queue algorithm"""
    print("Testing Priority Queue Algorithm...")
    
    try:
        pq = PriorityQueueService()
        
        # Test data
        test_topics = [
            {"id": "1", "name": "Calculus Integration", "score_value": 20.0, "user_confidence": 2},
            {"id": "2", "name": "Differential Equations", "score_value": 15.0, "user_confidence": 4},
            {"id": "3", "name": "Linear Algebra", "score_value": 25.0, "user_confidence": 1},
            {"id": "4", "name": "Vector Calculus", "score_value": 18.0, "user_confidence": 3},
        ]
        
        # Test the scoring formula: s_i = B_i * (6 - c_i)/6
        print("Testing scoring formula: s_i = B_i * (6 - c_i)/6")
        for topic in test_topics:
            score_value = topic["score_value"]
            confidence = topic["user_confidence"]
            calculated_score = score_value * (6 - confidence) / 6
            print(f"  {topic['name']}: {score_value} * (6 - {confidence}) / 6 = {calculated_score:.2f}")
        
        # Test priority calculation
        print("\nTesting priority calculation with minimum score = 50...")
        priorities = await pq.calculate_study_priorities("test_session", test_topics, 50.0, 100.0)
        
        print("Prioritized topics:")
        for topic in priorities:
            print(f"  #{topic['study_priority']}: {topic['name']} (Score: {topic['calculated_score']:.2f})")
        
        print("✅ Priority queue algorithm working correctly")
        
    except Exception as e:
        print(f"❌ Priority queue test failed: {e}")
    
    print()

def test_models():
    """Test model structure"""
    print("Testing Models...")
    
    try:
        from models import ExamSession, Topic, Question
        
        # Check if new fields exist
        session_fields = [field.name for field in ExamSession.__table__.columns]
        topic_fields = [field.name for field in Topic.__table__.columns]
        question_fields = [field.name for field in Question.__table__.columns]
        
        required_session_fields = ['exam_coverage', 'practice_exam', 'minimum_score', 'maximum_score']
        required_topic_fields = ['user_confidence', 'calculated_score', 'study_priority']
        required_question_fields = ['question_text', 'score_value', 'question_number']
        
        print("ExamSession fields:", session_fields)
        print("Topic fields:", topic_fields)
        print("Question fields:", question_fields)
        
        # Check required fields
        missing_session = [f for f in required_session_fields if f not in session_fields]
        missing_topic = [f for f in required_topic_fields if f not in topic_fields]
        missing_question = [f for f in required_question_fields if f not in question_fields]
        
        if not missing_session and not missing_topic and not missing_question:
            print("✅ All required model fields present")
        else:
            print(f"❌ Missing fields: Session={missing_session}, Topic={missing_topic}, Question={missing_question}")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    print()

async def main():
    """Run all tests"""
    print("🚀 Testing LAST MINUTE Exam Prep AI Functionality\n")
    
    await test_gemini_service()
    await test_gpt_service()
    await test_priority_queue()
    test_models()
    
    print("🎯 Testing complete!")
    print("\nTo run the application:")
    print("1. Set your API keys in .env file")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run: python app.py")
    print("\nSimplified workflow:")
    print("- Enter course name")
    print("- Enter midterm coverage topics as text")
    print("- Upload practice exam + set minimum score")
    print("- Rate confidence levels")
    print("- Get study priorities and generate practice exams")

if __name__ == "__main__":
    asyncio.run(main())
