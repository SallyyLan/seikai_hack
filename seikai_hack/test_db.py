#!/usr/bin/env python3
"""
Simple database test script to verify models and connection
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_database():
    """Test database connection and model creation"""
    try:
        print("Testing database connection...")
        
        # Import database components
        from database import engine, test_connection
        from models import Base, ExamSession, Topic, Question
        
        print("✅ Models imported successfully")
        
        # Test basic connection
        if test_connection():
            print("✅ Database connection test passed")
        else:
            print("❌ Database connection test failed")
            return False
        
        # Test table creation
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        
        # Test model instantiation
        print("Testing model instantiation...")
        session = ExamSession(
            id="test-123",
            course_name="Test Course"
        )
        print("✅ ExamSession model created successfully")
        
        topic = Topic(
            id="topic-123",
            session_id="test-123",
            name="Test Topic"
        )
        print("✅ Topic model created successfully")
        
        question = Question(
            id="q-123",
            session_id="test-123",
            extracted_text="Test question",
            is_correct=True
        )
        print("✅ Question model created successfully")
        
        print("\n🎯 All database tests passed!")
        print("The database and models are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database()
    if success:
        print("\n🚀 Database is ready! You can now run the application.")
    else:
        print("\n❌ Database setup failed. Please check the errors above.")
