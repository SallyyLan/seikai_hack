#!/usr/bin/env python3
"""
Database reset script for LAST MINUTE Exam Prep AI
"""

import os
from dotenv import load_dotenv

def reset_database():
    """Reset the database by removing the old file and recreating tables"""
    try:
        print("🗄️  Resetting database...")
        
        # Load environment variables
        load_dotenv()
        
        # Import database components
        from database import engine
        from models import Base
        
        db_file = "exam_prep.db"
        
        # Remove old database file if it exists
        if os.path.exists(db_file):
            os.remove(db_file)
            print(f"✅ Removed old database: {db_file}")
        else:
            print("ℹ️  No existing database file found")
        
        # Create new database with updated schema
        print("Creating new database with updated schema...")
        Base.metadata.create_all(bind=engine)
        print("✅ New database created successfully!")
        
        # Test the new database
        print("Testing new database...")
        from sqlalchemy import text
        with engine.connect() as conn:
            # Test if new columns exist
            result = conn.execute(text("SELECT exam_coverage, practice_exam, minimum_score, maximum_score FROM exam_sessions LIMIT 1"))
            print("✅ New schema verified successfully!")
        
        print("\n🎯 Database reset complete!")
        print("You can now run the application with the updated schema.")
        return True
        
    except Exception as e:
        print(f"❌ Database reset failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = reset_database()
    if not success:
        print("\n❌ Database reset failed. Please check the errors above.")
