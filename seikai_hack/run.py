#!/usr/bin/env python3
"""
LAST MINUTE Exam Prep AI - Startup Script
"""

import uvicorn
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print("🚨 LAST MINUTE Exam Prep AI 🚨")
    print("=" * 40)
    print(f"Starting server on {host}:{port}")
    print("Open your browser to: http://localhost:8000")
    print("=" * 40)
    
    # Check for required API keys
    required_keys = ["OPENAI_API_KEY", "MATHPIX_APP_ID", "MATHPIX_APP_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("⚠️  WARNING: Missing required API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease set these in your .env file")
        print("See env.example for reference")
        print("\nStarting in demo mode (some features may not work)")
    
    # Start the server
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )

if __name__ == "__main__":
    main()
