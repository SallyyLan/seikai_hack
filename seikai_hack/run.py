#!/usr/bin/env python3
"""
Exam Prep AI - Dual AI System - Startup Script
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
    
    print("📚 Exam Prep AI - Dual AI System 📚")
    print("=" * 45)
    print(f"Starting server on {host}:{port}")
    print("Open your browser to: http://localhost:8000")
    print("=" * 45)
    
    # Check for required API keys (Dual AI System only)
    required_keys = ["GEMINI_API_KEY", "HF_TOKEN"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("⚠️  WARNING: Missing required API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease set these in your .env file:")
        print("   GEMINI_API_KEY=your_gemini_key_here")
        print("   HF_TOKEN=your_huggingface_token_here")
        print("\nSee README_DUAL_AI.md for setup instructions")
        print("\nStarting in demo mode (some features may not work)")
    else:
        print("✅ All required API keys found!")
        print("🤖 Gemini: Ready for PDF/Image → LaTeX extraction")
        print("🟢 GPT-OSS: Ready for validation & practice generation")
    
    print("=" * 45)
    
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
