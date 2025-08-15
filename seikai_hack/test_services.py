#!/usr/bin/env python3
"""
Test script to check service initialization and API keys
"""

import os
from dotenv import load_dotenv

def test_services():
    """Test service initialization"""
    print("🔍 Testing Service Initialization...")
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    print("\n📋 Environment Variables:")
    gemini_key = os.getenv("GEMINI_API_KEY")
    gpt_model = os.getenv("GPT_OSS_MODEL")
    hf_token = os.getenv("HF_TOKEN")
    
    print(f"GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Missing'}")
    print(f"GPT_OSS_MODEL: {'✅ Set' if gpt_model else '❌ Missing'}")
    print(f"HF_TOKEN: {'✅ Set' if hf_token else '❌ Missing'}")
    
    # Test service imports
    print("\n🧪 Testing Service Imports:")
    
    try:
        from services.gemini_service import GeminiService
        print("✅ GeminiService imported successfully")
        
        if gemini_key:
            try:
                gemini = GeminiService()
                print("✅ GeminiService initialized successfully")
                
                # Test model availability
                print("Testing Gemini model availability...")
                import asyncio
                test_result = asyncio.run(gemini.test_model_availability())
                if test_result["status"] == "success":
                    print("✅ Gemini models are working correctly")
                else:
                    print(f"⚠️  Gemini models test failed: {test_result['error']}")
                    
            except Exception as e:
                print(f"❌ GeminiService initialization failed: {e}")
                print("💡 This might be due to:")
                print("   - Invalid API key")
                print("   - Model availability issues")
                print("   - API version compatibility")
        else:
            print("⚠️  Skipping GeminiService test (no API key)")
            
    except ImportError as e:
        print(f"❌ Failed to import GeminiService: {e}")
    
    try:
        from services.gpt_service import GPTService
        print("✅ GPTService imported successfully")
        
        if gpt_model and hf_token:
            try:
                gpt = GPTService()
                print("✅ GPTService initialized successfully")
            except Exception as e:
                print(f"❌ GPTService initialization failed: {e}")
        else:
            print("⚠️  Skipping GPTService test (missing API keys)")
            
    except ImportError as e:
        print(f"❌ Failed to import GPTService: {e}")
    
    try:
        from services.priority_queue import PriorityQueueService
        priority_queue = PriorityQueueService()
        print("✅ PriorityQueueService initialized successfully")
    except Exception as e:
        print(f"❌ PriorityQueueService initialization failed: {e}")
    
    try:
        from services.file_processor import FileProcessor
        file_processor = FileProcessor()
        print("✅ FileProcessor initialized successfully")
    except Exception as e:
        print(f"❌ FileProcessor initialization failed: {e}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not gemini_key:
        print("- Set GEMINI_API_KEY in your .env file")
        print("- Get your API key from: https://makersuite.google.com/app/apikey")
    
    if not gpt_model or not hf_token:
        print("- Set GPT_OSS_MODEL and HF_TOKEN in your .env file")
        print("- Get your Hugging Face token from: https://huggingface.co/settings/tokens")
    
    if gemini_key and gpt_model and hf_token:
        print("✅ All required API keys are set!")
        print("Try running the application again: python app.py")

if __name__ == "__main__":
    test_services()
