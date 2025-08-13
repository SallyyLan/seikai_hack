# Exam Prep AI - Dual AI System

A streamlined exam preparation application that uses **two AI models** working together:

- **Gemini 2.0 Flash**: Extracts LaTeX code from PDFs and handwritten images
- **GPT-OSS 20B**: Validates LaTeX, classifies topics, and generates practice exams

## 🚀 Quick Start

### 1. Set up environment variables
Create a `.env` file in the project root:

```bash
# AI API Keys
GEMINI_API_KEY=AIzaSyC_uYzO4pxlE4E4E6jbWRRO2OOIhgHWiEU
HF_TOKEN=hf_your_huggingface_token_here

# AI Model Settings
GEMINI_MODEL=gemini-2.0-flash-exp
GPT_OSS_MODEL=openai/gpt-oss-20b:fireworks-ai

# Server Settings
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python run.py
```

The app will be available at `http://localhost:8000`

## 🔧 How It Works

### Workflow Overview

1. **Upload Practice Exam** (PDF)
   - Gemini extracts LaTeX code from the PDF
   - GPT-OSS validates the LaTeX and classifies topics

2. **Upload Student Work** (Image)
   - Gemini extracts LaTeX from handwritten work
   - GPT-OSS validates the LaTeX syntax

3. **Generate Practice Exam**
   - GPT-OSS creates new practice problems based on identified topics
   - Maintains the same format and style as the original exam

### AI Model Responsibilities

#### Gemini (Google)
- **PDF Processing**: Converts PDF content to LaTeX
- **Image Analysis**: Extracts LaTeX from handwritten work
- **Content Extraction**: Identifies mathematical expressions and formatting

#### GPT-OSS (Hugging Face)
- **LaTeX Validation**: Checks syntax and identifies errors
- **Topic Classification**: Analyzes content and categorizes by subject
- **Practice Generation**: Creates new exam questions and solutions
- **Work Analysis**: Evaluates student solutions and provides feedback

## 📁 Project Structure

```
seikai_hack/
├── app.py                 # Main FastAPI application
├── services/
│   ├── gemini_service.py      # Gemini API integration
│   ├── gpt_oss_service.py     # GPT-OSS API integration
│   ├── priority_queue.py      # Topic prioritization
│   └── file_processor.py      # File handling utilities
├── templates/
│   └── index.html             # Main UI
├── static/                    # CSS and static files
├── uploads/                   # User uploaded files
├── processed/                 # AI analysis results
└── requirements.txt           # Python dependencies
```

## 🎯 API Endpoints

### Core Endpoints
- `GET /` - Main application page
- `GET /api/status` - Check AI service availability
- `GET /api/health` - Health check

### Workflow Endpoints
- `POST /api/upload-practice-exam` - Upload and analyze practice exam
- `POST /api/upload-student-work` - Upload and analyze student work
- `POST /api/generate-practice-exam` - Generate new practice exam
- `POST /api/analyze-student-work` - Analyze student work vs solution
- `GET /api/session/{session_id}` - Get session data

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `HF_TOKEN` | Hugging Face token for GPT-OSS | Yes |
| `GEMINI_MODEL` | Gemini model to use | No (default: gemini-2.0-flash-exp) |
| `GPT_OSS_MODEL` | GPT-OSS model to use | No (default: openai/gpt-oss-20b:fireworks-ai) |

### Getting API Keys

#### Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with Google account
3. Click "Get API key"
4. Create new key or copy existing one

#### Hugging Face Token
1. Go to [Hugging Face](https://huggingface.co/)
2. Sign up/login
3. Go to Settings → Access Tokens
4. Create new token with "Read" permissions

## 📱 Usage

### Step 1: Upload Practice Exam
1. Upload a PDF practice exam
2. Gemini extracts LaTeX code
3. GPT-OSS validates and classifies topics
4. Get session ID for next steps

### Step 2: Upload Student Work
1. Enter session ID from Step 1
2. Upload image of handwritten work
3. Gemini extracts LaTeX from image
4. GPT-OSS validates the LaTeX

### Step 3: Generate Practice Exam
1. Enter session ID and target topics
2. Select difficulty level
3. GPT-OSS generates new practice problems
4. View questions and solutions in LaTeX format

## 🔍 Features

- **Dual AI Pipeline**: Gemini for extraction, GPT-OSS for analysis
- **LaTeX Support**: Full LaTeX code generation and validation
- **Topic Classification**: Automatic identification of mathematical topics
- **Practice Generation**: AI-generated practice exams matching original format
- **Real-time Validation**: Instant feedback on LaTeX syntax
- **Session Management**: Track multiple exam sessions

## 🚨 Troubleshooting

### Common Issues

#### Gemini API Errors
- Check `GEMINI_API_KEY` is set correctly
- Verify API key has proper permissions
- Check Gemini service status

#### GPT-OSS API Errors
- Verify `HF_TOKEN` is valid
- Check Hugging Face service status
- Ensure token has "Read" permissions

#### File Upload Issues
- Check file size limits
- Verify file format is supported
- Ensure proper file permissions

### Debug Mode
Set `DEBUG=true` in `.env` for detailed error messages and logging.

## 🔮 Future Enhancements

- **Database Integration**: Store results persistently
- **User Authentication**: Multi-user support
- **Advanced Analytics**: Performance tracking and insights
- **Export Features**: PDF generation, LaTeX compilation
- **Mobile App**: Native mobile interface
- **Batch Processing**: Handle multiple files simultaneously

## 📄 License

This project is for educational purposes. Please ensure compliance with API terms of service for both Gemini and GPT-OSS.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation for Gemini and GPT-OSS
3. Check FastAPI and Python dependencies
4. Open an issue in the repository

---

**Built with ❤️ using Gemini + GPT-OSS for intelligent exam preparation**
