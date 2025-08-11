# 🚨 LAST MINUTE Exam Prep AI 🚨

An AI-powered application that helps students with last-minute exam preparation by analyzing their handwritten practice work and creating a dynamic priority queue for study topics.

## ✨ Features

- **📸 Multi-Format Text Extraction**: 
  - Handwritten work via Mathpix OCR
  - PDF documents via PyPDF2
  - Multiple image formats (PNG, JPG, GIF, BMP, WebP, TIFF)
- **🧠 AI Analysis**: GPT-4 analyzes correctness and provides detailed feedback
- **🎯 Dynamic Priority Queue**: Automatically prioritizes topics based on performance
- **📚 Course Material Integration**: Upload textbooks, slides, homework, and past exams
- **⚡ Urgent-Focused Interface**: Simple, fast interface for last-minute studying
- **📊 Real-time Feedback**: Instant analysis and study recommendations
- **🔍 Enhanced File Processing**: Intelligent file type detection and metadata extraction

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Mathpix API credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd seikai_hack
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8000`

## 🔑 API Keys Required

### OpenAI API
- Get your API key from [OpenAI Platform](https://platform.openai.com/)
- Add to `.env`: `OPENAI_API_KEY=your_key_here`

### Mathpix API
- Sign up at [Mathpix](https://mathpix.com/)
- Get your App ID and App Key
- Add to `.env`:
  ```
  MATHPIX_APP_ID=your_app_id
  MATHPIX_APP_KEY=your_app_key
  ```

## 📱 How It Works

### 1. Start Exam Session
- Enter course name and exam date
- Get a unique session ID

### 2. Upload Course Materials
- Textbook (PDF/DOC)
- Lecture slides (PDF/PPT)
- Homework assignments
- Past exams with solutions
- Course syllabus

### 3. Upload Practice Work
- **Multiple File Formats Supported:**
  - 📱 **Images**: PNG, JPG, JPEG, GIF, BMP, WebP, TIFF
  - 📄 **PDF Documents**: Practice tests, scanned work, digital documents
  - ✍️ **Handwritten Work**: Photos of your handwritten solutions
- AI extracts text using Mathpix (images) or PyPDF2 (PDFs)
- GPT-4 analyzes correctness and provides detailed feedback

### 4. Get Study Priorities
- Dynamic priority queue based on performance
- Topics you struggle with get higher priority
- Success improves topics move down the queue

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS)
    ↓
FastAPI Backend
    ↓
Services:
├── MathpixService (Text extraction)
├── GPTService (AI analysis)
├── PriorityQueueService (Topic prioritization)
└── FileProcessor (File handling)
    ↓
SQLite Database
```

## 📁 Project Structure

```
seikai_hack/
├── app.py                 # Main FastAPI application
├── models.py             # Database models
├── database.py           # Database configuration
├── requirements.txt      # Python dependencies
├── env.example          # Environment variables template
├── README.md            # This file
├── services/            # Service layer
│   ├── mathpix_service.py
│   ├── gpt_service.py
│   ├── priority_queue.py
│   └── file_processor.py
├── templates/           # HTML templates
│   ├── index.html       # Main interface
│   └── session.html     # Session summary
├── static/              # Static files (CSS, JS)
├── uploads/             # Uploaded files
└── processed/           # Processed file data
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `MATHPIX_APP_ID` | Mathpix App ID | Required |
| `MATHPIX_APP_KEY` | Mathpix App Key | Required |
| `DATABASE_URL` | Database connection string | `sqlite:///./exam_prep.db` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

### File Upload Limits

- Maximum file size: 10MB
- Supported formats: PDF, DOC, DOCX, PPT, PPTX, Images
- Files are automatically cleaned up after 24 hours

## 🎯 Priority Queue Algorithm

The priority queue dynamically adjusts based on:

- **Correct Answers**: Decrease priority (less need to study)
- **Incorrect Answers**: Increase priority (more need to study)
- **Success Rate**: Topics with low success rates get higher priority
- **Confidence**: Low confidence in AI analysis increases priority
- **Practice Frequency**: Recent practice affects priority scoring

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker (coming soon)
```bash
docker build -t exam-prep-ai .
docker run -p 8000:8000 exam-prep-ai
```

## 🔒 Security Features

- File type validation
- File size limits
- Secure file handling
- API key protection
- Input sanitization

## 🧪 Testing

```bash
# Run basic tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=app tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Create a GitHub issue
- **Questions**: Check the documentation or create a discussion
- **Emergency**: For urgent issues, contact the maintainers

## 🎓 For Students

This app is designed for **URGENT** exam prep:

- ⚡ **Fast Setup**: Get started in under 2 minutes
- 🎯 **Smart Focus**: AI tells you exactly what to study
- 📱 **Mobile Friendly**: Works on phones and tablets
- 🚨 **Last Minute**: Perfect for cram sessions

## 🔮 Future Features

- [ ] Mobile app (iOS/Android)
- [ ] Integration with learning management systems
- [ ] Collaborative study groups
- [ ] Progress tracking over time
- [ ] Custom study plans
- [ ] Video explanations
- [ ] Practice problem generation

---

**Built with ❤️ for students who need to study smart, not just hard!**

*Remember: This is for last-minute prep. Start studying earlier next time! 😉*
