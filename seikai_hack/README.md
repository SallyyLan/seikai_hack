# 🚨 LAST MINUTE Exam Prep AI 🚨

An intelligent exam preparation system that uses AI to help students prioritize their study time and generate personalized practice materials.

## ✨ Features

### 🔍 **Smart File Processing with Gemini**
- **Exam Coverage Analysis**: Upload your exam coverage document and let Gemini identify the main topics
- **Practice Exam Processing**: Upload practice exams to extract questions, topics, and score values
- **LaTeX Conversion**: Convert handwritten work and PDFs to clean LaTeX code for better AI processing
- **Multi-format Support**: Handles PDFs, images (PNG, JPG, JPEG), and handwritten work

### 🎯 **Intelligent Study Prioritization**
- **Confidence Scoring**: Rate your confidence on each topic using a 6-scale system (1 = Very Unconfident, 6 = Very Confident)
- **Smart Algorithm**: Uses the formula `s_i = B_i * (6 - c_i)/6` where:
  - `s_i` = calculated score for topic
  - `B_i` = score worth of topic in practice exam
  - `c_i` = your confidence level (1-6)
- **Priority Queue**: Automatically builds study priorities to reach your target score
- **Similar Score Detection**: When scores are similar (within threshold), AI determines which topic is easier to study

### 📝 **AI-Generated Practice Exams**
- **Topic-Focused**: Generates practice exams based on your prioritized study topics
- **LaTeX Format**: Creates professional LaTeX documents ready for Overleaf compilation
- **Difficulty Control**: Choose difficulty level (easy, medium, hard)
- **Comprehensive Coverage**: Ensures all prioritized topics are included

### 🧠 **AI-Powered Analysis**
- **OpenAI GPT-4**: For practice exam generation and work analysis
- **Google Gemini**: For file processing, topic extraction, and LaTeX conversion
- **Mathpix**: For mathematical content extraction from images

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd seikai_hack

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your API keys
```

### 2. **API Keys Required**
```env
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
MATHPIX_APP_ID=your_mathpix_app_id_here
MATHPIX_APP_KEY=your_mathpix_app_key_here
```

### 3. **Run the Application**
```bash
python app.py
```

The app will be available at `http://localhost:8000`

## 📋 Usage Workflow

### **Step 1: Start Exam Session**
- Enter course name
- Create a new study session

### **Step 2: Enter Midterm Coverage**
- Enter the topics and concepts that will be on your midterm exam
- Simply type or copy-paste from your syllabus, study guide, or course materials
- AI extracts and organizes the main topics automatically

### **Step 3: Upload Practice Exam**
- Upload a practice exam (PDF or image)
- Set your target minimum score
- AI analyzes questions and assigns score values to topics

### **Step 4: Rate Your Confidence**
- For each topic, rate your confidence from 1-6
- 1 = Very Unconfident, 6 = Very Confident
- AI calculates study priorities using the scoring algorithm

### **Step 5: Upload Course Materials** (Optional)
- Upload textbooks, slides, homework, etc.
- Helps AI better understand your course content

### **Step 6: Upload Practice Work**
- Upload your practice work for instant feedback
- Get topic-specific recommendations

### **Results & Practice Exam Generation**
- View your prioritized study topics
- Generate new practice exams focused on your weak areas
- Download LaTeX files ready for Overleaf

## 🔧 Technical Details

### **Architecture**
- **FastAPI**: Modern, fast web framework
- **SQLAlchemy**: Database ORM
- **SQLite**: Lightweight database (can be changed to PostgreSQL)
- **Jinja2**: HTML templating

### **AI Services**
- **GeminiService**: Handles file processing and LaTeX conversion
- **GPTService**: Generates practice exams and analyzes work
- **PriorityQueueService**: Implements the study prioritization algorithm
- **MathpixService**: Extracts mathematical content from images

### **Database Models**
- **ExamSession**: Stores session data, exam coverage, and practice exam info
- **Topic**: Tracks topics with confidence scores and study priorities
- **Question**: Stores practice work and analysis results

## 📊 Study Priority Algorithm

The system uses a sophisticated algorithm to determine study priorities:

1. **Score Calculation**: `s_i = B_i * (6 - c_i)/6`
2. **Topic Sorting**: Topics sorted by calculated score (highest first)
3. **Priority Building**: Start with highest score, add topics until minimum score is reached
4. **Similar Score Handling**: When scores are within threshold, AI determines easier topic
5. **Study Order**: Final priority list shows optimal study sequence

## 🎨 UI Features

- **Red Theme**: Maintains the urgent, last-minute aesthetic
- **Progressive Disclosure**: Steps appear as you complete previous ones
- **Interactive Confidence Bars**: Click to set confidence levels (1-6)
- **Real-time Updates**: See priority changes as you adjust confidence
- **Responsive Design**: Works on desktop and mobile devices

## 📁 File Structure

```
seikai_hack/
├── app.py                 # Main FastAPI application
├── models.py             # Database models
├── database.py           # Database configuration
├── services/             # AI service modules
│   ├── gemini_service.py # Gemini integration
│   ├── gpt_service.py    # OpenAI GPT integration
│   ├── priority_queue.py # Study prioritization algorithm
│   └── mathpix_service.py # Mathpix integration
├── templates/            # HTML templates
├── static/              # CSS and static files
├── requirements.txt     # Python dependencies
└── env.example         # Environment variables template
```

## 🧪 Testing

Run the test script to verify functionality:

```bash
python test_functionality.py
```

## 🔒 Security Notes

- API keys are stored in environment variables
- File uploads are validated for type and size
- Database uses parameterized queries to prevent injection

## 🚧 Future Enhancements

- **Multi-language Support**: Support for different languages
- **Advanced Analytics**: Detailed performance tracking and insights
- **Collaborative Study**: Group study sessions and shared materials
- **Mobile App**: Native mobile application
- **Integration**: LMS integration (Canvas, Blackboard, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include error logs and steps to reproduce

---

**Built for students who need to study smart, not just hard! 🧠⚡**
