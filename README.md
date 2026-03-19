# Offline-MultiRag-System

A powerful **offline multi-modal RAG (Retrieval-Augmented Generation) system** that combines document search with advanced image analysis capabilities. Built with Ollama's LLaVA vision model, this system can understand and analyze technical diagrams, blueprints, and schematics from PDF documents without requiring internet connectivity.

## 🎯 Project Overview

This system is designed for analyzing complex technical documents like ship manuals, engineering blueprints, and operational guides. It leverages:
- **Vector-based semantic search** for intelligent document retrieval
- **Multi-modal analysis** combining text and image understanding
- **Vision-based diagram interpretation** for technical drawings
- **Automatic image detection** to understand user intent
- **Session-based context tracking** for intelligent follow-up questions

Perfect for maritime, engineering, and technical documentation domains.

## ✨ Key Features

### 1. **Multi-Modal RAG Architecture**
- Combines text retrieval with vision-based analysis
- Sends relevant document context + images together to LLaVA
- Intelligent fallback from image analysis to text search

### 2. **Automatic Image Detection**
- Detects when users are asking about images/diagrams
- Regex-based pattern matching for 13+ image query indicators
- No manual image filename specification needed
- Tracks shown images per user session

### 3. **Advanced Image Analysis**
- **Basic Image Interpretation**: General image analysis and explanation
- **Diagram Element Highlighting**: Identifies and highlights technical components
- **Blueprint Component Analysis**: Deep technical analysis of specific components
- **Detailed Image Interpretation**: Comprehensive multi-paragraph analysis

### 4. **Markdown-First Output**
- All responses formatted in clean, readable markdown
- Structured headings, lists, and code blocks
- Professional formatting for technical content
- Frontend rendering with Marked.js for beautiful display

### 5. **Session-Based Context Tracking**
- Remembers images shown in conversation
- Intelligently uses context for follow-up questions
- Prevents image relevance issues in multi-turn conversations
- Per-user image session management

### 6. **User Authentication**
- JWT-based authentication system
- Secure user accounts and session management
- Protected API endpoints

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │
│  (HTML/CSS/JS)  │
└────────┬────────┘
         │
    HTTP API
         │
┌────────▼────────────────┐
│   FastAPI Backend       │
├─────────────────────────┤
│ • User Auth (JWT)       │
│ • Session Management    │
│ • Request Routing       │
│ • Image Detection       │
└────────┬────────────────┘
         │
    ┌────┴──────────┬──────────────┐
    │               │              │
┌───▼───┐    ┌─────▼──────┐   ┌──▼──────┐
│ FAISS │    │   LLaVA    │   │ PyPDF   │
│Vector │    │  Vision    │   │ Text    │
│Search │    │  Model     │   │Extract  │
└───────┘    └────────────┘   └─────────┘
```

## 🚀 Installation Guide

### Prerequisites
- **Python 3.10+**
- **Ollama** installed and running
- **Git** for version control
- Windows/Linux/Mac compatible

### Step 1: Clone the Repository

```bash
git clone https://github.com/sushan2006/Offline-MultiRag-System.git
cd Offline-MultiRag-System
```

### Step 2: Create a Python Virtual Environment (Optional but Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

**🖥️ Using an NVIDIA GPU (RTX 5050, etc.)?** Install PyTorch with CUDA **first**, then the rest:

```bash
# 1. PyTorch with CUDA (check nvidia-smi for your CUDA version, use cu121/cu124/cu126)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Then install project dependencies
pip install -r requirements.txt

# 3. Verify GPU
python verify_gpu.py
```

See **[SETUP_GPU.md](SETUP_GPU.md)** for full GPU setup instructions. Without this, embeddings run on CPU.

**CPU-only install:**
```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `ollama` - LLaVA model integration
- `sentence-transformers` - Text embeddings
- `faiss-cpu` - Vector similarity search
- `pypdf` - PDF document extraction
- `numpy` - Numerical computing
- `opencv-python` - Image processing
- `Pillow` - Image manipulation
- `passlib` - Password hashing
- `python-jose` - JWT authentication
- `python-multipart` - File upload support

### Step 4: Install and Configure Ollama

1. **Download Ollama** from [ollama.ai](https://ollama.ai)

2. **Install the LLaVA model:**

```bash
ollama pull llava
```

**Alternative vision models:**
- `ollama pull bakllava` - Smaller, faster alternative
- `ollama pull minicpm-v` - Another capable model

3. **Start Ollama server** (in a separate terminal):

```bash
ollama serve
```

**Ollama API** will be available at `http://localhost:11434`

### Step 5: Prepare Your Documents

1. Create a `documents/` folder in the project root
2. Add your PDF files to the `documents/` folder

Example:
```
documents/
├── ship_manual.pdf
├── engine_operations.pdf
└── safety_procedures.pdf
```

### Step 6: Start the Backend Server

```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Step 7: Open the Frontend

1. Open a web browser
2. Navigate to `http://localhost:8000` or `http://127.0.0.1:8000`
3. Sign up for an account or log in
4. Start asking questions!

## 📖 Usage Guide

### 1. Upload Documents
- Click the "Upload" tab
- Select PDF files from your local system
- System extracts text and images automatically

### 2. Ask Questions

#### Text-based Questions:
```
"What are the engine specifications?"
"How do I perform routine maintenance?"
```

#### Image/Diagram Questions:
The system automatically detects these patterns:
```
"Explain this diagram"
"What does the highlighted part mean?"
"Analyze the blueprint components"
"Describe the schematic above"
"What does the image show?"
```

#### Follow-up Questions:
```
User: "Show me the engine diagram"
System: Displays diagram
User: "What's the highlighted part in that image?"
System: Understands you mean the previously shown image
```

### 3. View Results
- Responses are formatted in clean markdown
- Images are displayed inline with analysis
- Context preserved across conversation turns

## 🔌 API Endpoints

### Authentication
```
POST /api/auth/signup
POST /api/auth/login
```

### Core Endpoints

#### 1. Ask Question (Auto-detect: Text or Image)
```
POST /ask
Content-Type: application/json

{
  "user_id": "user123",
  "question": "What does this diagram show?",
  "last_image": "ENGINE_engine_operations.pdf_page0_0.jpx"  // Optional
}

Response:
{
  "answer": "The diagram shows...",
  "images": ["path/to/image1.png"],
  "confidence": 0.95
}
```

#### 2. Analyze Diagram (Explicit Image Analysis)
```
POST /analyze-diagram
Content-Type: application/json

{
  "user_id": "user123",
  "image_path": "ENGINE_engine_operations.pdf_page0_0.jpx"
}

Response:
{
  "analysis": "Detailed diagram analysis...",
  "elements": ["Component A", "Component B"],
  "highlighted_image": "base64_encoded_image"
}
```

#### 3. Analyze Blueprint Component
```
POST /analyze-component
Content-Type: application/json

{
  "user_id": "user123",
  "image_path": "blueprint.pdf_page1_0.png",
  "component_name": "Valve Assembly"
}

Response:
{
  "component_analysis": "Technical specifications...",
  "specifications": {...}
}
```

#### 4. Detailed Image Interpretation
```
POST /interpret-image-detailed
Content-Type: application/json

{
  "user_id": "user123",
  "image_path": "schematic.pdf_page2_0.png"
}

Response:
{
  "detailed_interpretation": "Comprehensive multi-paragraph analysis...",
  "technical_summary": {...}
}
```

## 📊 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI + Uvicorn | REST API Server |
| **Vision Model** | Ollama + LLaVA | Image Analysis |
| **Vector Search** | FAISS + Sentence Transformers | Semantic Search |
| **Document Processing** | PyPDF | PDF Extraction |
| **Authentication** | JWT + OAuth2 | User Security |
| **Image Processing** | OpenCV + Pillow | Image Manipulation |
| **Frontend** | HTML/CSS/JavaScript | User Interface |
| **Markdown Rendering** | Marked.js | Pretty Text Display |

## ⚙️ Configuration

### Environment Variables (Optional)

Create a `.env` file in the `backend/` folder:

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llava

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=sqlite:///./test.db
```

### Model Selection

Edit `backend/rag_engine.py` to change the vision model:

```python
# Line ~30
MODEL_NAME = "llava"  # Change to: bakllava, minicpm-v, etc.
```

## 🔍 Troubleshooting

### Issue: "Cannot connect to Ollama"
**Solution:**
```bash
# Make sure Ollama is running in another terminal
ollama serve

# Verify it's accessible
curl http://localhost:11434/api/status
```

### Issue: "Model not found"
**Solution:**
```bash
# Pull the model
ollama pull llava

# List available models
ollama list
```

### Issue: "Permission denied" on files
**Solution:**
- Ensure the `extracted_images/` folder has write permissions
- Windows: Right-click folder → Properties → Security → Edit

### Issue: PDF files not processing
**Solution:**
- Ensure PDFs are not encrypted
- Try converting PDF to ensure compatibility
- Check file permissions

### Issue: Images not showing in response
**Solution:**
- Verify extracted images exist in `backend/extracted_images/`
- Check browser console for errors (F12)
- Ensure image paths are correct in responses

## 📁 Project Structure

```
Offline-MultiRag-System/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── backend/
│   ├── main.py              # FastAPI application
│   ├── rag_engine.py        # RAG + Vision model logic
│   ├── db.py                # Database operations
│   └── extracted_images/    # Automatically extracted images
│
├── frontend/
│   ├── chat.html            # Chat interface
│   ├── login.html           # Login page
│   ├── signup.html          # Signup page
│   ├── upload.html          # Document upload
│   └── styles.css           # Styling
│
└── documents/               # Your PDF files (add here)
    └── (your-documents.pdf)
```

## 🎓 Example Workflows

### Workflow 1: Basic Question and Answer
```
1. User: "What are safety procedures?"
2. System: ✓ Searches documents
3. System: ✓ Returns markdown-formatted answer
4. User reads response
```

### Workflow 2: Diagram Analysis
```
1. User: "Show me the engine diagram"
2. System: ✓ Searches and displays diagram
3. User: "Explain the highlighted components"
4. System: ✓ Remembers shown image
5. System: ✓ Analyzes with LLaVA
6. System: ✓ Returns detailed explanation
```

### Workflow 3: Multi-Turn Technical Analysis
```
1. User: "Load the maintenance schedule blueprint"
2. System: ✓ Displays blueprint image
3. User: "What does this component do?"
4. System: ✓ Uses context from shown image
5. System: ✓ Provides technical analysis
6. User: "Highlight the critical components"
7. System: ✓ Returns marked-up image
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👨‍💻 Author

**Sushan** - Multi-modal AI Systems Developer

## 📧 Support

For issues, questions, or feature requests:
- Open an GitHub Issue
- Check the Troubleshooting section
- Review API documentation

## 🎯 Future Enhancements

- [ ] Support for DOCX and other document formats
- [ ] Real-time collaborative editing
- [ ] Advanced image annotation tools
- [ ] Custom model training
- [ ] Cloud deployment guides
- [ ] Mobile app support
- [ ] Multi-language support
- [ ] Advanced analytics dashboard

## ⭐ Acknowledgments

- **Ollama** for the LLaVA vision model
- **FastAPI** for the excellent web framework
- **FAISS** for vector similarity search
- **Sentence Transformers** for embeddings

---

**Last Updated:** February 2026

**Status:** ✅ Fully Functional and Production Ready
