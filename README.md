# 🎓 Epoka University Chatbot

An intelligent AI-powered chatbot that answers questions about Epoka University using Retrieval-Augmented Generation (RAG). The chatbot provides accurate information about admissions, programs, scholarships, academic calendar, tuition fees, and more.

## 📋 Project Overview

This chatbot uses a hybrid retrieval system combining:
- **Semantic Search**: Using sentence transformers for meaning-based retrieval
- **BM25**: For keyword-based matching
- **FAISS**: For fast similarity search across document embeddings

The system processes university documents (PDFs and FAQs) and provides contextual answers with source citations.

## 🛠️ Technologies Used

- **Python 3.10+**
- **FastAPI** - REST API backend
- **Gradio** - Web-based chat interface
- **LangChain** - Document processing and chunking
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **FAISS** - Vector similarity search
- **BM25** - Keyword-based ranking

## 📁 Project Structure

```
school-chatbot/
├── app.py                  # Main application (FastAPI + Gradio)
├── run.py                  # Alternative runner script
├── requirements.txt        # Python dependencies
├── test_questions.py       # Test script for chatbot
├── test_setup.py           # Setup verification script
├── data/
│   ├── raw_pdfs/           # Source PDF documents
│   ├── faqs/
│   │   └── faqs.txt        # FAQ document
│   └── processed/
│       └── epoka_index.faiss  # Pre-built FAISS index
└── src/
    ├── __init__.py
    ├── config.py           # Configuration settings
    ├── data_loader.py      # Document loading and chunking
    ├── embedding_manager.py # Embedding creation and FAISS indexing
    ├── retrieval.py        # Hybrid retrieval (semantic + BM25)
    └── response_generator.py # Answer generation
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/stinarifi03/school_chatbot.git
   cd school_chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional, for OpenAI features)
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY if needed
   ```

## 📄 Adding Documents

Place your university documents in the appropriate folders:

- **PDFs**: `data/raw_pdfs/` - Add PDF documents about the university
- **FAQs**: `data/faqs/faqs.txt` - Add Q&A pairs in the specified format

The chatbot will automatically process these documents on startup.

## ▶️ Running the Application

### Option 1: Gradio Interface (Recommended)
```bash
python app.py
```
Then open your browser to `http://localhost:7860`

### Option 2: FastAPI Only
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/info` | GET | Index information |
| `/chat` | POST | Submit a question |

### Example API Request
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What scholarships are available?"}'
```

### Response Format
```json
{
  "answer": "Epoka University offers several scholarship types...",
  "citations": [
    {"source": "faqs.txt", "page": 1, "snippet": "..."}
  ],
  "performance": {
    "response_time": 0.234
  },
  "university_info": {
    "name": "Epoka University",
    "suggested_contact": "admissions@epoka.edu.al"
  }
}
```

## 💬 Example Questions

- "When does the Fall semester 2025 begin?"
- "What are the tuition fees for international students?"
- "How do I apply for a Master's program?"
- "What scholarships are available?"
- "What CGPA must I maintain to keep my scholarship?"
- "When is the graduation ceremony?"

## ⚙️ Configuration

Key settings in `src/config.py`:

| Setting | Value | Description |
|---------|-------|-------------|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `CHUNK_SIZE` | 500 | Document chunk size |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K` | 8 | Number of retrieved chunks |
| `SCORE_THRESHOLD` | 0.25 | Minimum similarity score |

## 🧪 Testing

Run the test scripts to verify the setup:

```bash
# Test setup
python test_setup.py

# Test questions
python test_questions.py
```

## 📊 Features

- ✅ Hybrid retrieval (semantic + keyword search)
- ✅ Source citations with page numbers
- ✅ Response time tracking
- ✅ Pre-built FAISS index for fast startup
- ✅ Web-based chat interface (Gradio)
- ✅ REST API (FastAPI)
- ✅ PDF and FAQ document support

## 👥 Authors

- **Samuel Troci** - CEN352 Term Project 2025-26

## 📝 Course Information

- **Course**: CEN352 - Software Engineering
- **Institution**: Epoka University
- **Academic Year**: 2025-2026

## 📜 License

This project is created for educational purposes as part of the CEN352 course at Epoka University.
