# 📚 Large Document Interaction AI

A sophisticated RAG (Retrieval-Augmented Generation) system that allows users to interact with large documents through an AI assistant. Upload documents, ask questions, and get intelligent responses based on the document content.

## 🚀 Features

- **Multi-Document Support**: Upload and manage multiple documents
- **Persistent Storage**: Documents and chat histories are saved automatically
- **Smart Chunking**: Documents are intelligently split for optimal retrieval
- **Interactive Chat**: Natural conversation interface with document-specific context
- **Document Library**: Easy switching between uploaded documents
- **Chat History**: Persistent conversation history per document

## 🏗️ Project Structure

```
LLM_LargeDocumentInteraction/
├── main.py                 # Backend RAG logic (RAGBackend class)
├── frontend/
│   └── app.py             # Streamlit frontend interface
├── text_loader.py         # Document loading utilities
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── chroma_db/            # Persistent vector storage (auto-created)
├── data/                 # Metadata and chat histories (auto-created)
│   ├── uploaded_documents.json
│   └── chat_histories.json
└── Test_files/           # Sample documents
```

## ⚙️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here  # Optional
```

### 3. Run the Application
```bash
# Run the Streamlit frontend
streamlit run frontend/app.py

# Or test the backend directly
python main.py
```

## 📄 Reference
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
