# Intelligent Document Query System (Retrieval Augmented Generation - RAG)

## Project Overview

This project develops an **Intelligent Document Query System** using the Retrieval Augmented Generation (RAG) pattern. The goal is to allow users to "chat" with a large text document, asking questions and receiving answers grounded in the document's content.

This initial phase (MVP) establishes the core backend RAG pipeline, capable of processing, indexing, and intelligently querying a substantial text corpus.

## What is RAG? (Retrieval Augmented Generation)

RAG enhances Large Language Models (LLMs) by giving them external, relevant information to generate more accurate and factual responses, reducing "hallucinations."

### How RAG Works:

**1. Document Ingestion (Making the Document Searchable):**

We prepare the document by:
*   **Loading:** Reading the raw `.txt` file.
*   **Splitting:** Breaking it into smaller, manageable chunks.
*   **Embedding:** Converting each chunk into numerical "embeddings" (vectors that capture meaning).
*   **Storing:** Saving these chunks and embeddings in a vector database (ChromaDB) for quick retrieval.

![RAG Pipeline Steps](screenshots/RAG_steps.png)

**2. Answering Queries (Getting Answers from the Document):**

When a question is asked:
*   The question is converted into an embedding.
*   **Retrieve:** We find the most relevant document chunks (based on similarity to the question) from our vector database.
*   **Prompt:** These relevant chunks, along with the question and chat history, are sent to the LLM.
*   **LLM Generates:** The LLM uses *only* the provided context to create an answer.
*   **Answer:** The response is given to the user.

![RAG Query Process](screenshots/RAG_process.png)

## Chosen Document

The system currently uses:

*   **[YOUR DOCUMENT TITLE HERE]** (e.g., "War and Peace" by Leo Tolstoy)
*   **Why this document?** [Briefly explain: e.g., "Its extensive detail makes it ideal for a RAG system to quickly extract specific facts and insights."]

This document should be placed in the `./data/` directory and be in `.txt` format.

## System Architecture (Current MVP - Backend)

The backend is built with modular Python components:

*   **`text_loader.py`**: Loads and chunks documents from the `data/` directory.
*   **`vector_store.py`**: Handles generating embeddings (`HuggingFaceEmbeddings`) and storing them persistently in **ChromaDB**. It also manages retrieving relevant chunks.
*   **`rag_system.py`** (or `retriever.py`): The main orchestrator. It initializes the LLM (`ChatGroq` for fast responses), integrates with the vector store, and manages conversational memory for multi-turn chats. It uses a "singleton" pattern to ensure efficient, one-time loading/indexing of documents.

### Key Technologies

*   **Langchain**: Orchestrates the RAG pipeline.
*   **Groq**: Provides a fast LLM (`llama-3.1-8b-instant`).
*   **ChromaDB**: Persistent vector database (embeddings saved to disk).
*   **HuggingFaceEmbeddings**: Embeddings model (`sentence-transformers/all-MiniLM-L6-v2`).

## Current Status: Phase 1 (Backend Complete)

The core RAG backend is fully functional. It can:

*   Process and index your large `.txt` document.
*   Store embeddings persistently, so you only index once.
*   Answer questions about the document, maintaining chat history.
*   Start up quickly after the initial indexing.

## Getting Started (Local Backend Test)

Follow these steps to set up and test the RAG backend on your machine.

### Prerequisites

*   Python 3.9+
*   `pip`
*   Git

### Setup Steps:

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL_HERE]
    cd [YOUR_PROJECT_FOLDER_NAME]
    ```
2.  **Create & Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # macOS/Linux: source venv/bin/activate
    # Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes: `langchain`, `langchain-chroma`, `langchain-groq`, `sentence-transformers`, `chromadb`, `tiktoken`, `python-dotenv`)*
4.  **Place Your Document:**
    Put your chosen `.txt` document (e.g., `my_novel.txt`) into the `./data/` folder in your project root.
5.  **Configure `.env` File:**
    Create a file named `.env` in your project's root. **Add `.env` to your `.gitignore`!**
    ```env
    # .env
    DATA_DIRECTORY=./data
    CHROMA_PERSIST_DIRECTORY=./chroma_db
    EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    LLM_MODEL_NAME=llama-3.1-8b-instant
    LLM_TEMPERATURE=0.7
    RETRIEVAL_K=4
    GROQ_API_KEY=YOUR_GROQ_API_KEY_HERE # Get yours from console.groq.com/keys
    ```
6.  **Run the Test Script:**
    The first run will index your document (may take a few minutes). Subsequent runs will be fast.
    ```bash
    python test_rag.py
    ```
    You'll see messages confirming initialization, document loading, and chat interactions.

## Next Steps

*   **Phase 2: Streamlit Frontend**: Build the interactive user interface.
*   **Phase 3: Enhancements**: Add features like document upload.
*   **Phase 4 & 5: Docker & Deployment**: Containerize and deploy the app to a VPS.

## Project Structure

```
LLM_LargeDocumentInteraction/
├── 📁 backend/
│   ├── __init__.py
│   ├── simple_rag.py         # 🧠 Main RAG orchestration & chat logic
│   ├── vector_store.py       # 🗄️ ChromaDB vector database management
│   └── text_loader.py        # 📄 Document loading & text chunking
├── 📁 data/                  # 📚 Your .txt documents go here
│   ├── .gitkeep             # (Placeholder - add your documents)
│   └── your_document.txt    # Example: place your text files here
├── 📁 chroma_db/            # 🗄️ Vector database (auto-created)
│   └── .gitkeep             # (Auto-generated embeddings stored here)
├── 📄 test_backend.py       # 🧪 Backend testing script
├── 📄 .env                  # ⚙️ Configuration file (create from .env.example)
├── 📄 .env.example         # 📋 Template for environment variables
├── 📄 requirements.txt     # 📦 Python dependencies
├── 📄 .gitignore           # 🚫 Files to exclude from git
└── 📄 README.md           # 📖 This documentation
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `simple_rag.py` | Main RAG system with conversation memory and singleton pattern |
| `vector_store.py` | Manages ChromaDB operations, embeddings, and document indexing |
| `text_loader.py` | Handles document loading, text splitting, and file management |
| `test_backend.py` | Comprehensive testing script to validate the entire pipeline |
| `.env` | Configuration file with API keys and settings (never commit this!) |

## Usage Examples

### Basic Chat
```python
from backend.simple_rag import chat_with_documents

# Ask questions about your documents
response = chat_with_documents("What is the main theme of this document?")
print(response)

# Follow-up questions (with memory)
response = chat_with_documents("Can you elaborate on that theme?")
print(response)
```

### Advanced Usage
```python
from backend.simple_rag import get_rag_instance, clear_chat_history, reset_rag_system

# Get the RAG instance (singleton pattern)
rag = get_rag_instance()

# Check system status
doc_count = rag.get_document_count()
print(f"📄 Loaded {doc_count} document chunks")

# Clear conversation history (keeps documents loaded)
clear_chat_history()

# Reset entire system (reload everything)
reset_rag_system()
```

## Requirements

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for large documents)
- **Storage**: 1GB free space for embeddings and dependencies

### API Requirements
- **Groq API Key**: Free tier available at [console.groq.com](https://console.groq.com)
  - Sign up for free account
  - Generate API key from dashboard
  - Add to `.env` file

### Dependencies
All required packages are in `requirements.txt`:
```
langchain
langchain-groq
langchain-community
chromadb
sentence-transformers
python-dotenv
```

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| 🔴 `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| 🔴 `GROQ_API_KEY missing` | Add your API key to `.env` file |
| 🔴 `No documents found` | Put `.txt` files in `./data/` directory |
| 🔴 `Permission denied` on ChromaDB | Check write permissions on project directory |
| 🔴 `Out of memory` | Use smaller `CHUNK_SIZE` in `.env` |

### Performance Tips

- **First run is slow**: Document indexing takes time, subsequent runs are fast
- **Large documents**: Increase `CHUNK_SIZE` to 1500-2000 for better context
- **Better answers**: Use `RETRIEVAL_K=6` to get more context chunks
- **Faster responses**: Reduce `LLM_TEMPERATURE` to 0.3 for more focused answers

### Debug Mode

Enable verbose logging by modifying your test:
```python
# In test_backend.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Configuration Options

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *Required* | Your Groq API key for LLM access |
| `DATA_DIRECTORY` | `./data` | Directory containing your `.txt` documents |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | Where vector embeddings are stored |
| `LLM_MODEL_NAME` | `llama-3.1-8b-instant` | Groq model to use |
| `LLM_TEMPERATURE` | `0.7` | Response creativity (0.0-1.0) |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size for processing |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `4` | Number of relevant chunks to retrieve |

### Customization

**For different document types:**
- Academic papers: `CHUNK_SIZE=1500, RETRIEVAL_K=6`
- Novels/stories: `CHUNK_SIZE=800, RETRIEVAL_K=4`
- Technical docs: `CHUNK_SIZE=1200, RETRIEVAL_K=5`

**For different response styles:**
- Focused answers: `LLM_TEMPERATURE=0.3`
- Creative responses: `LLM_TEMPERATURE=0.8`
- Balanced: `LLM_TEMPERATURE=0.7` (default)

## Testing

### Run Full Test Suite
```bash
python test_backend.py
```

### Expected Test Output
```
--- RAG System Test ---
GROQ_API_KEY: ✅ Set
DATA_DIRECTORY: ./data
CHROMA_PERSIST_DIRECTORY: ./chroma_db

🚀 Initializing RAG system...
✅ RAG system ready! Indexed 156 document chunks.

==================================================
Query 1:
Q: What is the main theme of the document?
A: [AI-generated response based on your document]

==================================================
Query 2 (with memory):
Q: Can you elaborate on that?
A: [Follow-up response with conversation context]

✅ --- Test Complete ---
```

## Development

### Contributing
1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature-name`
3. **Make** your changes
4. **Test** with: `python test_backend.py`
5. **Submit** pull request

### Code Style
- Follow **PEP 8** formatting
- Add **docstrings** to new functions
- Keep **error handling** simple for MVP
- Test all changes with the test script

### Adding New Features
- **Document types**: Extend `text_loader.py`
- **LLM providers**: Modify `simple_rag.py`
- **Vector stores**: Update `vector_store.py`
- **New functionality**: Always add tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

### Get Help
- 📖 **Documentation**: Read this README thoroughly
- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: Create a feature request issue
- 📧 **Direct Contact**: [your-email@example.com]

### FAQ

**Q: Can I use PDF documents?**
A: Currently only `.txt` files are supported. PDF support is planned for Phase 3.

**Q: How large can my documents be?**
A: The system efficiently handles documents up to 1M+ words. For larger documents, consider splitting them.

**Q: Can I use different LLM providers?**
A: Yes! The system is designed to be LLM-agnostic. You can easily swap Groq for OpenAI, Anthropic, or local models.

**Q: Is my data secure?**
A: Yes! All processing happens locally. Only queries (not documents) are sent to the LLM API.

## Deployed Application URL

*(Will be updated in Phase 5 after deployment)*

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**Built with ❤️ using LangChain, Groq, and ChromaDB**

[🔝 Back to top](#intelligent-document-query-system-retrieval-augmented-generation---rag)

</div>
