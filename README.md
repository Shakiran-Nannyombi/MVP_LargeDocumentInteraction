# LARGE TEXT FILE RETRIEVAL RAG MVP

![APP User View](screenshots/app.png)

[🌐 **To view App Live tap this link**](https://largetextmvp.duckdns.org/)


## 🚀 Project Overview

Welcome to the **Intelligent Document Query System**!  
This project lets you chat with large text documents using Retrieval Augmented Generation (RAG). Upload a `.txt` file, ask questions, and get answers grounded in your document's content—no more hallucinations, just real context.

## 🤖 What is RAG? (Retrieval Augmented Generation)

RAG supercharges Large Language Models (LLMs) by giving them access to external, relevant information, making responses more accurate and factual.

### How RAG Works:

1. **Document Ingestion**
   - **Load:** Read your `.txt` file.
   - **Split:** Break it into manageable chunks.
   - **Embed:** Convert each chunk into a vector (embedding).
   - **Store:** Save embeddings in a vector database (ChromaDB).

   ![RAG Pipeline Steps](screenshots/RAG_steps.png)

2. **Question Answering**
   - **Embed Query:** Your question is embedded.
   - **Retrieve:** Find the most relevant chunks from ChromaDB.
   - **Augment Prompt:** Combine retrieved context, your question, and chat history.
   - **LLM Response:** The LLM answers using only the provided context.

   ![RAG Query Process](screenshots/RAG_process.png)

---

## Project Structure

```
LLM_LargeDocumentInteraction/
│
├── app.py                # Streamlit app (main UI and logic)
├── rag_system.py         # RAG backend: chunking, embedding, retrieval
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Docker config for ChromaDB and app.py
├── Dockerfile            # Dockerfile for app.py
├── .env                  # Environment variables (API keys, config)
├── .env.example          # Environment variables template
├── README.md             # This file!
│
├── chroma_db/            # ChromaDB vector database for persistent storage
│   └── chroma.sqlite3
│
├── data/                 # Storage directory for uploaded documents
│   └── *.txt             # User uploaded text files
│
├── chats/                # Saved chat histories in JSON format, per document
│   └── *.json            # Chat history files with timestamps
│
├── screenshots/          # Documentation images
│   ├── app.png           # App interface screenshot
│   ├── RAG_steps.png     # RAG pipeline visualization
│   └── RAG_process.png   # RAG query process diagram
│
└── .venv/                # Python virtual environment (not tracked in git)
    └── ...               # Virtual environment files
```

## HOW TO SET-UP

### 1. **Clone the repo and install dependencies**
```bash
git clone https://github.com/yourusername/LLM_LargeDocumentInteraction.git
cd LLM_LargeDocumentInteraction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. **Set up your `.env` file**
```env
GROQ_API_KEY=your-groq-key
GROQ_MODEL=llama-3.1-8b-instant
MISTRALAI_API_KEY=your-mistral-key
# ...other keys as needed
```

### 3. **Start ChromaDB (in Docker)**

`docker run -p 8083:8000 -v ./chroma_db:/chroma/chroma chromadb/chroma:latest`


### 4. **Run the app**

`streamlit run app.py --server.port 8502 --server.address 0.0.0.0`

Visit [http://localhost:8502](http://localhost:8502) (or on your own VPS IP/domain).


## 🚀 Production Deployment (VPS/Cloud)

- **Environment:** Deployed on a VPS using AWS EC2 and Ubuntu 22.04 plus Docker Compose.
- **Reverse Proxy & HTTPS:** Nginx is set up as a reverse proxy with Certbot for HTTPS.
- **Domain:** App is accessible via a custom domain (e.g., `https://largetextmvp.duckdns.org/`).
- **Key Steps:**
  1. Open ports 80, 443 (Nginx), and 8502 (Streamlit) in your VPS firewall/security group.
  2. Copy project files and `.env` to the server (e.g., with `scp`).
  3. Run `docker compose up -d --build` to start the app and ChromaDB.
  4. Configure Nginx for SSL and WebSocket support (see sample config in repo or deployment notes).
  5. Use Certbot to obtain and auto-renew HTTPS certificates.
- **Troubleshooting:**
  - Check logs: `docker logs <container_name>`
  - Restart: `docker compose restart` or `sudo systemctl reload nginx`

## Features

- **Chat with any large `.txt` document**
- **Fast, chunked document processing with progress bar**
- **Persistent chat history (save/load/delete)**
- **Runs locally or on a VPS**
- **ChromaDB for vector search (runs in Docker)**
- **Environment-based configuration management**
- **Docker containerization for easy deployment**
- **Real-time document processing with progress tracking**

---

## 🙏 Credits

- Built by Kiran
- Referenced from Langchain, Chroma and Streamlit Documentations


