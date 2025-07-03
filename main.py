import os
import datetime
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Loading environment variables from a .env file
load_dotenv()

# Setting environment variables for LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"

class RAGPipeline:
    """Backend class for handling RAG operations with persistent storage"""
    
    def __init__(self):
        # Configuration
        self.chroma_db_dir = "chroma_db"
        self.data_dir = "data"
        self.doc_metadata_file = os.path.join(self.data_dir, "uploaded_documents.json")
        self.chat_history_file = os.path.join(self.data_dir, "chat_histories.json")
        
        # Create directories
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Validate API keys
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize models
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()
        
    def _init_llm(self):
        """Initialize the language model"""
        return ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.7
        )
    
    def _init_embeddings(self):
        """Initialize the embeddings model"""
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Document & Chat History Management Methods
    def load_documents_metadata(self) -> Dict[str, str]:
        """Load metadata of all uploaded documents"""
        if os.path.exists(self.doc_metadata_file):
            with open(self.doc_metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_documents_metadata(self, metadata: Dict[str, str]):
        """Save metadata of all uploaded documents"""
        with open(self.doc_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def load_chat_histories(self) -> Dict[str, List[Dict]]:
        """Load all chat histories"""
        if os.path.exists(self.chat_history_file):
            with open(self.chat_history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_chat_histories(self, histories: Dict[str, List[Dict]]):
        """Save all chat histories"""
        with open(self.chat_history_file, 'w') as f:
            json.dump(histories, f, indent=4)
    
    def get_document_hash(self, file_content_bytes: bytes) -> str:
        """Generate a unique MD5 hash for the file content"""
        return hashlib.md5(file_content_bytes).hexdigest()
    
    def process_and_store_document(self, file_content_bytes: bytes, original_filename: str) -> str:
        """
        Process an uploaded document, create a new ChromaDB collection for it,
        and update the metadata.
        """
        doc_id = self.get_document_hash(file_content_bytes)
        
        # Check if document already exists
        docs_metadata = self.load_documents_metadata()
        if doc_id in docs_metadata:
            print(f"Document '{original_filename}' already exists.")
            return doc_id
        
        # Convert bytes to string and create document
        string_data = file_content_bytes.decode("utf-8")
        uploaded_document = [Document(page_content=string_data, metadata={"source": original_filename})]
        
        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(uploaded_document)
        
        # Create new ChromaDB collection
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=doc_id,
            persist_directory=self.chroma_db_dir
        )
        vectorstore.persist()
        
        # Update document metadata
        docs_metadata[doc_id] = original_filename
        self.save_documents_metadata(docs_metadata)
        
        print(f"Document '{original_filename}' processed and stored with ID: {doc_id}")
        return doc_id
    
    def load_vectorstore_by_id(self, doc_id: str):
        """Load an existing ChromaDB collection by its document ID"""
        return Chroma(
            collection_name=doc_id,
            embedding_function=self.embeddings,
            persist_directory=self.chroma_db_dir
        )
    
    def create_rag_qa_chain(self, vectorstore):
        """Create a RAG QA chain with the given vector store"""
        rag_prompt_template = """You are an AI assistant that helps users interact with a large document.
        Use the following context to answer the user's question.
        If the answer is not in the context, politely state that you cannot find the information in the provided document.
        
        Context:
        {context}

        Question: {question}
        
        Answer:"""
        
        rag_prompt = PromptTemplate.from_template(rag_prompt_template)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": rag_prompt}
        )
        return qa_chain
    
    def query_document(self, doc_id: str, question: str) -> str:
        """Query a specific document with a question"""
        try:
            vectorstore = self.load_vectorstore_by_id(doc_id)
            qa_chain = self.create_rag_qa_chain(vectorstore)
            response = qa_chain.invoke({"query": question})
            return response.get("result", "Sorry, I couldn't find an answer.")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_available_documents(self) -> Dict[str, str]:
        """Get all available documents"""
        return self.load_documents_metadata()
    
    def get_chat_history(self, doc_id: str) -> List[Dict]:
        """Get chat history for a specific document"""
        histories = self.load_chat_histories()
        return histories.get(doc_id, [])
    
    def save_message(self, doc_id: str, role: str, content: str):
        """Save a message to chat history"""
        histories = self.load_chat_histories()
        if doc_id not in histories:
            histories[doc_id] = []
        histories[doc_id].append({"role": role, "content": content})
        self.save_chat_histories(histories)
    
    def clear_chat_history(self, doc_id: str):
        """Clear chat history for a specific document"""
        histories = self.load_chat_histories()
        if doc_id in histories:
            del histories[doc_id]
            self.save_chat_histories(histories)

# Create a global instance for use by frontend
def get_rag_backend():
    """Get or create RAG backend instance"""
    if not hasattr(get_rag_backend, "instance"):
        get_rag_backend.instance = RAGBackend()
    return get_rag_backend.instance

if __name__ == "__main__":
    # Example usage (without requiring API keys for demo)
    print("🚀 RAG Backend Structure:")
    print("- Document processing and storage")
    print("- Persistent vector database with ChromaDB")
    print("- Chat history management")
    print("- Multi-document support")
    print()
    print("To use with real API:")
    print("1. Set GROQ_API_KEY in .env file")
    print("2. Run: streamlit run frontend/app.py")
    print()
    print("Project structure created successfully! ✅")
    
    # Show available test files
    test_files_dir = "Test_files"
    if os.path.exists(test_files_dir):
        print(f"\n📁 Available test files in {test_files_dir}:")
        for file in os.listdir(test_files_dir):
            if file.endswith('.txt'):
                print(f"  - {file}")
    else:
        print(f"\n📁 Test files directory not found: {test_files_dir}")
    
    # Create necessary directories
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    print("\n📂 Created necessary directories: chroma_db/, data/")
    
    print("\n🎯 Next steps:")
    print("1. Add your GROQ_API_KEY to .env file")
    print("2. Run: streamlit run frontend/app.py")
    print("3. Upload documents and start chatting!")

