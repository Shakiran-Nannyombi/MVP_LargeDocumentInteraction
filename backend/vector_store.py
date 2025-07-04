import os
from typing import List, Optional
from .text_loader import TextFileManager
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document

class VectorStore:

    # Function to initialize the VectorStore class
    def __init__(self):
        self.text_manager = TextFileManager()
        
        # Import constants from .env file
        self.chroma_persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "6"))

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
        
        # Initialize vector store (will load existing or create new)
        self.vector_store = self._initialize_vector_store()

    # Function to initialize or load the vector store
    def _initialize_vector_store(self) -> Chroma:
        if not os.path.exists(self.chroma_persist_directory):
            os.makedirs(self.chroma_persist_directory)
        
        vector_store = Chroma(
            persist_directory=self.chroma_persist_directory, 
            embedding_function=self.embedding_model
        )
        return vector_store

    # Function to clear the vector store
    def clear_vector_store(self):
        collection = self.vector_store._collection
        collection.delete()

    # Function to add documents to the vector store
    def add_documents(self, documents: List[Document]) -> bool:
        if not documents:
            return False
        
        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            print(f"Added {len(documents)} documents to the vector store.")
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False

    # Function to search the vector store for similar document objects
    def search(self, query: str) -> Optional[List[Document]]:
        if not query or not query.strip():
            return None
        
        results = self.vector_store.similarity_search(query, k=self.retrieval_k)
        return results

    # Function to get retriever for the vector store
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})

    # Function to load documents from text manager and add them to vector store
    def load_and_index_documents(self, force_reload: bool = False) -> bool:
        pass
        try:
            # CRUCIAL EFFICIENCY: Check if already indexed
            if not force_reload and self.get_document_count() > 0:
                print("Documents already indexed. Skipping.")
                return True
            
            documents = self.text_manager.load_and_split()
            if not documents:
                print("No documents found to index.")
                return False
            
            return self.add_documents(documents)
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False

    # Function to get the number of documents in the vector store
    def get_document_count(self) -> int:
        try:
            collection = self.vector_store._collection
            return collection.count()
        except:
            return 0

    # Function to force clear and re-index all documents
    def force_reindex(self) -> bool:
        self.clear_vector_store()
        return self.load_and_index_documents(force_reload=True)
