import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from .vector_store import VectorStore
from .text_loader import TextFileManager

class SimpleRAG:

    # Initializing the main class to connect all components
    def __init__(self):
        self.vector_store = VectorStore()
        self.text_manager = TextFileManager()
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        # Initializing memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key='answer'
        )
        # My custom prompt for question answering
        self.qa_prompt = self._create_custom_prompt()
        self.chat_chain = self._create_chat_chain()
    
    def _create_custom_prompt(self) -> PromptTemplate:
        prompt_template = (
            "You are Large Text Reader AI, an expert assistant for answering questions about documents.\n"
            "Use only the information provided in the context below to answer the user's question.\n"
            "Do not use outside knowledge or make assumptions.\n"
            "If the context does not contain enough information to answer, reply with:\n"
            "\"I cannot find the answer to that question in the provided document.\"\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

    # Creating chain to handle chat interactions
    def _create_chat_chain(self):
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.get_retriever(),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt}
        )

    # Main chat function to handle user questions
    def chat(self, question: str) -> str:
        try:
            self.chat_chain = self._create_chat_chain()
            response = self.chat_chain.invoke({"question": question})
            return response["answer"]
        except Exception as e:
            print(f"Chat error: {e}")
            return "Sorry, I encountered an error. Please try again."
    
    # Function to load default documents from the data directory
    def load_documents(self) -> bool:
        if self.vector_store.get_document_count() > 0:
            print("Vector store already contains documents. Skipping default indexing.")
            return True
        
        print("Loading and indexing default documents from data directory...")
        documents = self.text_manager.load_and_split()
        if not documents:
            print("No default documents found to index.")
            return False
        
        return self.vector_store.add_documents(documents)

    # Function to handle uploaded documents and index them
    def load_uploaded_document(self, file_content: str, filename: str) -> bool:
        try:
            print(f"Processing and indexing uploaded document: {filename}")
            self.vector_store.clear_vector_store()
            chunks = self.text_manager.process_text_content_to_chunks(file_content, filename)
            if not chunks:
                print("No chunks generated from uploaded document. Indexing failed.")
                return False
            success = self.vector_store.add_documents(chunks)
            self.chat_chain = self._create_chat_chain()
            return success
        except Exception as e:
            print(f"Error loading uploaded document: {e}")
            return False
    
    def clear_memory(self):
        self.memory.clear()

    def get_document_count(self) -> int:
        return self.vector_store.get_document_count()


# Global instance to ensure only one RAG system is initialized
_rag_instance = None

# Function to get the singleton instance of SimpleRAG
def get_rag_instance() -> SimpleRAG:
    global _rag_instance
    if _rag_instance is None:
        try:
            print("Initializing RAG system...")
            _rag_instance = SimpleRAG()
            if _rag_instance.vector_store.get_document_count() == 0:
                _rag_instance.load_documents()
            print("RAG system ready!")
        except Exception as e:
            print(f"Failed to initialize: {e}")
            raise
    return _rag_instance

# Function to upload and index a document directly from the frontend
def upload_and_index_document(file_content: str, filename: str) -> bool:
    rag = get_rag_instance()
    return rag.load_uploaded_document(file_content, filename)


# Function to handle chat queries with the RAG system
def chat_with_documents(question: str) -> str:
    try:
        rag = get_rag_instance()
        return rag.chat(question)
    except Exception as e:
        print(f"System error: {e}")
        return "System unavailable. Please try again later."

# Function to reset the RAG system and clear the vector store
def reset_rag_system():
    global _rag_instance
    if _rag_instance:
        _rag_instance.vector_store.clear_vector_store()
        _rag_instance = None
        print("RAG system reset and vector store cleared. Will reinitialize on next use.")
    else:
        print("RAG system not initialized, no reset needed.")

# Function to clear chat history in the RAG system without affecting the vector store
def clear_chat_history():
    rag = get_rag_instance()
    rag.clear_memory()
    print("Chat history cleared.")

