import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from .vector_store import VectorStore

class SimpleRAG:

    # Initialize the RAG system with necessary components
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key='answer'
        )
        
        # Create a custom prompt for question answering
        self.qa_prompt = self._create_custom_prompt()
        
        # Create the conversational retrieval chain
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.get_retriever(),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt}
        )
    
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

    # User-friendly function to chat with the system
    def chat(self, question: str) -> str:
        try:
            response = self.chat_chain({"question": question})
            return response["answer"]
        except Exception as e:
            print(f"Chat error: {e}")
            return "Sorry, I encountered an error. Please try again."
    
    # Load and index documents into the vector store
    def load_documents(self) -> bool:
        """Simple load - let underlying errors bubble up"""
        return self.vector_store.load_and_index_documents()
    
    # Clear the conversation memory
    def clear_memory(self):
        """Simple method - no error handling needed"""
        self.memory.clear()

    # Getting count of documents in the vector store
    def get_document_count(self) -> int:
        """Simple getter - let vector store handle errors"""
        return self.vector_store.get_document_count()


# Global instance
_rag_instance = None

# Function to get or create the RAG instance
def get_rag_instance() -> SimpleRAG:
    global _rag_instance
    if _rag_instance is None:
        try:
            _rag_instance = SimpleRAG()
            _rag_instance.load_documents()
            print("RAG system ready!")
        except Exception as e:
            print(f"Failed to initialize: {e}")
            raise  # Let caller handle it
    return _rag_instance

# Main user interface function to chat with documents
def chat_with_documents(question: str) -> str:
    try:
        rag = get_rag_instance()
        return rag.chat(question)
    except Exception as e:
        print(f"System error: {e}")
        return "System unavailable. Please try again later."

# Function to reset the RAG system and clear memory
def reset_rag_system():
    global _rag_instance
    _rag_instance = None

# Function to clear chat history
def clear_chat_history():
    rag = get_rag_instance()
    rag.clear_memory()