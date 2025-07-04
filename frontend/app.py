import os
import sys

# Adding the project root directory to Python path
# This allows us to import modules from the backend directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
from dotenv import load_dotenv
from backend.rag_system import get_rag_instance, clear_chat_history, reset_rag_system, upload_and_index_document

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(
    page_title="MVP RAG System",
    page_icon="📚",
    layout="centered"
)

# Initializing session states
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores chat history
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False # To track if RAG system is ready
if "current_document_source" not in st.session_state:
    st.session_state.current_document_source = "Pre-loaded from data/ directory"
# Document checksum is stored to avoid re-uploading the same document
if "last_processed_upload_checksum" not in st.session_state:
    st.session_state.last_processed_upload_checksum = None


# Functions for RAG Interaction: Document loading and chat handling
@st.cache_resource(show_spinner="Initializing RAG system and loading documents...") # ensures this runs only once
def initialize_rag_system():
    rag_system = get_rag_instance()
    st.session_state.rag_initialized = True
    return rag_system

# Function to Processes user question and updates chat history.
def handle_query(question: str):
    rag_system = initialize_rag_system() # Get the already initialized instance

    # Adds user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_system.chat(question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})



st.title("📚 Intelligent Document Query System")

# Sidebar for user inputs
with st.sidebar:
    st.title("Settings & Controls")
    st.markdown("---")

    # File upload for new documents
    st.subheader("Current Document")
    rag_system_instance = initialize_rag_system() 
    
    if st.session_state.rag_initialized:
        st.success("RAG System Ready!")
        doc_count = rag_system_instance.vector_store.get_document_count()
        st.info(f"Currently querying: **{st.session_state.current_document_source}**")
        st.info(f"Chunks indexed: {doc_count}")
    else:
        st.warning("RAG system is initializing for the first time or after reset. Please wait.")

    # Chat history display and handling
    st.subheader("Chat Actions")
    if st.button("Clear Chat History", help="Clear all messages in the current conversation."):
        clear_chat_history() # Clears memory in the RAG system
        st.session_state.messages = [] # Clears display messages
        st.rerun() # Rerun to update the display

    # Resets RAG system to re-initialize and re-index documents
    if st.button("Reset RAG System", help="Clear chat, re-initialize RAG and re-index documents (if needed)."):
        reset_rag_system() # Forces re-initialization of the RAG backend
        st.session_state.messages = []
        st.session_state.rag_initialized = False # Set flag to re-show initialization message
        st.success("RAG system reset. Reloading...")
        st.rerun() # Rerun to trigger re-initialization

    st.markdown("---")
    st.caption("Developed using Langchain, Streamlit, and Groq.")


# Main chat interface


# Displaying chat messages from history on app rerun if any
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at the bottom
if question := st.chat_input("Ask a question about the document..."):
    handle_query(question)