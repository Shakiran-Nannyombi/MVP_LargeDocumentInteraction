import os
import streamlit as st
from dotenv import dotenv_values
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rag_system import RAGSystem

# Loading environment variables from .env file
config = dotenv_values(".env")

# LangSmith tracing configuration
os.environ["LANGCHAIN_TRACING_V2"] = config.get("LANGSMITH_TRACING", "false") # Changed default to "false" - only enable if you have API key and want to use it
os.environ["LANGCHAIN_API_KEY"] = config.get("LANGSMITH_API_KEY", "") 
os.environ["LANGCHAIN_PROJECT"] = config.get("LANGSMITH_PROJECT", "RAG-Capstone-MVP") # Added project name, good practice

# Page configuration
st.set_page_config(
    page_title="MVP", # Changed title for clarity
    page_icon="📚",
    layout="wide"
)

# Check essential API keys and model names
# Adjusted check to include Azure OpenAI keys
required_groq_keys = ["GROQ_API_KEY", "GROQ_MODEL"]
required_azure_keys = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION"]

missing_groq_keys = [key for key in required_groq_keys if not config.get(key)]
if missing_groq_keys:
    st.error(f"Please set the following Groq keys in your .env file: {', '.join(missing_groq_keys)}")
    st.stop()

# Check for embedding configuration 
has_azure_config = all(config.get(key) for key in required_azure_keys)

if not has_azure_config:
    st.error("No valid embedding configuration found in .env. Please provide Azure OpenAI details (AZURE_OPENAI_API_KEY, etc.).")
    st.stop()

# Initialize RAGSystem (singleton pattern for Streamlit)
@st.cache_resource
def get_rag_system():
    """Caches the RAGSystem instance to avoid re-initialization on every rerun."""
    try:
        return RAGSystem(config)
    except ConnectionError as e:
        st.error(f"Failed to connect to ChromaDB: {e}") # specific error message
        st.stop()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.stop()

rag_system = get_rag_system()

# Streamlit Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores conversation history (HumanMessage, AIMessage)
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None # To track which file is currently processed

# System message to provide context to the LLM 
system_instruction_content = """
You are a helpful assistant that can answer questions about large documents.
You can summarize, extract information, and provide insights based on the content of the document.
Your responses should be based on the provided document context and the conversation history.
Your responses should be concise, accurate, and directly relevant to the user's queries, grounded in the provided document content.
If you are unsure about something or the information is not in the document, state that you don't have enough information.
"""

# Sidebar
with st.sidebar:
    st.title("📚 Intelligent Document Query System")
    
    st.divider()
    
    # Assistant status
    st.subheader("🤖 Assistant Status")
    st.write("The assistant is currently online.")
    st.write(f"Connected to Groq Model: `{config['GROQ_MODEL']}`")
    
    # Check if Azure OpenAI embedding configuration is available
    if has_azure_config:
        st.write("Using Embedding Model: `Azure OpenAI Embedding`")
    else:
        st.write("No embedding model configured.") 

    # Display current document status
    if st.session_state.current_file_name:
        st.info(f"Currently chatting with: **{st.session_state.current_file_name}**")
    else:
        st.warning("No document loaded yet. Please upload one.")

    # File upload
    st.subheader("📁 Document Upload")
    
    # File uploader for document uploads
    uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])
    
    if uploaded_file is not None:
        # Create data directory if it doesn't exist
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Define the path where the file will be saved
        file_save_path = os.path.join(data_dir, uploaded_file.name)
        
        # Only process if it's a new file OR if the file on disk is different
        needs_processing = True
        if st.session_state.current_file_name == uploaded_file.name and os.path.exists(file_save_path):
            # If the file is already loaded and exists, don't process it again
             needs_processing = False
        
        if needs_processing: # Only proceed if processing is needed
            with st.spinner(f"Saving and processing '{uploaded_file.name}'..."):
                # Save uploaded file (overwrite if exists)
                with open(file_save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the document using our RAGSystem
                try:
                    rag_system.process_document(file_save_path, uploaded_file.name)
                    st.session_state.current_file_name = uploaded_file.name
                    st.success(f"Document '{uploaded_file.name}' processed successfully and loaded!")
                    # Clear chat history when a new document is loaded
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    st.session_state.current_file_name = None # Indicate failure

            st.rerun() # Rerun to update UI with new document status and cleared chat
        else:
            st.info(f"Document '{uploaded_file.name}' already loaded.") # Inform user it's already loaded


    # Clear Current Chat History Button
    if st.button("Clear Current Chat History", key="clear_chat_button"):
        st.session_state.messages = [] # Clear displayed messages
        st.success("Chat history cleared!")
    
    # footer
    st.divider()
    st.markdown("Made with ❤️ by Kiran")

# --- Main Content Area ---

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input widget for user to type messages
if user_query := st.chat_input("Enter your question about the document..."):
    if not st.session_state.current_file_name:
        st.warning("Please upload a document first to start asking questions.")
        st.stop() # Prevent further processing without a document

    # Add user message to conversation history
    st.session_state.messages.append(HumanMessage(content=user_query))
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the RAGSystem to generate the response
            try:
                response_content = rag_system.generate_response(
                    user_query,
                    st.session_state.messages[:-1], # Pass all previous messages (excluding the current user_query)
                    system_instruction_content # Pass the base system instruction
                )
                ai_message = AIMessage(content=response_content)
                st.write(ai_message.content)
                st.session_state.messages.append(ai_message) # Add AI response to history
            except Exception as e:
                st.error(f"An error occurred while generating response: {e}")
                st.session_state.messages.append(AIMessage(content="Sorry, I encountered an error and couldn't generate a response."))
    
    st.rerun() # Rerun the app to display the new messages
    