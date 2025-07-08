import os
import streamlit as st
from dotenv import dotenv_values
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rag_system import RAGSystem
import json
from datetime import datetime

# Loading environment variables from .env file and Render
if os.path.exists(".env"):
    # Load from .env for local development
    config = dotenv_values(".env")
    # Set defaults for CHROMA_HOST/PORT if not in .env, for local testing without Docker Compose
    # Although, better to run with Docker Compose even locally for consistency
    config.setdefault("CHROMA_HOST", "localhost")
    config.setdefault("CHROMA_PORT", "8000")
else:
    # Environment variables for Render deployment or other prod env
    config = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GROQ_MODEL": os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        "MISTRALAI_API_KEY": os.getenv("MISTRALAI_API_KEY"),
        "USE_MEMORY_CHROMA": os.getenv("USE_MEMORY_CHROMA", "true").lower() == "true",
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING", "false"),
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY", ""),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT", "RAG-Capstone-MVP"),
        # Add ChromaDB host/port here from environment variables too, 
        # so RAGSystem can pick them up if running without a .env file directly
        "CHROMA_HOST": os.getenv("CHROMA_HOST", "localhost"), # Default to localhost if not set
        "CHROMA_PORT": os.getenv("CHROMA_PORT", "8000") # Default to 8000 if not set
    }

# Page configuration
st.set_page_config(
    page_title="MVP", 
    page_icon="📚",
    layout="wide"
)

# Check essential API keys and model names
# Adjusted check to include Azure OpenAI keys
required_groq_keys = ["GROQ_API_KEY", "GROQ_MODEL"]
required_mistral_keys = ["MISTRALAI_API_KEY"]

missing_groq_keys = [key for key in required_groq_keys if not config.get(key)]
if missing_groq_keys:
    st.error(f"Please set the following Groq keys in your .env file: {', '.join(missing_groq_keys)}")
    st.stop()

# Check for embedding configuration 
has_mistral_config = all(config.get(key) for key in required_mistral_keys)

if not has_mistral_config:
    st.error("No valid embedding configuration found in .env. Please provide MistralAI embedding details (MISTRALAI_API_KEY).")
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

# Your SystemMessage variable
SystemMessage_content = """ 
You are a helpful assistant that can answer questions about large documents.
Be able to reply to greetings, provide summaries, and answer specific questions about the content of the document.
You should always respond in a friendly and informative manner.
You should be able to respond to gestures like "hello", "hi", "hey", and similar greetings.
And aslo appreciation like thank you, thanks, etc.
If user says ok and bye, you should end the conversation politely.
You should be able to handle questions about the document's content, such as "What is the main topic of the document?", "Can you summarize this document?", or "What are the key points
You can summarize, extract information, and provide insights based on the content of the document.
Your responses should be based on the provided document context and the conversation history.
Your responses should be concise, accurate, and directly relevant to the user's queries, grounded in the provided document content.
If you are unsure about something or the information is not in the document, state that you don't have enough information.
""" 


# --- Sidebar ---
with st.sidebar:
    st.header("📚 Intelligent Document Query System")
    st.divider()
    st.subheader("🤖 Assistant Status")

    # Display current document status
    if st.session_state.current_file_name:
        st.info(f"Currently chatting with: **{st.session_state.current_file_name}**")
    else:
        st.warning("No document loaded yet. Please upload one.")

    # File upload
    st.subheader("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])
    
    if uploaded_file is not None:
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        file_save_path = os.path.join(data_dir, uploaded_file.name)
        needs_processing = True
        if st.session_state.current_file_name == uploaded_file.name and os.path.exists(file_save_path):
            needs_processing = False

        if needs_processing:
            # Save uploaded file
            with open(file_save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Progress bar in main chat area
            progress_bar = st.progress(0, text="Processing document...")

            def progress_callback(progress, total):
                percent = int(progress / total * 100)
                progress_bar.progress(percent, text=f"Processing: {percent}%")

            try:
                rag_system.process_document(file_save_path, uploaded_file.name, progress_callback=progress_callback)
                st.session_state.current_file_name = uploaded_file.name
                st.success("Document ready! Chat with assistant.")
                st.session_state.messages = []
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.session_state.current_file_name = None

            st.rerun()
        else:
            st.info(f"Document '{uploaded_file.name}' already loaded.")

    # --- Chat Save/Load/Delete ---
    os.makedirs("chats", exist_ok=True)
    chat_files = [f for f in os.listdir("chats") if f.endswith(".json")]
    selected_chat = st.selectbox("Load previous chat", ["-"] + chat_files, index=0)
    if selected_chat != "-":
        if st.button("Load Chat", key="load_chat_btn"):
            with open(f"chats/{selected_chat}", "r", encoding="utf-8") as f:
                chat_data = json.load(f)
            doc_name = chat_data.get("document_name")
            # If the document is not currently loaded, process it
            if st.session_state.current_file_name != doc_name and os.path.exists(f"data/{doc_name}"):
                # Progress bar for loading existing document
                progress_bar = st.progress(0, text="Loading document...")

                def progress_callback(progress, total):
                    percent = int(progress / total * 100)
                    progress_bar.progress(percent, text=f"Loading: {percent}%")

                try:
                    rag_system.process_document(f"data/{doc_name}", doc_name, progress_callback=progress_callback)
                    st.session_state.current_file_name = doc_name
                except Exception as e:
                    st.error(f"Error loading document: {e}")
                    st.session_state.current_file_name = None

            # Load messages from chat file
            st.session_state.messages = [HumanMessage(content=m["content"]) if m["role"]=="HumanMessage" else AIMessage(content=m["content"]) for m in chat_data["messages"]]
            st.success(f"Loaded chat for document: {doc_name}")
            st.rerun()
        if st.button("Delete Chat", key="delete_chat_btn"):
            os.remove(f"chats/{selected_chat}")
            st.success(f"Deleted chat: {selected_chat}")
            st.rerun()

    # Save chat button
    if st.session_state.get("messages") and st.session_state.get("current_file_name"):
        if st.button("Save Current Chat", key="save_chat_btn"):
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            filename = f"chats/{st.session_state.current_file_name}_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump({
                    "document_name": st.session_state.current_file_name,
                    "messages": [{"role": type(m).__name__, "content": m.content} for m in st.session_state.messages]
                }, f, ensure_ascii=False, indent=2)
            st.success(f"Chat saved as {filename}")

    # Clear Current Chat History Button
    if st.button("Clear Current Chat History", key="clear_chat_button"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        
    # Footer
    st.markdown("Made with ❤️ by Kiran")

# --- Main Content Area ---

# Show welcome/instructions if no document is loaded
if st.session_state.current_file_name is None:
    st.markdown("""
    # 👋 Welcome to the Intelligent Large Document Query System MVP!
    
    **Instructions:**
    1. Upload a `.txt` document using the uploader below.
    2. Wait for the progress bar to finish processing your document.
    3. Once you see "Document ready! Chat with assistant.", you can start chatting with the assistant about your document.
    4. Ask questions, request summaries, or extract information from your uploaded document.
    
    _Your document is processed securely and never stored permanently. 
    The assistant uses advanced AI to understand and respond to your queries based on the content of the document you provide._
    
    Happy querying! 😊
    """)

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
        st.stop()

    st.session_state.messages.append(HumanMessage(content=user_query))
    
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_content = rag_system.generate_response(
                    user_query,
                    st.session_state.messages[:-1],
                    SystemMessage_content
                )
                ai_message = AIMessage(content=response_content)
                st.write(ai_message.content)
                st.session_state.messages.append(ai_message)
            except Exception as e:
                # Print full error details to terminal
                print(f"\n--- ERROR IN LLM INVOCATION ---")
                import traceback
                traceback.print_exc()
                print(f"Error details: {e}")
                print(f"--- END ERROR ---\n")
                st.error(f"An error occurred while generating response. Please check the terminal for details. Error: {e}")
                st.session_state.messages.append(AIMessage(content="Sorry, I encountered an error and couldn't generate a response."))
    
    st.rerun()
