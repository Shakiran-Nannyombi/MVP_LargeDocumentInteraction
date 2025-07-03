import datetime
import os
import sys
from dotenv import load_dotenv
import streamlit as st

# Add parent directory to path to import main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import get_rag_backend

# Loading environment variables from a .env file
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Large Document Interaction AI",
    page_icon="📚",
    layout="wide"
)

# Title of the app
st.title("📚 Large Document Interaction AI App")

# Initialize RAG backend
@st.cache_resource
def init_backend():
    """Initialize and cache the RAG backend"""
    try:
        return get_rag_backend()
    except Exception as e:
        st.error(f"Failed to initialize backend: {str(e)}")
        st.error("Please make sure your GROQ_API_KEY is set in the .env file")
        st.stop()

backend = init_backend()

# Initialize session state
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for document management
with st.sidebar:
    # Try to load logo from different possible paths
    logo_paths = [
        "images/logo.png",
        "frontend/images/logo.png", 
        os.path.join(os.path.dirname(__file__), "images", "logo.png")
    ]
    
    logo_loaded = False
    for logo_path in logo_paths:
        try:
            if os.path.exists(logo_path):
                st.image(logo_path, width=32)
                logo_loaded = True
                break
        except:
            continue
    
    if not logo_loaded:
        st.markdown("# 📚")
    
    st.markdown("## Document & Chat Management")
    
    # Clear Chat History Button
    if st.button("🗑️ Clear Current Chat", key="clear_chat"):
        if st.session_state.active_doc_id:
            backend.clear_chat_history(st.session_state.active_doc_id)
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
        else:
            st.warning("No document selected.")
    
    st.markdown("---")
    st.markdown("### 📤 Upload New Document")
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

    if uploaded_file is not None:
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            try:
                file_bytes = uploaded_file.read()
                new_doc_id = backend.process_and_store_document(file_bytes, uploaded_file.name)
                
                # Set newly uploaded document as active
                st.session_state.active_doc_id = new_doc_id
                
                # Load chat history for the new doc
                st.session_state.messages = backend.get_chat_history(new_doc_id)

                st.success(f"✅ '{uploaded_file.name}' loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    st.markdown("---")
    st.markdown("### 📚 Select Existing Document")
    
    # Load available documents
    available_docs = backend.get_available_documents()
    
    if available_docs:
        # Create options for selectbox
        doc_options = {"--- Select a document ---": None}
        doc_options.update(available_docs)
        
        # Determine current selection
        current_doc_name = None
        if st.session_state.active_doc_id:
            current_doc_name = available_docs.get(st.session_state.active_doc_id, "--- Select a document ---")
        
        selected_doc_name = st.selectbox(
            "Choose from your library:",
            options=list(doc_options.keys()),
            index=list(doc_options.keys()).index(current_doc_name) if current_doc_name else 0,
            key="doc_selector"
        )

        if selected_doc_name != "--- Select a document ---":
            selected_doc_id = None
            # Find the doc_id for the selected name
            for doc_id, name in available_docs.items():
                if name == selected_doc_name:
                    selected_doc_id = doc_id
                    break
            
            if selected_doc_id and selected_doc_id != st.session_state.active_doc_id:
                # Switch to selected document
                st.session_state.active_doc_id = selected_doc_id
                st.session_state.messages = backend.get_chat_history(selected_doc_id)
                st.success(f"📖 Switched to: **{selected_doc_name}**")
                st.rerun()
    else:
        st.info("No documents uploaded yet.")

    st.markdown("---")
    if st.session_state.active_doc_id:
        current_doc_name = available_docs.get(st.session_state.active_doc_id, 'Unknown')
        st.markdown(f"**📖 Current Document:** {current_doc_name}")
        st.markdown(f"**💬 Messages:** {len(st.session_state.messages)}")
    else:
        st.markdown("**📖 Current Document:** None selected")

# Main chat interface
if st.session_state.active_doc_id:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("💬 Ask a question about the document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        backend.save_message(st.session_state.active_doc_id, "user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                start_time = datetime.datetime.now()
                
                # Query the document
                response = backend.query_document(st.session_state.active_doc_id, prompt)
                
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time

                st.markdown(response)
                st.caption(f"⏱️ Response generated in {elapsed_time.total_seconds():.2f} seconds")

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        backend.save_message(st.session_state.active_doc_id, "assistant", response)
        
        st.rerun()

else:
    # Welcome screen when no document is selected
    st.markdown("""
    ## 👋 Welcome to the Document AI Assistant!
    
    To get started:
    1. **📤 Upload a new document** using the sidebar
    2. **📚 Select an existing document** from your library
    
    Once you've selected a document, you can:
    - 💬 Ask questions about the content
    - 🔍 Search for specific information
    - 📝 Get summaries and insights
    
    Your chat history is automatically saved for each document!
    """)
    
    # Show available documents if any
    available_docs = backend.get_available_documents()
    if available_docs:
        st.markdown("### 📚 Your Document Library:")
        for doc_id, filename in available_docs.items():
            chat_count = len(backend.get_chat_history(doc_id))
            st.markdown(f"- **{filename}** ({chat_count} messages)")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and Groq")
