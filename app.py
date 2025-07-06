import os
import streamlit as st
import chromadb

from dotenv import dotenv_values
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


# Load environment variables from .env file
config = dotenv_values(".env")

# LangSmith tracing
config.get("LANGSMITH_TRACING") == "true"
config.get("LANGSMITH_API_KEY")

# Page configuration
st.set_page_config(
    page_title="MVP",
    page_icon="📚",
    layout="wide"
)

# Check if API key was set
if not config.get("GROQ_API_KEY"):
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()  # Halt execution if API key is missing

if not config.get("GROQ_MODEL"):
    st.error("Please set your GROQ_MODEL in the .env file")
    st.stop()  # Halt execution if model name is missing

# Extracting the API key and model name from the config
groq_api_key = config["GROQ_API_KEY"]
groq_model = config["GROQ_MODEL"]

# Initialize the LLM with Groq
llm = ChatGroq(
    model=groq_model,
    api_key=groq_api_key,
    temperature=0.3,  
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
    )

# Embeddings
embedding_model_name = config.get("EMBEDDING_MODEL_NAME")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(
    host="http://localhost:8000",  
    port=8080,  
)
chroma_client.heartbeat()
if not chroma_client.is_ready():
    st.error("ChromaDB client is not ready. Please ensure the ChromaDB server is running.")
    st.stop()  #if ChromaDB is not ready
else:
    st.success("ChromaDB client initialized successfully.")

# Collection for storing documents
collection_name = "uploads"
if not chroma_client.has_collection(collection_name):
    chroma_client.create_collection(collection_name)

#Intializing session states
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store the last messages sent to the LLM
if "file_path" not in st.session_state:
    st.session_state.file_path = "data/"  # Store uploaded file path (for future use)
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None

    
# System message to provide context to the LLM
system_message = SystemMessage(
    content="""You are a helpful assistant that can answer questions about large documents.
You will be provided with a conversation history and the current time.
Your responses should be concise and relevant to the user's queries.
If the user asks about a specific document, you should reference that document.
You can also provide summaries or explanations of the documents if requested.
Remember to always consider the current time when providing time-sensitive information.
For example, if the user asks about the time, you should respond with the current time.
You can also provide information about the documents that have been uploaded by the user.
If the user asks about a document, you should reference that document.
If the user asks about a specific topic, you should provide relevant information from the documents.
If you are unsure about something, you can ask the user for clarification.
""",
)


# Sidebar
with st.sidebar:
    st.title("📚 Large Document Interaction MVP")
    
    st.divider()
    
    # Assistant status
    st.subheader("🤖 Assistant Status")
    st.write("The assistant is currently online.")
    
    # File upload
    st.subheader("📁 Document Upload")
    
    # File uploader for document uploads
    if file := st.file_uploader("Upload a file", type=["txt", "pdf", "docx"]):
        # Create file path in data directory
        st.session_state.file_path = f"data/{file.name}"
        # Check if file doesn't already exist to avoid overwriting
        if not os.path.exists(st.session_state.file_path):
            file.seek(0)  # Reset file pointer to beginning
            contents = file.read()  # Read file contents

            # Save uploaded file to local data directory
            with open(st.session_state.file_path, "wb") as f:
                f.write(contents)

        st.rerun()  # Refresh app to show updated file path
   
   # Clear Current Chat History Button
    if st.button("Clear Current Chat History", key="clear_chat_button"):
        if st.session_state.active_doc_id:
            # rag_pipeline.clear_chat_history_for_document(st.session_state.active_doc_id)
            st.session_state.messages = [] # Clear displayed messages
            st.success("Chat history for current document cleared!")
        else:
            st.warning("No document selected to clear chat history.")
    
    # footer
    st.divider()
    st.markdown("Made with ❤️ by Kiran")


# Main content area

# Function to retrieve uploaded document for RAG 
def get_uploaded_document(message: HumanMessage) -> list[str]:
    
    #text loader
    loader = TextLoader(
        file_path=st.session_state.file_path,  
        encoding="utf-8",
        autodetect_encoding=True  
    )
    documents = loader.lazy_load()  # Load the document from the file path

    # Load the document and split it into chunks
    all_splits = text_splitter.split_documents(documents)
    
    # Store the document in the ChromaDB collection
    for documents in all_splits:
        # Convert document to embeddings
        embedding = embeddings.embed_query(documents.page_content)

        # Add document to ChromaDB collection
        chroma_client.get_collection(collection_name).add(
            documents=[documents.page_content],
            embeddings=[embedding],
            ids=[documents.metadata.get("id", documents.page_content[:10])],  # Use first 10 chars as ID if not set
        )
    return [doc.page_content for doc in all_splits]

# Function to generate AI responses using the LLM
def generate_response():
    start_time = datetime.now()  # Capture current timestamp

    # Check if a document is uploaded
    if not st.session_state.file_path or not os.path.exists(st.session_state.file_path):
        st.error("Please upload a document to interact with.")
        return AIMessage(content="No document uploaded. Please upload a document to start interacting.")

    # Create an enhanced system message with current timestamp
    combined_sys_message = f"""Current time: {start_time}
{system_message.content}"""
    
    # Prepare the full conversation context for the LLM
    messages = [SystemMessage(content=combined_sys_message)] + st.session_state.messages
    
    # Send messages to the LLM and get response
    response = llm.invoke(messages)
    
    # Add the AI response to conversation history
    st.session_state.messages.append(response)
    
    # Store the messages that were sent to the LLM (for debugging/monitoring)
    st.session_state.latest_msgs_sent = messages
    
    return response


# Display chat history
# Loop through all stored messages and display them with appropriate chat bubbles
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        # Display user messages with "user" avatar
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        # Display AI messages with "assistant" avatar
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input widget for user to type messages
# Uses walrus operator (:=) to capture input and check if it's not empty
if msg := st.chat_input("Enter a message"):
    # Add user message to conversation history
    st.session_state.messages.append(HumanMessage(content=msg))
    
    # Generate AI response and add it to history
    response = generate_response()
    
    # Rerun the app to display the new messages
    st.rerun()