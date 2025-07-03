import os

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GroqEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Groq


# Loading environment variables from a .env file
load_dotenv()

# Fetching the API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Using langchain to load a text file
loader = TextLoader("Test_files/Blender 4.3 ManuaBlenderDocumentationTeam.txt", encoding="utf-8")
documents = loader.load()

# Splitting the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # characters per chunk
    chunk_overlap=200 # characters overlap between chunks
)
chunks = text_splitter.split_documents(documents)

# Embedding and Vector Database Setup
embedding = GroqEmbeddings(api_key=GROQ_API_KEY)
vector_store = Chroma(embedding_function=embedding)

# Adding the chunks to the vector store
vector_store.add_documents(chunks)

#using groq specified llm model
llm = Groq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
