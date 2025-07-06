import chromadb
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from datetime import datetime

class RAGSystem:
    def __init__(self, config: dict):
        """
        Initializes the RAG system with LLM, embeddings, text splitter, and ChromaDB client.
        """
        self.config = config
        
        # API keys and model names 
        if not self.config.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in config.")
        if not self.config.get("GROQ_MODEL"):
            raise ValueError("GROQ_MODEL not found in config.")
        azure_api_key = self.config.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = self.config.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment_name = self.config.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        azure_api_version = self.config.get("AZURE_OPENAI_API_VERSION")

        # Initialize the LLM (Groq)
        self.llm = ChatGroq(
            model=self.config["GROQ_MODEL"],
            api_key=self.config["GROQ_API_KEY"],
            temperature=0.3,
        )

        # Initialize Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            add_start_index=True,
        )

        # Initialize Embeddings Model
        azure_api_key = self.config.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = self.config.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment_name = self.config.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        azure_api_version = self.config.get("AZURE_OPENAI_API_VERSION")

        if all([azure_api_key, azure_endpoint, azure_deployment_name, azure_api_version]):
            # Use Azure OpenAI Embeddings
            print("Using Azure OpenAI Embeddings")
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=azure_deployment_name,
                openai_api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
        else:
            raise ValueError("No embedding configuration found. Please provide either Azure OpenAI or HuggingFace embedding configuration.")

        # Initialize ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=self.config.get("CHROMA_HOST", "localhost"), 
            port=int(self.config.get("CHROMA_PORT", 8000)), 
        )
        self.collection_name = "uploads_base" 
        self._ensure_chroma_connection()
        self.collection = self._get_or_create_collection()
        print(f"ChromaDB collection '{self.collection_name}' ready.")

    def _ensure_chroma_connection(self):
        """Ensuring connection to ChromaDB server."""
        try:
            self.chroma_client.heartbeat()
        except Exception as e:
            raise ConnectionError(f"ChromaDB server is not ready or accessible. Please ensure it's running. Error: {e}")

    def _get_or_create_collection(self):
        """Gets or creates the ChromaDB collection."""
        return self.chroma_client.get_or_create_collection(
            self.collection_name,
            embedding_function=None # None because embeddings are generated outside ChromaDB client
        )

    def process_document(self, file_path: str, file_name: str):
        """
        Loads, chunks, embeds, and stores a document in ChromaDB.
        Existing chunks associated with the same file_name will be replaced.
        """
        print(f"Processing document: {file_name} from {file_path}")
        loader = TextLoader(
            file_path=file_path,
            encoding="utf-8",
            autodetect_encoding=True
        )
        documents = loader.lazy_load()
        all_splits = self.text_splitter.split_documents(documents)

        # Deleting all existing chunks for this file to implement "replace" functionality
        try:
            # First, find IDs of documents associated with this file name
            results = self.collection.get(where={"source_file": file_name}, include=['ids'])
            ids_to_delete = results['ids']
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                print(f"Removed {len(ids_to_delete)} old chunks for '{file_name}'.")
        except Exception as e:
            print(f"Warning: Could not delete old chunks for '{file_name}'. Error: {e}")

        # intializing lists to hold new data
        new_ids = []
        new_documents = []
        new_embeddings = []
        new_metadatas = []

        # Processing new chunks for each document
        for i, doc in enumerate(all_splits):
            doc_id = f"{file_name}_{i}"
            new_ids.append(doc_id)
            new_documents.append(doc.page_content)
            new_embeddings.append(self.embeddings.embed_query(doc.page_content))
            new_metadatas.append({"source_file": file_name, "chunk_index": i, **doc.metadata})

        # Adding new chunks to the collection
        if new_ids:
            self.collection.add(
                documents=new_documents,
                embeddings=new_embeddings,
                ids=new_ids,
                metadatas=new_metadatas
            )
            print(f"Successfully added {len(new_ids)} new chunks for '{file_name}'.")
        else:
            print(f"No content to add for document '{file_name}'.")


    # Semantic Search Functionality
    def search_document(self, query: str, top_k: int = 3) -> list[str]:
        """
        Searches for relevant document chunks in ChromaDB based on a query.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=['documents']
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"Error during document search: {e}")
            return []

    # Response Generation according to user message and chat history
    def generate_response(self, user_message: str, chat_history: list, system_instruction_content: str) -> str:
        
        relevant_chunks = self.search_document(user_message)
        
        # Takes the system message and enhances it with relevant chunks
        # If no relevant chunks are found, it uses the system instruction content as is
        enhanced_system_prompt = system_instruction_content
        
        if relevant_chunks:
            context = "\n\n".join(relevant_chunks)
            enhanced_system_prompt += f"\n\nHere is relevant information from the knowledge base:\n{context}\n\nUse this information to answer the user's question. Prioritize information from the knowledge base."

        # Add current time if requested (from original app.py logic)
        needs_time = any(word in user_message.lower() for word in ['time', 'when', 'date', 'today', 'now'])
        if needs_time:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            enhanced_system_prompt = f"Current time: {current_time}\n{enhanced_system_prompt}"

        # Prepare messages for LLM invocation
        # history passed here does *not* include the initial SystemMessage
        # It includes HumanMessage and AIMessage objects
        messages_for_llm = [SystemMessage(content=enhanced_system_prompt)] + chat_history + [HumanMessage(content=user_message)]
        
        # If no relevant chunks, just use the system message and user message
        print(f"Invoking LLM with {len(messages_for_llm)} messages and {len(relevant_chunks)} context chunks.")
        response = self.llm.invoke(messages_for_llm)
        return response.content