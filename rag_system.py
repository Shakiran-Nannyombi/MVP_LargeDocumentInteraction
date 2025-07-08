import chromadb
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from datetime import datetime
import os
import time

class LangchainEmbeddingFunction(EmbeddingFunction):
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return self.langchain_embeddings.embed_documents(input)

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
        if not self.config.get("MISTRALAI_API_KEY"):
            raise ValueError("MISTRALAI_API_KEY not found in config.")

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
        # Set MistralAI API key from config or environment
        if "MISTRALAI_API_KEY" in self.config:
            self.mistral_api_key = self.config["MISTRALAI_API_KEY"]
        else:
            raise ValueError("MISTRALAI_API_KEY not found in config or environment.")

        # Use MistralAI Embeddings
        print("Using MistralAI Embeddings")
        self.embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=self.mistral_api_key)
        self.chroma_embedding_function = LangchainEmbeddingFunction(self.embeddings)

        # Initialize ChromaDB client
        # Initialize ChromaDB client with default host and port
        self.chroma_client = chromadb.HttpClient(
            host='localhost',
            port=8000
        )

        self.collection_name = "uploads"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.chroma_embedding_function,
            metadata={"hnsw:space": "cosine"}  # Ensures vector search uses cosine similarity
        )

        self._ensure_chroma_connection()
        print(f"ChromaDB collection '{self.collection_name}' ready.")

    def _ensure_chroma_connection(self):
        """Ensuring connection to ChromaDB server."""
        try:
            self.chroma_client.heartbeat()
        except Exception as e:
            raise ConnectionError(f"ChromaDB server is not ready or accessible. Please ensure it's running. Error: {e}")
        

    def process_document(self, file_path: str, file_name: str, progress_callback=None):
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
        print(f"Total chunks: {len(all_splits)}")

        # full reset for new file
        try:
            # Get all existing documents to clear the collection
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                print(f"Removed {len(all_data['ids'])} old chunks from collection.")
        except Exception as e:
            print(f"Warning: Could not delete old chunks. Error: {e}")

        # intializing lists to hold new data
        new_ids = []
        new_documents = []
        new_metadatas = []

        # Prepare all texts for batch embedding
        texts = []
        for i, doc in enumerate(all_splits):
            print(f"Preparing chunk {i+1}/{len(all_splits)} for '{file_name}'...")
            doc_id = f"{file_name}_{i}"
            new_ids.append(doc_id)
            new_documents.append(doc.page_content)
            new_metadatas.append({"source_file": file_name, "chunk_index": i, **doc.metadata})
            texts.append(doc.page_content)

        # Batch embed all chunks at once (batch size 64)
        def batch(iterable, n=64):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        all_embeddings = []
        start = time.time()
        total_batches = (len(texts) + 63) // 64
        for batch_num, batch_texts in enumerate(batch(texts, 64)):
            print(f"Embedding batch {batch_num+1} ({len(batch_texts)} chunks)...")
            all_embeddings.extend(self.embeddings.embed_documents(batch_texts))
            # Progress callback is used to update the progress of the document processing
            if progress_callback:
                progress_callback(batch_num + 1, total_batches)
        print(f"Embedding took {time.time() - start:.2f} seconds")
        new_embeddings = all_embeddings

        # Adding new chunks to the collection
        if new_ids:
            self.collection.add(
                documents=new_documents,
                embeddings=new_embeddings,
                ids=new_ids,
                metadatas=new_metadatas
            )
            print(f"Successfully added {len(new_ids)} new chunks for '{file_name}'.")
            print(f"Collection count after add: {self.collection.count()}")
        else:
            print(f"No content to add for document '{file_name}'.")
        # Track current file name for search
        self.current_file_name = file_name

    # Semantic Search Functionality
    def search_document(self, query: str, top_k: int = 3) -> list[str]:
        """
        Searches for relevant document chunks in ChromaDB based on a query.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"source_file": getattr(self, 'current_file_name', None)},
                include=['documents']
            )
            print(f"Search results: {results}")
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"Error during document search: {e}")
            return []

    # Response Generation according to user message and chat history
    def generate_response(self, user_message: str, chat_history: list, SystemMessage: str) -> str:
        relevant_chunks = self.search_document(user_message)
        
        # Takes the system message and enhances it with relevant chunks
        # If no relevant chunks are found, it uses the system instruction content as is
        enhanced_system_prompt = SystemMessage
        
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