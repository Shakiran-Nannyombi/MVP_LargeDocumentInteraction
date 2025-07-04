import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Generator, List
from langchain_core.documents import Document


class TextFileManager:
    
    # Directory to store text files
    def __init__(self, data_directory: str = None, encoding: str = "utf-8"):
        self.data_directory = data_directory or os.getenv("DATA_DIRECTORY", "./data")
        self.encoding = encoding
        os.makedirs(self.data_directory, exist_ok=True)

    # Lazy loading text files from the specified directory
    def lazy_load(self) -> Generator[Document, None, None]:
        """Lazy load text files from the specified directory."""
        if not os.path.exists(self.data_directory):
            print(f"Warning: Directory {self.data_directory} does not exist")
            return
            
        paths = Path(self.data_directory).glob('**/*.txt')
        for path in paths:
            print(f"Loading {path}")
            try:
                loader = TextLoader(str(path), encoding=self.encoding)
                yield from loader.load()
            except Exception as e:
                print(f"Error loading {path}: {e}")

    # Loading all documents into memory
    def load(self) -> List[Document]:
        """Load all documents into memory."""
        return list(self.lazy_load())
    
    # Splitting documents into chunks
    def load_and_split(self, chunk_size: int = None, chunk_overlap: int = None, separator: str = "\n") -> List[Document]:
        """Load documents and split them into chunks."""
        chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # lazy loading documents and splitTing them
        documents = []
        for doc in self.lazy_load():
            documents.extend(text_splitter.split_documents([doc]))
        
        return documents

    # Handling file uploads and text content processing
    def process_text_content_to_chunks(self, text_content: str, source_name: str = "uploaded_content") -> List[Document]:
        if not text_content or not text_content.strip():
            print("Warning: No text content provided for chunking.")
            return []

        chunk_size = int(os.getenv("CHUNK_SIZE", "1500"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        separator = os.getenv("TEXT_SEPARATOR", "\n\n")
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[separator, "\n\n"]
        )
        
        # Create a single Langchain Document object from the raw text
        doc = Document(page_content=text_content, metadata={"source": source_name})
        
        # Split this single document into chunks
        chunks = text_splitter.split_documents([doc])
        
        print(f"Processed text content into {len(chunks)} chunks.")
        return chunks
    
    # File management functions
    def get_file_count(self) -> int:
        """Get the number of text files in the directory."""
        if not os.path.exists(self.data_directory):
            return 0
        return len(list(Path(self.data_directory).glob('**/*.txt')))

    def list_files(self) -> List[str]:
        """List all text files in the directory."""
        if not os.path.exists(self.data_directory):
            return []
        return [str(path) for path in Path(self.data_directory).glob('**/*.txt')]