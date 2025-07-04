import os
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Generator, List
from langchain_core.documents import Document


class TextFileManager:
    
    def __init__(self, data_directory: str = None, encoding: str = "utf-8"):
        """
        Initialize TextFileManager
        
        Args:
            data_directory: Directory path containing text files to load
            encoding: File encoding (default: utf-8)
        """
        self.data_directory = data_directory or os.getenv("DATA_DIRECTORY", "./data")
        self.encoding = encoding
        os.makedirs(self.data_directory, exist_ok=True)

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

    def load(self) -> List[Document]:
        """Load all documents into memory."""
        return list(self.lazy_load())
    
    def load_and_split(self, chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
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
        
        # Using generator directly to avoid loading everything into memory twice
        documents = []
        for doc in self.lazy_load():
            documents.extend(text_splitter.split_documents([doc]))
        
        return documents

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