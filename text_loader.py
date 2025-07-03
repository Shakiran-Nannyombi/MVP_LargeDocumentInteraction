import os
from typing import List, Optional
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentLoader:
    """
    A class to handle document loading and text splitting
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the DocumentLoader
        
        Args:
            chunk_size (int): Number of characters per chunk
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_text_file(self, file_path: str, encoding: str = "utf-8") -> List[Document]:
        """
        Load a text file and return the documents
        
        Args:
            file_path (str): Path to the text file
            encoding (str): File encoding (default: utf-8)
            
        Returns:
            List[Document]: List of loaded documents
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            loader = TextLoader(file_path, encoding=encoding)
            documents = loader.load()
            return documents
        
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return []
    
    def load_from_bytes(self, file_content_bytes: bytes, filename: str) -> List[Document]:
        """
        Load document from bytes content
        
        Args:
            file_content_bytes (bytes): File content as bytes
            filename (str): Original filename for metadata
            
        Returns:
            List[Document]: List of documents
        """
        try:
            string_data = file_content_bytes.decode("utf-8")
            document = Document(
                page_content=string_data,
                metadata={"source": filename}
            )
            return [document]
        except Exception as e:
            print(f"Error loading from bytes: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Document]: List of document chunks
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            return []
    
    def load_and_split_file(self, file_path: str, encoding: str = "utf-8") -> List[Document]:
        """
        Load a file and split it into chunks in one step
        
        Args:
            file_path (str): Path to the text file
            encoding (str): File encoding (default: utf-8)
            
        Returns:
            List[Document]: List of document chunks
        """
        documents = self.load_text_file(file_path, encoding)
        if documents:
            chunks = self.split_documents(documents)
            print(f"Loaded {len(documents)} documents and created {len(chunks)} chunks")
            return chunks
        return []
    
    def load_and_split_bytes(self, file_content_bytes: bytes, filename: str) -> List[Document]:
        """
        Load from bytes and split into chunks in one step
        
        Args:
            file_content_bytes (bytes): File content as bytes
            filename (str): Original filename for metadata
            
        Returns:
            List[Document]: List of document chunks
        """
        documents = self.load_from_bytes(file_content_bytes, filename)
        if documents:
            chunks = self.split_documents(documents)
            print(f"Loaded document '{filename}' and created {len(chunks)} chunks")
            return chunks
        return []
    
    def load_multiple_files(self, file_paths: List[str], encoding: str = "utf-8") -> List[Document]:
        """
        Load multiple text files and combine them
        
        Args:
            file_paths (List[str]): List of file paths
            encoding (str): File encoding (default: utf-8)
            
        Returns:
            List[Document]: List of document chunks from all files
        """
        all_chunks = []
        
        for file_path in file_paths:
            chunks = self.load_and_split_file(file_path, encoding)
            all_chunks.extend(chunks)
        
        print(f"Total chunks from {len(file_paths)} files: {len(all_chunks)}")
        return all_chunks

# Convenience functions for quick loading
def quick_load_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, encoding: str = "utf-8") -> List[Document]:
    """
    Quick function to load and split a single file
    
    Args:
        file_path (str): Path to the text file
        chunk_size (int): Number of characters per chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        encoding (str): File encoding
        
    Returns:
        List[Document]: List of document chunks
    """
    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_and_split_file(file_path, encoding)

def quick_load_bytes(file_content_bytes: bytes, filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Quick function to load and split from bytes
    
    Args:
        file_content_bytes (bytes): File content as bytes
        filename (str): Original filename for metadata
        chunk_size (int): Number of characters per chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[Document]: List of document chunks
    """
    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_and_split_bytes(file_content_bytes, filename)

