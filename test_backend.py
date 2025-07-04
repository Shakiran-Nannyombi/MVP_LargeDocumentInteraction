import os
from dotenv import load_dotenv
from backend.rag_system import get_rag_instance, clear_chat_history, reset_rag_system

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    print("--- RAG System Test ---")
    
    # Check if environment variables are loaded
    print(f"GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")
    print(f"DATA_DIRECTORY: {os.getenv('DATA_DIRECTORY', './data')}")
    print(f"CHROMA_PERSIST_DIRECTORY: {os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')}")

    try:
        # Get the RAG instance (will initialize and load documents on first call)
        print("\nInitializing RAG system...")
        rag_instance = get_rag_instance()
        
        # Check document count
        doc_count = rag_instance.get_document_count()
        print(f"📄 Document chunks loaded: {doc_count}")
        
        if doc_count == 0:
            print("⚠️  No documents found! Make sure you have .txt files in your data directory.")
            print(f"📁 Looking for files in: {os.getenv('DATA_DIRECTORY', './data')}")
            exit(1)

        # Test a query
        print("\n" + "="*50)
        print("Query 1:")
        question1 = "What is the main theme of the document?"
        answer1 = rag_instance.chat(question1)
        print(f"Q: {question1}")
        print(f"A: {answer1}")

        # Test another query to see if memory works
        print("\n" + "="*50)
        print("Query 2 (with memory):")
        question2 = "Can you elaborate on that?"
        answer2 = rag_instance.chat(question2)
        print(f"Q: {question2}")
        print(f"A: {answer2}")

        # Test the clear history function
        print("\n" + "="*50)
        print("Clearing chat history...")
        clear_chat_history()
        
        print("\nQuery 3 (after clearing history):")
        question3 = "What is the main theme of the document?"
        answer3 = rag_instance.chat(question3)
        print(f"Q: {question3}")
        print(f"A: {answer3}")

        # Test resetting the system
        print("\n" + "="*50)
        print("Resetting RAG system...")
        reset_rag_system()

        print("\nQuery 4 (after reset, should re-initialize):")
        question4 = "What is the document about?"
        answer4 = get_rag_instance().chat(question4)
        print(f"Q: {question4}")
        print(f"A: {answer4}")

        print("\n✅ --- Test Complete ---")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("\nDebugging tips:")
        print("1. Check if GROQ_API_KEY is set in .env")
        print("2. Make sure you have .txt files in ./data directory")
        print("3. Check if all dependencies are installed")