# LARGE TEXT FILE RETRIEVAL RAG MVP

## Project Overview

This project develops an **Intelligent Document Query System** using the Retrieval Augmented Generation (RAG) pattern. The goal is to allow users to "chat" with a large text document, asking questions and receiving answers grounded in the document's content.

This initial phase (MVP) establishes the core backend RAG pipeline, capable of processing, indexing, and intelligently querying a substantial text corpus.

## What is RAG? (Retrieval Augmented Generation)

RAG enhances Large Language Models (LLMs) by giving them external, relevant information to generate more accurate and factual responses, reducing "hallucinations."

### How RAG Works:

**1. Making the Document Searchable:**

We prepare the document by:
*   **Loading:** Reading the raw `.txt` file.
*   **Splitting:** Breaking it into smaller, manageable chunks.
*   **Embedding:** Converting each chunk into numerical "embeddings" (vectors that capture meaning).
*   **Storing:** Saving these chunks and embeddings in a vector database (ChromaDB) for quick retrieval.

![RAG Pipeline Steps](screenshots/RAG_steps.png)

**2. Getting Answers from the Document:**

When a question is asked:
*   The question is converted into an embedding.
*   **Retrieve:** We find the most relevant document chunks (based on similarity to the question) from our vector database.
*   **Prompt:** These relevant chunks, along with the question and chat history, are sent to the LLM.
*   **LLM Generates:** The LLM uses *only* the provided context to create an answer.
*   **Answer:** The response is given to the user.

![RAG Query Process](screenshots/RAG_process.png)

