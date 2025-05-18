# GenAI RAG Cohort
A Retrieval-Augmented Generation (RAG) system that enhances LLM responses with relevant context from documents.

## Overview
This project implements a RAG pipeline that:
1. Loads PDF documents2. Splits them into manageable chunks
3. Embeds these chunks using OpenAI embeddings4. Stores them in a Qdrant vector database
5. Retrieves relevant context based on user queries6. Generates accurate responses using OpenAI's models

## Example Usage
```python
# Query the systemquery = "What are the key features of Sniffer?"
# System retrieves relevant document chunks from the vector store
# Then generates a response based on the retrieved context# Output: "Sniffer is a network analysis tool that offers packet capture, 
# protocol analysis, and traffic monitoring capabilities..."```

The system combines the power of vector similarity search with LLM capabilities to provide contextually relevant and accurate responses based on your document collection.
