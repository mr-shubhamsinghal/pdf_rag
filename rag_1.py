import os
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore


# # Data source loading
# pdf_path = Path(__file__).parent / "1.1 Sniffer.pdf"
# loader = PyPDFLoader(pdf_path)
# docs = loader.load()

# # Chunking
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=20
# )

# split_docs = text_splitter.split_documents(documents=docs)

# print("DOCS", len(docs))
# print("SPLIT", len(split_docs))

# Embedding & vector store

embeddings = OpenAIEmbeddings(
    model=os.getenv('OPENAI_API_EMBEDDING_MODEL'),
    api_key=os.getenv('OPENAI_API_KEY')
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     collection_name="sniffer",
#     url=os.getenv('VECTOR_DB_URL'),
#     embedding=embeddings
# )

# vector_store.add_documents(documents=split_docs)

# retriever (get relevant chunks from vector store db)

retriever = QdrantVectorStore.from_existing_collection(
    collection_name="sniffer",
    url=os.getenv('VECTOR_DB_URL'),
    embedding=embeddings
)

query = input('Enter you query: ')

relevant_chunks = retriever.similarity_search(
    query=query
)

# print("Relevant Chunks", relevant_chunks)
# pprint(len(relevant_chunks))

context = ' '.join([chunk.page_content for chunk in relevant_chunks])

openai_client = OpenAI()

SYSTEM_PROMPT = """
You are an helpful AI Assistant who responds base of the available context.

Context:
{context}
"""

result = openai_client.chat.completions.create(
    model=os.getenv('OPENAI_API_GPT_MODEL'),
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
)

pprint(result.choices[0].message.content)