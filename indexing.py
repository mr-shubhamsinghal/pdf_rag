import os
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore


# # Data source loading
pdf_path = Path(__file__).parent / "1.1 Sniffer.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# # Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

split_docs = text_splitter.split_documents(documents=docs)

# print("DOCS", len(docs))
# print("SPLIT", len(split_docs))

# Embedding & vector store

embeddings = OpenAIEmbeddings(
    model=os.getenv('OPENAI_API_EMBEDDING_MODEL'),
    api_key=os.getenv('OPENAI_API_KEY')
)

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    collection_name="sniffer",
    url=os.getenv('VECTOR_DB_URL'),
    embedding=embeddings
)

vector_store.add_documents(documents=split_docs)
