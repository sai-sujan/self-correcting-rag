import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Folder paths
DOCS_DIR = "docs"
MARKDOWN_DIR = "markdown"
PARENT_STORE_PATH = "parent_store"
CHILD_COLLECTION = "document_child_chunks"

# Initialize LLM (Ollama)
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    temperature=0.1
)

# Dense embeddings (for semantic search)
dense_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Sparse embeddings (for keyword search)
sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

# Vector database client
client = QdrantClient(path="qdrant_db")