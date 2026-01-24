import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()

# Folder paths
DOCS_DIR = "docs"
MARKDOWN_DIR = "markdown"
PARENT_STORE_PATH = "parent_store"
CHILD_COLLECTION = "document_child_chunks"

# Initialize LLM (Ollama)
# Initialize LLM (Ollama)
# Config D (Winner): temperature=0.3 for balanced performance
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    temperature=0.3
)

# Small LLM for extraction/compression tasks
# Config D (Winner): temperature=0.3 for better recall/creativity
llm_small = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.3
)

# Dense embeddings (for semantic search)
dense_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Sparse embeddings (for keyword search)
sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

# Cross-encoder reranker (lightweight, ~22M params)
# Options: "cross-encoder/ms-marco-MiniLM-L-6-v2" (fastest)
#          "BAAI/bge-reranker-base" (better quality)
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)

# Vector database client
client = QdrantClient(path="qdrant_db")