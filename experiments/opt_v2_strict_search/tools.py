import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import os
import json
from typing import List
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from core.config import (
    client,
    dense_embeddings,
    sparse_embeddings,
    CHILD_COLLECTION,
    PARENT_STORE_PATH
)

# Initialize vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=CHILD_COLLECTION,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID
)

@tool
def search_child_chunks(query: str, k: int = 5) -> List[dict]:
    """Search for relevant child chunks - returns ONLY parent IDs and snippets.

    OPTIMIZATION: Returns minimal data (parent_ids + 100 char snippets)
    instead of full 500-char chunks. Saves ~2,400 tokens per query.
    """
    try:
        results = vector_store.similarity_search(query, k=k)

        return [
            {
                "parent_id": doc.metadata.get("parent_id", ""),
                "source": doc.metadata.get("source", ""),
                "snippet": doc.page_content[:100] + "..."  # Only first 100 chars!
            }
            for doc in results
        ]
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

@tool
def retrieve_parent_chunks(parent_ids: List[str]) -> List[dict]:
    """Retrieve full parent chunks from disk."""
    results = []
    unique_ids = list(set(parent_ids))

    for parent_id in unique_ids:
        file_path = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "content": data["content"],
                        "parent_id": parent_id,
                        "metadata": data["metadata"]
                    })
            except Exception as e:
                print(f"❌ Error loading {parent_id}: {e}")

    return results
