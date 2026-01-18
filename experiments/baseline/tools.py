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

# Initialize vector store (connects to our database)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=CHILD_COLLECTION,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID
)

@tool
def search_child_chunks(query: str, k: int = 5) -> List[dict]:
    """Search for relevant child chunks using hybrid search.
    
    Args:
        query: The search query
        k: Number of results to return (default 5)
    
    Returns:
        List of relevant chunks with their parent IDs
    """
    try:
        results = vector_store.similarity_search(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "parent_id": doc.metadata.get("parent_id", ""),
                "source": doc.metadata.get("source", "")
            }
            for doc in results
        ]
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

@tool
def retrieve_parent_chunks(parent_ids: List[str]) -> List[dict]:
    """Retrieve full parent chunks from disk using their IDs.
    
    Args:
        parent_ids: List of parent chunk IDs
    
    Returns:
        List of parent chunks with full content
    """
    results = []
    
    # Remove duplicates
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