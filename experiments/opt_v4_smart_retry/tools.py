import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import os
import json
import re
from typing import List, Set, Tuple
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from core.config import (
    client,
    dense_embeddings,
    sparse_embeddings,
    reranker,
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

# =============================================================================
# CROSS-ENCODER RERANKING
# =============================================================================
def rerank_with_cross_encoder(docs: list, query: str, top_k: int = 5) -> List[Tuple[float, any]]:
    """
    Re-rank documents using cross-encoder for better semantic matching.

    Cross-encoders are more accurate than bi-encoders because they
    jointly encode query+document pairs, but are slower.

    Args:
        docs: Documents from initial retrieval
        query: Original query string
        top_k: Number of top results to return

    Returns:
        List of (score, doc) tuples, sorted by score descending
    """
    if not docs:
        return []

    # Create query-document pairs for cross-encoder
    pairs = [(query, doc.page_content[:512]) for doc in docs]  # Truncate to max_length

    # Score all pairs
    scores = reranker.predict(pairs)

    # Combine scores with documents
    scored_docs = list(zip(scores, docs))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return scored_docs[:top_k]


# =============================================================================
# HEADER-AWARE BOOSTING (lightweight, supplements cross-encoder)
# =============================================================================
HEADER_BOOST_WEIGHT = 0.1  # Reduced since cross-encoder is primary ranker
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
             "have", "has", "had", "do", "does", "did", "will", "would", "could",
             "should", "may", "might", "must", "shall", "can", "to", "of", "in",
             "for", "on", "with", "at", "by", "from", "as", "into", "through",
             "during", "before", "after", "above", "below", "between", "under",
             "again", "further", "then", "once", "here", "there", "when", "where",
             "why", "how", "all", "each", "few", "more", "most", "other", "some",
             "such", "no", "nor", "not", "only", "own", "same", "so", "than",
             "too", "very", "just", "and", "but", "if", "or", "because", "until",
             "while", "what", "which", "who", "whom", "this", "that", "these",
             "those", "it", "its", "i", "you", "he", "she", "we", "they", "me",
             "him", "her", "us", "them", "my", "your", "his", "our", "their"}


def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text, removing stopwords."""
    # Remove markdown formatting and special chars
    clean_text = re.sub(r'[*#_`\[\]()]', '', text.lower())
    # Split into words
    words = re.findall(r'\b[a-z]{2,}\b', clean_text)
    # Remove stopwords
    return {w for w in words if w not in STOPWORDS}


def calculate_header_score(query_keywords: Set[str], doc_metadata: dict) -> float:
    """
    Calculate header overlap score between query and document headers.

    Checks H1, H2, H3 headers for keyword overlap.
    Returns score between 0.0 and 1.0.
    """
    if not query_keywords:
        return 0.0

    header_keywords = set()

    # Extract keywords from all header levels
    for header_key in ["H1", "H2", "H3", "H4"]:
        header_text = doc_metadata.get(header_key, "")
        if header_text:
            header_keywords.update(extract_keywords(header_text))

    if not header_keywords:
        return 0.0

    # Calculate Jaccard-like overlap
    overlap = len(query_keywords & header_keywords)
    union = len(query_keywords | header_keywords)

    if union == 0:
        return 0.0

    return overlap / union


def apply_header_boost(scored_docs: List[Tuple[float, any]], query: str) -> List[Tuple[float, any]]:
    """
    Apply header boost on top of cross-encoder scores.

    Args:
        scored_docs: List of (cross_encoder_score, doc) tuples
        query: Original query string

    Returns:
        List of (final_score, doc) tuples with header boost applied
    """
    if not scored_docs:
        return []

    query_keywords = extract_keywords(query)

    boosted = []
    for ce_score, doc in scored_docs:
        header_score = calculate_header_score(query_keywords, doc.metadata)
        # Add small header boost to cross-encoder score
        final_score = ce_score + (header_score * HEADER_BOOST_WEIGHT)
        boosted.append((final_score, ce_score, header_score, doc))

    # Sort by final score
    boosted.sort(key=lambda x: x[0], reverse=True)

    # Log boosting for debugging
    for i, (final, ce, header, doc) in enumerate(boosted):
        if header > 0:
            print(f"   📈 [Header Boost] #{i+1} ce={ce:.3f} + header={header:.2f}×{HEADER_BOOST_WEIGHT} → {final:.3f}")

    return [(final, doc) for final, ce, header, doc in boosted]


@tool
def search_child_chunks(query: str, k: int = 5) -> List[dict]:
    """Search for relevant child chunks with cross-encoder reranking + header boost.

    RETRIEVAL PIPELINE:
    1. Hybrid search (dense + sparse) → top 10 candidates
    2. Cross-encoder reranking → semantic re-scoring
    3. Header boost → prioritize topic-aligned chunks
    4. Return top k with minimal data
    """
    try:
        # Step 1: Retrieve candidates (2x for reranking headroom)
        candidates = vector_store.similarity_search(query, k=k * 2)

        if not candidates:
            return []

        print(f"🔍 [Search] Retrieved {len(candidates)} candidates")

        # Step 2: Cross-encoder reranking (primary ranker)
        print(f"🎯 [Rerank] Cross-encoder scoring {len(candidates)} candidates...")
        ce_scored = rerank_with_cross_encoder(candidates, query, top_k=k + 2)

        # Log cross-encoder scores
        for i, (score, doc) in enumerate(ce_scored[:3]):
            print(f"   🎯 [CE] #{i+1} score={score:.3f}: {doc.page_content[:50]}...")

        # Step 3: Apply header boost (secondary signal)
        final_scored = apply_header_boost(ce_scored, query)

        # Take top k
        top_k_docs = [doc for score, doc in final_scored[:k]]

        return [
            {
                "parent_id": doc.metadata.get("parent_id", ""),
                "source": doc.metadata.get("source", ""),
                "snippet": doc.page_content[:250] + "...",
                "headers": {
                    key: val for key, val in doc.metadata.items()
                    if key.startswith("H") and key[1:].isdigit()
                }
            }
            for doc in top_k_docs
        ]
    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
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
