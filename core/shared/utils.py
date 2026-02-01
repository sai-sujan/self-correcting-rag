import hashlib
from typing import Dict, Tuple, List

# =============================================================================
# RETRIEVAL CACHE
# =============================================================================
_retrieval_cache: Dict[str, Tuple[list, list]] = {}
CACHE_MAX_SIZE = 100

def _get_cache_key(query: str) -> str:
    normalized = query.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()

def cache_retrieval(query: str, search_results: list, parent_ids: list):
    global _retrieval_cache
    if len(_retrieval_cache) >= CACHE_MAX_SIZE:
        keys = list(_retrieval_cache.keys())
        for k in keys[:len(keys)//2]:
            del _retrieval_cache[k]
    key = _get_cache_key(query)
    _retrieval_cache[key] = (search_results, parent_ids)

def get_cached_retrieval(query: str) -> Tuple[list, list] | None:
    key = _get_cache_key(query)
    if key in _retrieval_cache:
        search_results, parent_ids = _retrieval_cache[key]
        print(f"[Cache HIT] {query[:30]}...")
        return search_results, parent_ids
    return None

def clear_cache():
    """Clear the retrieval cache."""
    global _retrieval_cache
    _retrieval_cache = {}

# =============================================================================
# CONSTANTS & THRESHOLDS
# =============================================================================
HALLUCINATION_THRESHOLD = 0.7

RETRIEVAL_FAILURE_PHRASES = [
    "couldn't find any documents",
    "search returned no results",
    "unable to retrieve",
    "retrieval failed"
]

VALID_NO_INFO_PHRASES = [
    "documents don't contain",
    "not covered in the documents",
    "no relevant information found",
    "outside the scope"
]
