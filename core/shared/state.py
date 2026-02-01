"""Unified AgentState for all RAG experiments."""
from typing import List, Any, Dict
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    Unified state for all RAG experiments.

    Inherits 'messages' from MessagesState.
    Only includes fields that are actually used.
    """
    # Conversation context
    conversation_summary: str = ""
    question_is_clear: bool = True

    # Search & Retrieval
    search_queries: List[str] = []
    search_results: List[Any] = []
    retrieved_docs: List[Any] = []
    original_query: str = ""
    retrieval_k: int = 5

    # Extraction
    extracted_content: str = ""

    # Judgment & Self-Correction
    judgment: Dict = {}
    retry_count: int = 0
    max_retries: int = 2
    retry_strategy: str = ""
