from typing import TypedDict, List, Any, Dict
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    """State for reliable extraction experiment - fixes extraction and multi-query search"""
    conversation_summary: str = ""
    question_is_clear: bool = True
    # Fields for multi-query search
    search_queries: List[str] = []  # Multiple queries for better retrieval
    search_results: List[Any] = []
    search_performed: bool = False
    search_query: str = ""
    retrieved_docs: List[Any] = []
    parent_ids: List[str] = []
    # Fields for query-focused extraction (using stronger LLM)
    extracted_content: str = ""
    extraction_logs: List[Dict] = []
    # Fields for LLM judge
    judgment: Dict = {}
    judgment_log: Dict = {}
    # Self-correction fields
    retry_count: int = 0
    max_retries: int = 2
    retry_reason: str = ""
    retry_strategy: str = ""
    original_query: str = ""
    retrieval_k: int = 5
