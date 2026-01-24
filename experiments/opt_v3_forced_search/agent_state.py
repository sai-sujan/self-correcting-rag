from typing import TypedDict, List, Any, Dict
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    """State for forced search experiment - includes search results in state"""
    conversation_summary: str = ""
    question_is_clear: bool = True
    # New fields for forced search flow
    search_results: List[Any] = []
    search_performed: bool = False
    search_query: str = ""
    retrieved_docs: List[Any] = []
    parent_ids: List[str] = []
    # New fields for query-focused extraction
    extracted_content: str = ""
    extraction_logs: List[Dict] = []
    # New fields for LLM judge
    judgment: Dict = {}  # JSON verdict from judge
    judgment_log: Dict = {}  # Debug info (context, answer, raw response)
    # Self-correction fields
    retry_count: int = 0  # Number of retry attempts
    max_retries: int = 2  # Maximum retry attempts allowed
    retry_reason: str = ""  # Why retry was triggered
    retry_strategy: str = ""  # Which strategy was used (expand_k, rewrite_query, switch_mode)
    original_query: str = ""  # Store original query for rewrites
    retrieval_k: int = 5  # Current k value for retrieval
