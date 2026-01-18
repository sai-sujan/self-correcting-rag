from typing import TypedDict
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    """State for baseline experiment"""
    conversation_summary: str = ""
    question_is_clear: bool = True
