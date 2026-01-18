from typing import TypedDict, Annotated
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    """
    State that flows through our multi-agent system.
    
    MessagesState gives us:
    - messages: List of conversation messages
    
    We add:
    - conversation_summary: Summary of past conversation
    - question_is_clear: Whether the question needs clarification
    """
    conversation_summary: str = ""
    question_is_clear: bool = True
"""

---

## What is "State"?

**Think of State like a shared notebook** that all agents can read and write to.

**Example flow:**

Initial State:
{
  "messages": ["What is JavaScript?"],
  "conversation_summary": "",
  "question_is_clear": True
}

After Summarization Agent:
{
  "messages": ["What is JavaScript?"],
  "conversation_summary": "User learning programming basics",  ← Added
  "question_is_clear": True
}

After Query Rewriter Agent:
{
  "messages": ["What is JavaScript and its features?"],  ← Updated
  "conversation_summary": "User learning programming basics",
  "question_is_clear": True
}
"""