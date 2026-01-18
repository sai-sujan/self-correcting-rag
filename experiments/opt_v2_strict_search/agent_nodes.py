import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from core.config import llm
from experiments.opt_v2_strict_search.agent_state import AgentState

def summarization_agent(state: AgentState):
    """Summarizes history and prunes old messages"""
    messages = state["messages"]

    if len(messages) < 6:
        return {"conversation_summary": ""}

    human_ai_msgs = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]

    if len(human_ai_msgs) < 4:
        return {"conversation_summary": ""}

    messages_to_summarize = human_ai_msgs[:-4]

    if not messages_to_summarize:
        return {"conversation_summary": ""}

    prompt = "Summarize the key topics from this conversation in 1-2 sentences:\n\n"
    for msg in messages_to_summarize:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        prompt += f"{role}: {content}\n"

    prompt += "\nBrief Summary:"

    response = llm.invoke([SystemMessage(content=prompt)])
    summary = response.content

    messages_to_keep = human_ai_msgs[-4:]
    messages_to_delete = [
        RemoveMessage(id=m.id)
        for m in messages
        if m not in messages_to_keep and not isinstance(m, SystemMessage)
    ]

    print(f"📊 [Strict Search] Pruning {len(messages_to_delete)} old messages")

    return {
        "conversation_summary": summary,
        "messages": messages_to_delete
    }


def query_rewriter_agent(state: AgentState):
    """Rewrites unclear questions"""
    last_message = state["messages"][-1].content
    summary = state.get("conversation_summary", "")

    pronouns = ["it", "that", "this", "them", "those", "these"]
    has_pronoun = any(f" {p} " in f" {last_message.lower()} " for p in pronouns)

    if not has_pronoun:
        return {"question_is_clear": True}

    if not summary:
        return {
            "question_is_clear": False,
            "messages": [AIMessage(content="What specific topic are you asking about?")]
        }

    prompt = f"""Context: {summary}

User question: "{last_message}"

Rewrite the question to replace pronouns with specific terms from context.
Return ONLY the rewritten question.

Rewritten question:"""

    response = llm.invoke([SystemMessage(content=prompt)])
    rewritten = response.content.strip()

    return {
        "question_is_clear": True,
        "messages": [
            RemoveMessage(id=state["messages"][-1].id),
            HumanMessage(content=rewritten)
        ]
    }


def retrieval_agent(state: AgentState):
    """Uses tools to search and answer with STRICT search requirements"""
    from experiments.opt_v2_strict_search.tools import search_child_chunks, retrieve_parent_chunks

    llm_with_tools = llm.bind_tools([search_child_chunks, retrieve_parent_chunks])

    system_msg = SystemMessage(content="""You are a document search assistant with strict search requirements.

MANDATORY WORKFLOW - NO EXCEPTIONS:

For EVERY user question, you MUST:

1. ALWAYS call search_child_chunks FIRST
   - Extract 3-5 keywords from the question
   - Call search even if question is unclear
   - Call search even if you think you know the answer

2. IF search returns results:
   - ALWAYS call retrieve_parent_chunks to get full context
   - Use the full context to generate your answer

   IF search returns nothing:
   - Try again with different/broader keywords
   - If still nothing, explain you searched but found nothing

3. ONLY answer after completing steps 1 and 2

FORBIDDEN BEHAVIORS:
- Answering without calling search_child_chunks first
- Saying "I don't have information" without attempting search
- Using your training knowledge instead of searching documents
- Skipping retrieve_parent_chunks if search found results

EXAMPLES:

Example 1 - Clear question:
User: "What are JavaScript variables?"
You: [Calls search_child_chunks("JavaScript variables var let const")]
You: [Calls retrieve_parent_chunks with results]
You: [Answers using retrieved context]

Example 2 - Unclear question:
User: "How do I install it?"
You: [Calls search_child_chunks("install installation setup")]
You: [Calls retrieve_parent_chunks with results]
You: [Answers using context]

Example 3 - No results:
User: "What is Python?"
You: [Calls search_child_chunks("Python programming")]
You: No results → [Calls search_child_chunks("Python")]
You: Still no results → "I searched for 'Python' but found no information in the documents."

KEY PRINCIPLE: ALWAYS search. NEVER assume. Documents are your ONLY source of truth.""")

    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}
