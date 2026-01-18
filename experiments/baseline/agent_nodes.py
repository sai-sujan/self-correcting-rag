import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from core.config import llm
from experiments.baseline.agent_state import AgentState

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

    print(f"📊 [Baseline] Pruning {len(messages_to_delete)} old messages")

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
    """Uses tools to search and answer"""
    from experiments.baseline.tools import search_child_chunks, retrieve_parent_chunks

    llm_with_tools = llm.bind_tools([search_child_chunks, retrieve_parent_chunks])

    system_msg = SystemMessage(content="""You are a helpful assistant.

Workflow:
1. Call search_child_chunks (k=5)
2. Collect ALL unique parent_ids from results
3. Call retrieve_parent_chunks with all parent_ids
4. Answer using retrieved information

If no relevant info found, say "I don't have information about that." """)

    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}
