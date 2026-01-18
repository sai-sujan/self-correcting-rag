import sys
sys.path.append('../..')

import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from experiments.opt_v1_minimal_chunks.agent_state import AgentState
from experiments.opt_v1_minimal_chunks.agent_nodes import summarization_agent, query_rewriter_agent, retrieval_agent
from experiments.opt_v1_minimal_chunks.tools import search_child_chunks, retrieve_parent_chunks

# ONE project for all experiments
os.environ["LANGCHAIN_PROJECT"] = "self-correcting-rag"

# Tag this experiment
EXPERIMENT_TAG = "opt-v1-minimal-chunks"

def tool_executor_node(state: AgentState):
    """Execute tool calls"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    tools_map = {
        "search_child_chunks": search_child_chunks,
        "retrieve_parent_chunks": retrieve_parent_chunks
    }
    
    tool_messages = []
    for tool_call in tool_calls:
        result = tools_map[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    
    return {"messages": tool_messages}

# Create graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node("summarize", summarization_agent)
graph_builder.add_node("rewrite", query_rewriter_agent)
graph_builder.add_node("retrieve", retrieval_agent)
graph_builder.add_node("tools", tool_executor_node)

graph_builder.add_edge(START, "summarize")
graph_builder.add_edge("summarize", "rewrite")

def route_after_rewrite(state: AgentState):
    if state.get("question_is_clear", True):
        return "retrieve"
    else:
        return END

graph_builder.add_conditional_edges("rewrite", route_after_rewrite)

def tools_condition(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges("retrieve", tools_condition)
graph_builder.add_edge("tools", "retrieve")

checkpointer = MemorySaver()
agent_graph = graph_builder.compile(checkpointer=checkpointer)

__all__ = ['agent_graph', 'EXPERIMENT_TAG']