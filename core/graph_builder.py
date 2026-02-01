"""
Graph Builder - Constructs RAG graphs based on experiment configuration.

Instead of 600+ lines per experiment, graphs are built dynamically from config.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from core.shared.state import AgentState
from core.shared.nodes import (
    summarization_node,
    query_rewriter_node,
    single_query_search_node,
    multi_query_search_node,
    retrieve_parents_node,
    extract_relevant_batch_node,
    generate_answer_node,
    generate_answer_simple_node,
    judge_answer_node,
    self_correction_node,
    should_retry,
)
from core.shared.utils import clear_cache
from core.llm_manager import set_temperatures
from core.experiment_config import (
    ExperimentConfig,
    SearchMode,
    ExtractionMode,
    RetryMode,
    DEFAULT_CONFIG
)


def _route_after_rewrite(state: AgentState):
    """Route after query rewriting."""
    return "search" if state.get("question_is_clear", True) else END


def _route_after_judge(state: AgentState):
    """Route after judgment - retry or end."""
    return "self_correct" if should_retry(state) else END


def build_graph(config: ExperimentConfig = None):
    """
    Build a RAG graph based on experiment configuration.

    Args:
        config: ExperimentConfig (defaults to OPT_V7_D if None)

    Returns:
        Compiled LangGraph with checkpointer
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Set LLM temperatures
    set_temperatures(config.llm_temp, config.llm_small_temp)
    clear_cache()

    print(f"[Graph] {config.name}: {config.description}")
    print(f"        Temps: {config.llm_temp}/{config.llm_small_temp}, Search: {config.search_mode.value}")

    # Build graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("summarize", summarization_node)
    builder.add_node("rewrite", query_rewriter_node)

    # Search node (single or multi-query)
    if config.search_mode == SearchMode.MULTI_QUERY:
        builder.add_node("search", multi_query_search_node)
    else:
        builder.add_node("search", single_query_search_node)

    builder.add_node("retrieve_parents", retrieve_parents_node)

    # Extraction + Answer nodes
    if config.extraction_mode == ExtractionMode.BATCHED:
        builder.add_node("extract_relevant", extract_relevant_batch_node)
        builder.add_node("generate_answer", generate_answer_node)
    else:
        builder.add_node("generate_answer", generate_answer_simple_node)

    # Judge + Retry nodes (if enabled)
    if config.retry_mode == RetryMode.SMART:
        builder.add_node("judge_answer", judge_answer_node)
        builder.add_node("self_correct", self_correction_node)

    # Build edges
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "rewrite")
    builder.add_conditional_edges("rewrite", _route_after_rewrite)
    builder.add_edge("search", "retrieve_parents")

    if config.extraction_mode == ExtractionMode.BATCHED:
        builder.add_edge("retrieve_parents", "extract_relevant")
        builder.add_edge("extract_relevant", "generate_answer")
    else:
        builder.add_edge("retrieve_parents", "generate_answer")

    if config.retry_mode == RetryMode.SMART:
        builder.add_edge("generate_answer", "judge_answer")
        builder.add_conditional_edges("judge_answer", _route_after_judge)
        builder.add_edge("self_correct", "search")
    else:
        builder.add_edge("generate_answer", END)

    return builder.compile(checkpointer=MemorySaver())


def build_graph_by_name(name: str):
    """Build graph by experiment name (e.g., 'opt-v7-D')."""
    from core.experiment_config import get_experiment
    return build_graph(get_experiment(name))
