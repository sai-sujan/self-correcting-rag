"""
Experiment 7: Temperature Tuning - Compare different temperature settings

This experiment tests different temperature combinations to find optimal settings.
Based on v6 (fast multi-query with batched extraction).

Temperature configurations to test:
- Config A: llm=0.1, llm_small=0.0 (original - too strict)
- Config B: llm=0.3, llm_small=0.1 (current recommendation)
- Config C: llm=0.5, llm_small=0.2 (more creative)
- Config D: llm=0.3, llm_small=0.3 (balanced)
- Config E: llm=0.7, llm_small=0.1 (creative main, strict small)

Usage:
    from experiments.opt_v7_temperature_tuning.graph import create_graph

    # Create graph with specific temperature config
    agent_graph = create_graph(llm_temp=0.3, llm_small_temp=0.1)

    # Or use preset configs
    agent_graph = create_graph_with_config("B")  # llm=0.3, llm_small=0.1

Flow: START -> summarize -> rewrite -> multi_query_search -> retrieve_parents -> extract_relevant_batch -> generate_answer -> judge_answer -> [smart_retry_decision] -> END
"""
import sys
import os
import json
import re
import hashlib
from typing import Dict, Tuple, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_ollama import ChatOllama
from experiments.opt_v7_temperature_tuning.agent_state import AgentState
from experiments.opt_v7_temperature_tuning.tools import search_child_chunks, retrieve_parent_chunks

os.environ["LANGCHAIN_PROJECT"] = "self-correcting-rag"
EXPERIMENT_TAG = "opt-v7-temperature-tuning"

# =============================================================================
# TEMPERATURE CONFIGURATIONS
# =============================================================================
TEMPERATURE_CONFIGS = {
    "A": {"llm_temp": 0.1, "llm_small_temp": 0.0, "description": "Original (strict)"},
    "B": {"llm_temp": 0.3, "llm_small_temp": 0.1, "description": "Recommended (balanced)"},
    "C": {"llm_temp": 0.5, "llm_small_temp": 0.2, "description": "Creative"},
    "D": {"llm_temp": 0.3, "llm_small_temp": 0.3, "description": "Balanced both"},
    "E": {"llm_temp": 0.7, "llm_small_temp": 0.1, "description": "Creative main only"},
}

# Current active config (set via create_graph)
_current_llm = None
_current_llm_small = None
_current_config_name = "B"  # Default

def get_llm():
    global _current_llm
    if _current_llm is None:
        _current_llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=0.3
        )
    return _current_llm

def get_llm_small():
    global _current_llm_small
    if _current_llm_small is None:
        _current_llm_small = ChatOllama(
            model="qwen2.5:3b",
            temperature=0.1
        )
    return _current_llm_small

def set_temperatures(llm_temp: float, llm_small_temp: float, config_name: str = "custom"):
    """Set temperature values for both LLMs."""
    global _current_llm, _current_llm_small, _current_config_name

    _current_llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        temperature=llm_temp
    )
    _current_llm_small = ChatOllama(
        model="qwen2.5:3b",
        temperature=llm_small_temp
    )
    _current_config_name = config_name

    print(f"[Temperature] Config {config_name}: llm={llm_temp}, llm_small={llm_small_temp}")

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
    """Clear the retrieval cache (useful between experiments)."""
    global _retrieval_cache
    _retrieval_cache = {}

# =============================================================================
# SMART RETRY THRESHOLDS
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


# =============================================================================
# NODE 1: Summarization
# =============================================================================
def summarization_node(state: AgentState):
    """Summarizes history and prunes old messages"""
    messages = state.get("messages", [])

    if len(messages) < 6:
        return {"conversation_summary": "", "temperature_config": _current_config_name}

    human_ai_msgs = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]

    if len(human_ai_msgs) < 4:
        return {"conversation_summary": "", "temperature_config": _current_config_name}

    messages_to_summarize = human_ai_msgs[:-4]
    if not messages_to_summarize:
        return {"conversation_summary": "", "temperature_config": _current_config_name}

    prompt = "Summarize the key topics in 1-2 sentences:\n\n"
    for msg in messages_to_summarize:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        prompt += f"{role}: {msg.content[:200]}...\n"

    response = get_llm_small().invoke([SystemMessage(content=prompt)])

    messages_to_keep = human_ai_msgs[-4:]
    messages_to_delete = [
        RemoveMessage(id=m.id)
        for m in messages
        if m not in messages_to_keep and not isinstance(m, SystemMessage)
    ]

    return {
        "conversation_summary": response.content,
        "messages": messages_to_delete,
        "temperature_config": _current_config_name
    }


# =============================================================================
# NODE 2: Query Rewriter
# =============================================================================
def query_rewriter_node(state: AgentState):
    """Rewrites unclear questions using context"""
    messages = state.get("messages", [])
    if not messages:
        return {"question_is_clear": True}

    last_message = messages[-1].content
    summary = state.get("conversation_summary", "")

    pronouns = ["it", "that", "this", "them", "those", "these"]
    has_pronoun = any(f" {p} " in f" {last_message.lower()} " for p in pronouns)

    if not has_pronoun:
        return {"question_is_clear": True}

    if not summary:
        # If no history, assume the question is self-contained even if it has pronouns
        # (e.g. "What is X and how does it work?")
        return {"question_is_clear": True}

    prompt = f"""Context: {summary}
Question: "{last_message}"
Rewrite replacing pronouns with specific terms. Return ONLY the rewritten question."""

    response = get_llm_small().invoke([SystemMessage(content=prompt)])

    return {
        "question_is_clear": True,
        "messages": [
            RemoveMessage(id=messages[-1].id),
            HumanMessage(content=response.content.strip())
        ]
    }


# =============================================================================
# NODE 3: MULTI-QUERY SEARCH
# =============================================================================
def multi_query_search_node(state: AgentState):
    """Generate multiple search queries for better retrieval coverage."""
    messages = state.get("messages", [])
    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    if not last_human_msg:
        return {"search_results": [], "search_performed": True, "search_queries": []}

    retry_count = state.get("retry_count", 0)
    retry_strategy = state.get("retry_strategy", "")
    original_query = state.get("original_query", "")

    # Determine k based on retry
    if retry_count > 0:
        current_k = min(10, 5 + (retry_count * 3))
        print(f"[Retry #{retry_count}] Expanding k: {current_k}")
    else:
        current_k = state.get("retrieval_k", 5)

    # Generate multiple search queries using llm_small (faster)
    if retry_count > 0 and retry_strategy == "rewrite_query" and original_query:
        multi_query_prompt = f"""Generate 3 ALTERNATIVE search queries for: "{last_human_msg}"
Use different keywords, synonyms, or phrasings.
Return ONLY the queries, one per line."""
    else:
        multi_query_prompt = f"""Generate 2-3 search queries to find information for this question:
"{last_human_msg}"

If comparing two things, include a query for each.
Return ONLY the queries, one per line."""

    response = get_llm_small().invoke([SystemMessage(content=multi_query_prompt)])
    raw_queries = response.content.strip().split('\n')

    # Clean up queries
    queries = []
    for q in raw_queries:
        cleaned = re.sub(r'^[\d]+[.\)]\s*', '', q.strip())
        cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 2:
            queries.append(cleaned)

    if last_human_msg not in queries:
        queries.insert(0, last_human_msg)

    queries = queries[:3]

    print(f"[Multi-Query] Generated {len(queries)} queries:")
    for i, q in enumerate(queries):
        print(f"   {i+1}. {q}")

    # Execute all searches and merge results
    all_results = []
    seen_parent_ids = set()

    for query in queries:
        results = []
        if retry_count == 0:
            cached = get_cached_retrieval(query)
            if cached:
                results, _ = cached

        if not results:
            search_result = search_child_chunks.invoke({"query": query, "k": current_k})
            results = search_result if search_result else []

            if retry_count == 0 and results:
                parent_ids = list(set([r["parent_id"] for r in results if isinstance(r, dict) and "parent_id" in r]))
                cache_retrieval(query, results, parent_ids)

        for r in results:
            if isinstance(r, dict) and r.get("parent_id"):
                if r["parent_id"] not in seen_parent_ids:
                    seen_parent_ids.add(r["parent_id"])
                    all_results.append(r)

    print(f"[Multi-Query] Merged {len(all_results)} unique results from {len(queries)} queries")

    return {
        "search_results": all_results,
        "search_performed": True,
        "search_queries": queries,
        "search_query": queries[0] if queries else "",
        "original_query": last_human_msg if not original_query else original_query,
        "retrieval_k": current_k
    }


# =============================================================================
# NODE 4: Retrieve Parent Chunks
# =============================================================================
def retrieve_parents_node(state: AgentState):
    """Retrieves full parent documents"""
    search_results = state.get("search_results", [])

    if not search_results:
        return {"retrieved_docs": [], "parent_ids": []}

    parent_ids = list(set([
        r["parent_id"] for r in search_results
        if isinstance(r, dict) and "parent_id" in r
    ]))

    if not parent_ids:
        return {"retrieved_docs": [], "parent_ids": []}

    print(f"[Retrieve] Parent IDs: {parent_ids}")

    retrieved_docs = retrieve_parent_chunks.invoke({"parent_ids": parent_ids})

    return {"retrieved_docs": retrieved_docs, "parent_ids": parent_ids}


# =============================================================================
# NODE 5: BATCHED EXTRACTION
# =============================================================================
def extract_relevant_batch_node(state: AgentState):
    """Extract relevant content from ALL chunks in ONE LLM call."""
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])

    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    if not retrieved_docs or not last_human_msg:
        return {"extracted_content": "", "extraction_logs": []}

    # Build combined document text with clear separators
    combined_docs = ""
    for i, doc in enumerate(retrieved_docs):
        chunk_id = doc.get("parent_id", f"doc_{i}") if isinstance(doc, dict) else f"doc_{i}"
        content = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
        combined_docs += f"\n=== DOCUMENT {i+1} ({chunk_id}) ===\n{content[:1500]}\n"

    # Single batched extraction prompt
    extraction_prompt = f"""Extract ALL sentences from the documents below that help answer this question:

QUESTION: {last_human_msg}

DOCUMENTS:
{combined_docs[:6000]}

INSTRUCTIONS:
1. Find sentences that contain information about the question topic
2. Include definitions, explanations, facts, or examples
3. Copy exact sentences - do NOT paraphrase
4. Format as bullet points with "-"
5. If documents discuss the topic from different angles, include ALL relevant parts
6. If truly nothing relevant exists, return "- No relevant information found"

RELEVANT SENTENCES:"""

    try:
        response = get_llm_small().invoke([SystemMessage(content=extraction_prompt)])
        extracted = response.content.strip()

        # Check if extraction failed
        normalized = re.sub(r'[^a-z]', '', extracted.lower())
        if normalized in ["", "none", "norelevantinformationfound", "nothing"]:
            extracted = combined_docs[:2000] + "..."
            used_fallback = True
        elif len(extracted) < 50:
            extracted = extracted + "\n\n" + combined_docs[:1000] + "..."
            used_fallback = True
        else:
            used_fallback = False

        print(f"[Extract-Batch] {len(retrieved_docs)} docs -> {len(extracted)} chars {'(fallback)' if used_fallback else ''}")

        extraction_logs = [{
            "num_docs": len(retrieved_docs),
            "combined_length": len(combined_docs),
            "extracted_length": len(extracted),
            "used_fallback": used_fallback,
            "temperature_config": _current_config_name
        }]

    except Exception as e:
        extracted = combined_docs[:2000] + "..."
        extraction_logs = [{"error": str(e), "used_fallback": True}]
        print(f"[Extract-Batch] Error - using fallback")

    return {
        "extracted_content": extracted,
        "extraction_logs": extraction_logs
    }


# =============================================================================
# NODE 6: Generate Answer
# =============================================================================
def generate_answer_node(state: AgentState):
    """Generates answer using extracted content."""
    extracted_content = state.get("extracted_content", "")
    search_queries = state.get("search_queries", [])
    messages = state.get("messages", [])

    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    if not last_human_msg:
        return {"messages": [AIMessage(content="I couldn't understand your question.")]}

    if extracted_content and len(extracted_content) > 50:
        prompt = f"""Answer this question using ONLY the document excerpts provided.

DOCUMENT EXCERPTS:
{extracted_content[:4000]}

QUESTION: {last_human_msg}

INSTRUCTIONS:
1. Use ONLY information from the excerpts above
2. If excerpts contain relevant info, give a clear answer
3. If excerpts partially answer, share what IS available
4. Only say "documents don't contain this" if truly unrelated
5. Be concise but complete

ANSWER:"""
    else:
        prompt = f"""User asked: "{last_human_msg}"
Searched with: {search_queries}
No relevant content found.

Briefly explain that you searched but found no relevant information."""

    response = get_llm().invoke([SystemMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}


# =============================================================================
# LLM JUDGE FUNCTION
# =============================================================================
def llm_judge(question: str, context: str, answer: str) -> dict:
    """LLM-based judge that evaluates answer faithfulness."""
    judge_prompt = f"""Evaluate if the ANSWER is supported by the CONTEXT.

QUESTION: {question}

CONTEXT:
{context[:2500]}

ANSWER:
{answer}

For each statement in the answer, mark as SUPPORTED or UNSUPPORTED.
If answer says "documents don't contain info", mark SUPPORTED.

Return JSON:
{{"statements": [{{"text": "...", "verdict": "SUPPORTED"}}], "overall_verdict": "SUPPORTED|UNSUPPORTED"}}"""

    try:
        response = get_llm_small().invoke([SystemMessage(content=judge_prompt)])
        raw_response = response.content.strip()

        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            judgment = json.loads(json_match.group())
        else:
            judgment = {"statements": [], "overall_verdict": "SUPPORTED", "parse_error": True}

        return judgment

    except Exception as e:
        return {"statements": [], "overall_verdict": "ERROR", "error": str(e)}


# =============================================================================
# NODE 7: Judge Answer
# =============================================================================
def judge_answer_node(state: AgentState):
    """Judges if the generated answer is faithful."""
    extracted_content = state.get("extracted_content", "")
    messages = state.get("messages", [])

    question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            answer = msg.content
            break

    if not answer or not extracted_content:
        return {
            "judgment": {"overall_verdict": "NO_CONTEXT", "statements": []},
            "judgment_log": {"error": "Missing answer or context"}
        }

    judgment = llm_judge(question, extracted_content, answer)

    judgment_log = {
        "question": question,
        "context_length": len(extracted_content),
        "answer": answer,
        "raw_judgment": judgment,
        "temperature_config": _current_config_name
    }

    verdict = judgment.get("overall_verdict", "UNKNOWN")
    print(f"[Judge] Verdict: {verdict}")

    return {"judgment": judgment, "judgment_log": judgment_log}


# =============================================================================
# NODE 8: SMART RETRY DECISION
# =============================================================================
def should_retry(state: AgentState) -> bool:
    """Smart retry - does NOT retry on valid 'out of scope' responses."""
    judgment = state.get("judgment", {})
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    search_results = state.get("search_results", [])
    extracted = state.get("extracted_content", "")

    if retry_count >= max_retries:
        print(f"[Smart-Retry] Max retries ({max_retries}) reached")
        return False

    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            answer = msg.content.lower()
            break

    # Valid "out of scope" - don't retry
    if any(phrase in answer for phrase in VALID_NO_INFO_PHRASES):
        if search_results and len(search_results) > 0:
            print(f"[Smart-Retry] Valid 'out of scope' - no retry")
            return False

    # Trigger: No search results
    if not search_results:
        print(f"[Smart-Retry] Trigger: No search results")
        return True

    # Trigger: Extraction failed
    if search_results and (not extracted or len(extracted) < 50):
        print(f"[Smart-Retry] Trigger: Extraction failed")
        return True

    # Trigger: High hallucination
    verdict = judgment.get("overall_verdict", "")
    if verdict == "UNSUPPORTED" and extracted and len(extracted) >= 50:
        statements = judgment.get("statements", [])
        unsupported = sum(1 for s in statements if s.get("verdict") == "UNSUPPORTED")
        total = len(statements)
        if total > 0 and (unsupported / total) > HALLUCINATION_THRESHOLD:
            print(f"[Smart-Retry] Trigger: Hallucination ({unsupported}/{total})")
            return True

    return False


def self_correction_node(state: AgentState):
    """Prepare state for retry."""
    retry_count = state.get("retry_count", 0)
    judgment = state.get("judgment", {})
    extracted_content = state.get("extracted_content", "")
    messages = state.get("messages", [])

    if not extracted_content or len(extracted_content) < 50:
        strategy = "expand_k"
        reason = "No relevant documents"
    elif judgment.get("overall_verdict") == "UNSUPPORTED":
        strategy = "rewrite_query"
        reason = "Answer not supported"
    else:
        strategy = "expand_k"
        reason = "Low faithfulness"

    print(f"[Smart-Retry] Strategy: {strategy} | Reason: {reason}")

    messages_to_remove = []
    if messages and isinstance(messages[-1], AIMessage):
        messages_to_remove = [RemoveMessage(id=messages[-1].id)]

    return {
        "retry_count": retry_count + 1,
        "retry_strategy": strategy,
        "retry_reason": reason,
        "search_results": [],
        "retrieved_docs": [],
        "extracted_content": "",
        "judgment": {},
        "messages": messages_to_remove
    }


# =============================================================================
# ROUTING
# =============================================================================
def route_after_rewrite(state: AgentState):
    if state.get("question_is_clear", True):
        return "multi_query_search"
    return END


def route_after_judge(state: AgentState):
    if should_retry(state):
        return "self_correct"
    return END


# =============================================================================
# GRAPH FACTORY FUNCTIONS
# =============================================================================
def create_graph(llm_temp: float = 0.3, llm_small_temp: float = 0.1, config_name: str = "custom"):
    """Create a graph with specific temperature settings."""
    set_temperatures(llm_temp, llm_small_temp, config_name)
    clear_cache()  # Clear cache for fair comparison

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("summarize", summarization_node)
    graph_builder.add_node("rewrite", query_rewriter_node)
    graph_builder.add_node("multi_query_search", multi_query_search_node)
    graph_builder.add_node("retrieve_parents", retrieve_parents_node)
    graph_builder.add_node("extract_relevant", extract_relevant_batch_node)
    graph_builder.add_node("generate_answer", generate_answer_node)
    graph_builder.add_node("judge_answer", judge_answer_node)
    graph_builder.add_node("self_correct", self_correction_node)

    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "rewrite")
    graph_builder.add_conditional_edges("rewrite", route_after_rewrite)
    graph_builder.add_edge("multi_query_search", "retrieve_parents")
    graph_builder.add_edge("retrieve_parents", "extract_relevant")
    graph_builder.add_edge("extract_relevant", "generate_answer")
    graph_builder.add_edge("generate_answer", "judge_answer")
    graph_builder.add_conditional_edges("judge_answer", route_after_judge)
    graph_builder.add_edge("self_correct", "multi_query_search")

    checkpointer = MemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)


def create_graph_with_config(config_key: str):
    """Create a graph using a predefined temperature configuration."""
    if config_key not in TEMPERATURE_CONFIGS:
        raise ValueError(f"Unknown config: {config_key}. Available: {list(TEMPERATURE_CONFIGS.keys())}")

    config = TEMPERATURE_CONFIGS[config_key]
    return create_graph(
        llm_temp=config["llm_temp"],
        llm_small_temp=config["llm_small_temp"],
        config_name=config_key
    )


# Default graph (using recommended config B)
agent_graph = create_graph_with_config("B")

__all__ = [
    'agent_graph',
    'create_graph',
    'create_graph_with_config',
    'TEMPERATURE_CONFIGS',
    'EXPERIMENT_TAG',
    'llm_judge',
    'get_cached_retrieval',
    'cache_retrieval',
    'clear_cache',
    'set_temperatures'
]
