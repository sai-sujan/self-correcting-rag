"""
Experiment 5: Reliable Extraction - Fixes extraction and multi-query search

Key differences from v4:
1. MULTI-QUERY SEARCH: Generates 2-3 search queries for better retrieval coverage
2. STRONGER EXTRACTION: Uses main LLM instead of llm_small for reliable extraction
3. BETTER PROMPTS: More specific extraction instructions
4. IMPROVED ANSWER GENERATION: Uses extracted content properly

Problems with v4:
- Extraction with llm_small (qwen2.5:3b) was too weak - often returned "NONE"
- Single keyword search missed conceptual questions ("difference between X and Y")
- "documents don't contain" even when relevant docs were retrieved

Solution in v5:
- Generate multiple search queries (keywords + conceptual)
- Use main LLM for extraction (more reliable)
- Better prompts for extraction and answer generation
- Merge results from multiple searches

Flow: START -> summarize -> rewrite -> multi_query_search -> retrieve_parents -> extract_relevant -> generate_answer -> judge_answer -> [smart_retry_decision] -> END
"""
import sys
import os
import json
import re
import hashlib
from typing import Dict, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from experiments.opt_v5_reliable_extraction.agent_state import AgentState
from experiments.opt_v5_reliable_extraction.tools import search_child_chunks, retrieve_parent_chunks
from core.config import llm, llm_small

os.environ["LANGCHAIN_PROJECT"] = "self-correcting-rag"
EXPERIMENT_TAG = "opt-v5-reliable-extraction"

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
        return {"conversation_summary": ""}

    human_ai_msgs = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]

    if len(human_ai_msgs) < 4:
        return {"conversation_summary": ""}

    messages_to_summarize = human_ai_msgs[:-4]
    if not messages_to_summarize:
        return {"conversation_summary": ""}

    prompt = "Summarize the key topics in 1-2 sentences:\n\n"
    for msg in messages_to_summarize:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        prompt += f"{role}: {msg.content[:200]}...\n"

    response = llm.invoke([SystemMessage(content=prompt)])

    messages_to_keep = human_ai_msgs[-4:]
    messages_to_delete = [
        RemoveMessage(id=m.id)
        for m in messages
        if m not in messages_to_keep and not isinstance(m, SystemMessage)
    ]

    return {"conversation_summary": response.content, "messages": messages_to_delete}


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
        return {
            "question_is_clear": False,
            "messages": [AIMessage(content="What specific topic are you asking about?")]
        }

    prompt = f"""Context: {summary}
Question: "{last_message}"
Rewrite replacing pronouns with specific terms. Return ONLY the rewritten question."""

    response = llm.invoke([SystemMessage(content=prompt)])

    return {
        "question_is_clear": True,
        "messages": [
            RemoveMessage(id=messages[-1].id),
            HumanMessage(content=response.content.strip())
        ]
    }


# =============================================================================
# NODE 3: MULTI-QUERY SEARCH (key improvement)
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

    # Generate multiple search queries
    # On retry with rewrite_query strategy, use different prompt
    if retry_count > 0 and retry_strategy == "rewrite_query" and original_query:
        multi_query_prompt = f"""The previous search for "{original_query}" didn't find good results.
Generate 3 ALTERNATIVE search queries using different keywords, synonyms, or approaches.

Original question: {last_human_msg}

Think about:
1. Different terminology or synonyms
2. More specific or more general terms
3. Related concepts that might lead to the answer

Return ONLY the queries, one per line."""
    else:
        multi_query_prompt = f"""Generate 2-3 different search queries to find information for this question.

QUESTION: {last_human_msg}

Generate queries that:
1. Extract the main concepts as keywords
2. Rephrase as a definition lookup (e.g., "what is X")
3. If comparing things, include queries for each thing

Return ONLY the queries, one per line."""

    response = llm.invoke([SystemMessage(content=multi_query_prompt)])
    raw_queries = response.content.strip().split('\n')

    # Clean up queries - remove numbering, bullets, empty lines
    queries = []
    for q in raw_queries:
        # Remove common prefixes like "1.", "1)", "-", "*", etc.
        cleaned = re.sub(r'^[\d]+[.\)]\s*', '', q.strip())
        cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 2:
            queries.append(cleaned)

    # Also add the original question as a query (if not already there)
    if last_human_msg not in queries:
        queries.insert(0, last_human_msg)

    # Limit to 3 queries
    queries = queries[:3]

    print(f"[Multi-Query] Generated {len(queries)} queries:")
    for i, q in enumerate(queries):
        print(f"   {i+1}. {q}")

    # Execute all searches and merge results
    all_results = []
    seen_parent_ids = set()

    for query in queries:
        # Check cache (skip on retry to get fresh results)
        results = []
        if retry_count == 0:
            cached = get_cached_retrieval(query)
            if cached:
                results, _ = cached

        # If not cached or on retry, execute search
        if not results:
            search_result = search_child_chunks.invoke({"query": query, "k": current_k})
            results = search_result if search_result else []

            # Cache results (only on first attempt)
            if retry_count == 0 and results:
                parent_ids = list(set([r["parent_id"] for r in results if isinstance(r, dict) and "parent_id" in r]))
                cache_retrieval(query, results, parent_ids)

        # Merge results (dedupe by parent_id)
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
# NODE 5: Query-Focused Extraction (using MAIN LLM)
# =============================================================================
MIN_EXTRACTION_LENGTH = 50

def extract_relevant_node(state: AgentState):
    """Extract relevant sentences using the MAIN LLM (more reliable than llm_small)."""
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])

    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    if not retrieved_docs or not last_human_msg:
        return {"extracted_content": "", "extraction_logs": []}

    extracted_parts = []
    extraction_logs = []

    for doc in retrieved_docs:
        chunk_id = doc.get("parent_id", "unknown") if isinstance(doc, dict) else "unknown"
        original_text = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)

        # Better extraction prompt with clearer instructions
        extraction_prompt = f"""Your task is to extract sentences from the TEXT that help answer the QUESTION.

QUESTION: {last_human_msg}

TEXT:
{original_text[:2500]}

INSTRUCTIONS:
1. Find ALL sentences that contain information related to the question
2. Include definitions, explanations, examples, or facts that answer the question
3. Copy the exact sentences - do NOT paraphrase or summarize
4. Format as bullet points starting with "-"
5. If the text discusses the topic but from a different angle, INCLUDE it
6. If there is truly nothing relevant, return "- NONE"

EXTRACTED RELEVANT SENTENCES:"""

        try:
            # Use MAIN LLM for reliable extraction (not llm_small)
            response = llm.invoke([SystemMessage(content=extraction_prompt)])
            extracted = response.content.strip()

            used_fallback = False
            # Check if extraction is empty or just "NONE" in various formats
            # Normalize: lowercase, remove punctuation/whitespace
            normalized = re.sub(r'[^a-z]', '', extracted.lower())
            if normalized in ["", "none", "nonone", "nothing", "norelevant", "norelevantinformation"]:
                # Include first 500 chars as fallback context
                extracted = original_text[:500] + "..."
                used_fallback = True
            elif len(extracted) < MIN_EXTRACTION_LENGTH:
                # Extraction too short - supplement with original
                extracted = extracted + "\n\n" + original_text[:300] + "..."
                used_fallback = True

            extracted_parts.append(extracted)

            extraction_logs.append({
                "chunk_id": chunk_id,
                "original_length": len(original_text),
                "extracted_length": len(extracted),
                "used_fallback": used_fallback,
                "extracted_preview": extracted[:200] + "..." if len(extracted) > 200 else extracted
            })

            status = "(fallback)" if used_fallback else ""
            print(f"[Extract] {chunk_id}: {len(original_text)} -> {len(extracted)} chars {status}")

        except Exception as e:
            fallback = original_text[:500] + "..."
            extracted_parts.append(fallback)
            extraction_logs.append({
                "chunk_id": chunk_id,
                "error": str(e),
                "used_fallback": True
            })
            print(f"[Extract] {chunk_id}: Error - using fallback")

    extracted_content = "\n\n---\n\n".join(extracted_parts)

    print(f"[Extract] Total: {len(extracted_content)} chars from {len(retrieved_docs)} chunks")

    return {
        "extracted_content": extracted_content,
        "extraction_logs": extraction_logs
    }


# =============================================================================
# NODE 6: Generate Answer (improved prompt)
# =============================================================================
def generate_answer_node(state: AgentState):
    """Generates answer using extracted content with improved prompt."""
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
        prompt = f"""You are a helpful assistant that answers questions using ONLY the provided document excerpts.

DOCUMENT EXCERPTS:
{extracted_content[:5000]}

USER QUESTION: {last_human_msg}

INSTRUCTIONS:
1. Answer the question using ONLY information from the document excerpts above
2. If the excerpts contain relevant information, provide a clear and complete answer
3. If the excerpts discuss the topic but don't fully answer the question, share what IS available
4. Only say "The documents don't contain this information" if the excerpts are truly unrelated to the question
5. Be concise but thorough

ANSWER:"""
    else:
        prompt = f"""User asked: "{last_human_msg}"
Searched using queries: {search_queries}
No relevant content found in documents.

Respond briefly that you searched the documents but found no relevant information for this specific question."""

    response = llm.invoke([SystemMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}


# =============================================================================
# LLM JUDGE FUNCTION
# =============================================================================
def llm_judge(question: str, context: str, answer: str) -> dict:
    """LLM-based judge that evaluates answer faithfulness to context."""
    judge_prompt = f"""You are a strict factual judge. Evaluate if the ANSWER is supported by the CONTEXT.

QUESTION: {question}

CONTEXT:
{context[:3000]}

ANSWER:
{answer}

INSTRUCTIONS:
1. Break the ANSWER into individual factual statements
2. For EACH statement, determine if it is:
   - SUPPORTED: Directly stated or clearly implied in CONTEXT
   - PARTIALLY_SUPPORTED: Some parts are in CONTEXT but not complete
   - UNSUPPORTED: NOT found in CONTEXT at all

RULES:
- Use ONLY the CONTEXT provided - no external knowledge
- Be CONSERVATIVE: if unsure, mark as UNSUPPORTED
- If answer says "documents don't contain information", mark as SUPPORTED (honest admission)

OUTPUT FORMAT (JSON only):
{{
  "statements": [
    {{"text": "<statement>", "verdict": "SUPPORTED"}},
    {{"text": "<statement>", "verdict": "UNSUPPORTED"}}
  ],
  "overall_verdict": "SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED"
}}

OVERALL_VERDICT logic:
- SUPPORTED: ALL statements are SUPPORTED
- PARTIALLY_SUPPORTED: Mix of verdicts
- UNSUPPORTED: ALL or MOST statements are UNSUPPORTED

Return ONLY valid JSON:"""

    try:
        response = llm_small.invoke([SystemMessage(content=judge_prompt)])
        raw_response = response.content.strip()

        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            json_str = json_match.group()
            judgment = json.loads(json_str)
        else:
            judgment = {
                "statements": [{"text": answer, "verdict": "UNSUPPORTED"}],
                "overall_verdict": "UNSUPPORTED",
                "parse_error": "Could not extract JSON"
            }

        return judgment

    except json.JSONDecodeError as e:
        return {
            "statements": [{"text": answer, "verdict": "UNSUPPORTED"}],
            "overall_verdict": "UNSUPPORTED",
            "parse_error": f"JSON decode error: {str(e)}"
        }
    except Exception as e:
        return {
            "statements": [],
            "overall_verdict": "ERROR",
            "error": str(e)
        }


# =============================================================================
# NODE 7: Judge Answer
# =============================================================================
def judge_answer_node(state: AgentState):
    """Judges if the generated answer is faithful to the retrieved context."""
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
        "context_preview": (extracted_content[:500] + "...") if len(extracted_content) > 500 else extracted_content,
        "answer": answer,
        "raw_judgment": judgment
    }

    verdict = judgment.get("overall_verdict", "UNKNOWN")
    num_statements = len(judgment.get("statements", []))
    print(f"[Judge] Verdict: {verdict} ({num_statements} statements)")

    return {
        "judgment": judgment,
        "judgment_log": judgment_log
    }


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
            print(f"[Smart-Retry] Valid 'out of scope' response - no retry")
            return False

    # Trigger: No search results
    if not search_results or len(search_results) == 0:
        print(f"[Smart-Retry] Trigger: Search returned 0 results")
        return True

    # Trigger: Search found results but extraction failed
    if search_results and len(search_results) > 0 and (not extracted or len(extracted) < 50):
        print(f"[Smart-Retry] Trigger: Extraction failed")
        return True

    # Trigger: High hallucination
    verdict = judgment.get("overall_verdict", "")
    if verdict == "UNSUPPORTED" and extracted and len(extracted) >= 50:
        statements = judgment.get("statements", [])
        unsupported = sum(1 for s in statements if s.get("verdict") == "UNSUPPORTED")
        total = len(statements)
        if total > 0:
            unsupported_ratio = unsupported / total
            if unsupported_ratio > HALLUCINATION_THRESHOLD:
                print(f"[Smart-Retry] Trigger: High hallucination ({unsupported}/{total})")
                return True

    # Trigger: Explicit retrieval failure
    for phrase in RETRIEVAL_FAILURE_PHRASES:
        if phrase in answer:
            print(f"[Smart-Retry] Trigger: Retrieval failure phrase")
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
        reason = "No relevant documents found"
    elif judgment.get("overall_verdict") == "UNSUPPORTED":
        strategy = "rewrite_query"
        reason = "Answer not supported by context"
    else:
        strategy = "expand_k"
        reason = "Low faithfulness score"

    print(f"[Smart-Retry] Strategy: {strategy} | Reason: {reason}")

    # Remove the last AI message (the bad answer) before retry
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
# BUILD GRAPH
# =============================================================================
graph_builder = StateGraph(AgentState)

graph_builder.add_node("summarize", summarization_node)
graph_builder.add_node("rewrite", query_rewriter_node)
graph_builder.add_node("multi_query_search", multi_query_search_node)
graph_builder.add_node("retrieve_parents", retrieve_parents_node)
graph_builder.add_node("extract_relevant", extract_relevant_node)
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
agent_graph = graph_builder.compile(checkpointer=checkpointer)

__all__ = ['agent_graph', 'EXPERIMENT_TAG', 'llm_judge', 'get_cached_retrieval', 'cache_retrieval']
