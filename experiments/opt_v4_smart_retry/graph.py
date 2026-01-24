"""
Experiment 5: Smart Retry - Improved Self-Correction Logic

Key differences from v3:
1. SMART RETRY: Does NOT retry on valid "out of scope" responses
2. Only retries when retrieval genuinely failed or high hallucination detected
3. Improved answer_quality evaluation that recognizes valid "no info" responses
4. Same core architecture: forced search + extraction + judge + caching

Problem with v3:
- Self-correction over-triggered on valid "documents don't contain" responses
- This wasted compute and didn't improve answers for out-of-scope questions
- answer_quality penalized honest "no info" responses

Solution in v4:
- Recognize "documents don't contain" as a VALID response when docs were searched
- Only retry on actual retrieval failure (0 results) or high hallucination (>70%)
- Don't retry on "no info" phrases when search DID return results

Flow: START → summarize → rewrite → forced_search → retrieve_parents → extract_relevant → generate_answer → judge_answer → [smart_retry_decision] → END
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
from experiments.opt_v4_smart_retry.agent_state import AgentState
from experiments.opt_v4_smart_retry.tools import search_child_chunks, retrieve_parent_chunks
from core.config import llm, llm_small

os.environ["LANGCHAIN_PROJECT"] = "self-correcting-rag"
EXPERIMENT_TAG = "opt-v4-smart-retry"

# =============================================================================
# RETRIEVAL CACHE - Stores query → parent_ids mapping
# =============================================================================
_retrieval_cache: Dict[str, Tuple[list, list]] = {}
CACHE_MAX_SIZE = 100

def _get_cache_key(query: str) -> str:
    """Generate cache key from query (normalized)"""
    normalized = query.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()

def cache_retrieval(query: str, search_results: list, parent_ids: list):
    """Cache retrieval results for a query"""
    global _retrieval_cache
    if len(_retrieval_cache) >= CACHE_MAX_SIZE:
        keys = list(_retrieval_cache.keys())
        for k in keys[:len(keys)//2]:
            del _retrieval_cache[k]

    key = _get_cache_key(query)
    _retrieval_cache[key] = (search_results, parent_ids)
    print(f"💾 [Cache] Stored: {query[:30]}... → {len(parent_ids)} parents")

def get_cached_retrieval(query: str) -> Tuple[list, list] | None:
    """Get cached retrieval results if available"""
    key = _get_cache_key(query)
    if key in _retrieval_cache:
        search_results, parent_ids = _retrieval_cache[key]
        print(f"⚡ [Cache HIT] {query[:30]}... → {len(parent_ids)} parents")
        return search_results, parent_ids
    return None

# =============================================================================
# SMART RETRY THRESHOLDS (key improvement over v3)
# =============================================================================
HALLUCINATION_THRESHOLD = 0.7  # Only retry if >70% statements unsupported (stricter than v3's 50%)

# IMPORTANT: We should NOT retry when documents genuinely don't have the answer
# These phrases indicate retrieval FAILED, not "out of scope"
RETRIEVAL_FAILURE_PHRASES = [
    "couldn't find any documents",
    "search returned no results",
    "unable to retrieve",
    "retrieval failed"
]

# These are VALID responses - do NOT trigger retry
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
    messages = state["messages"]

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
Question: "{last_message}"
Rewrite replacing pronouns with specific terms. Return ONLY the rewritten question."""

    response = llm.invoke([SystemMessage(content=prompt)])

    return {
        "question_is_clear": True,
        "messages": [
            RemoveMessage(id=state["messages"][-1].id),
            HumanMessage(content=response.content.strip())
        ]
    }


# =============================================================================
# NODE 3: FORCED SEARCH
# =============================================================================
def forced_search_node(state: AgentState):
    """ALWAYS searches - graph structure guarantees this runs."""
    last_human_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    if not last_human_msg:
        return {"search_results": [], "search_performed": True, "search_query": ""}

    retry_count = state.get("retry_count", 0)
    retry_strategy = state.get("retry_strategy", "")
    original_query = state.get("original_query", "")

    # Determine search parameters based on retry strategy
    if retry_count > 0 and retry_strategy == "expand_k":
        current_k = min(10, 5 + (retry_count * 3))  # 5 → 8 → 10
        print(f"🔄 [Retry #{retry_count}] Expanding k: {current_k}")
    else:
        current_k = state.get("retrieval_k", 5)

    # Generate or rewrite query
    if retry_count > 0 and retry_strategy == "rewrite_query" and original_query:
        rewrite_prompt = f"""The original query "{original_query}" didn't find good results.
Generate an ALTERNATIVE search query using different keywords/synonyms.
Return ONLY the new keywords separated by spaces."""
        response = llm.invoke([SystemMessage(content=rewrite_prompt)])
        keywords = response.content.strip()
        print(f"🔄 [Retry #{retry_count}] Rewritten query: {keywords}")
    else:
        prompt = f"Extract 3-5 search keywords from: {last_human_msg}\nReturn ONLY keywords separated by spaces."
        response = llm.invoke([SystemMessage(content=prompt)])
        keywords = response.content.strip()

    print(f"🔍 [Forced Search] Query: {keywords} (k={current_k})")

    # Check cache first (skip on retry)
    if retry_count == 0:
        cached = get_cached_retrieval(keywords)
        if cached:
            search_results, _ = cached
            return {
                "search_results": search_results,
                "search_performed": True,
                "search_query": keywords,
                "original_query": keywords if not original_query else original_query,
                "retrieval_k": current_k
            }

    # Execute search
    search_results = search_child_chunks.invoke({"query": keywords, "k": current_k})

    # Cache results (only on first attempt)
    if retry_count == 0 and search_results:
        parent_ids = list(set([
            r["parent_id"] for r in search_results
            if isinstance(r, dict) and "parent_id" in r
        ]))
        cache_retrieval(keywords, search_results, parent_ids)

    return {
        "search_results": search_results,
        "search_performed": True,
        "search_query": keywords,
        "original_query": keywords if not original_query else original_query,
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

    print(f"📄 [Retrieve] Parent IDs: {parent_ids}")

    retrieved_docs = retrieve_parent_chunks.invoke({"parent_ids": parent_ids})

    return {"retrieved_docs": retrieved_docs, "parent_ids": parent_ids}


# =============================================================================
# NODE 5: Query-Focused Extraction
# =============================================================================
MIN_EXTRACTION_LENGTH = 50

def extract_relevant_node(state: AgentState):
    """Extract only sentences that directly answer the question."""
    retrieved_docs = state.get("retrieved_docs", [])

    last_human_msg = None
    for msg in reversed(state["messages"]):
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

        extraction_prompt = f"""Extract ONLY the sentences from this text that directly answer or define the question.

QUESTION: {last_human_msg}

TEXT:
{original_text[:2000]}

RULES:
- Copy exact sentences from the text - do NOT paraphrase
- Only include sentences that directly answer the question
- Return as bullet points
- If nothing is relevant, return "NONE"

EXTRACTED SENTENCES:"""

        try:
            response = llm_small.invoke([SystemMessage(content=extraction_prompt)])
            extracted = response.content.strip()

            used_fallback = False
            if extracted == "NONE" or len(extracted) < MIN_EXTRACTION_LENGTH:
                extracted = original_text[:300] + "..."
                used_fallback = True

            extracted_parts.append(extracted)

            extraction_logs.append({
                "chunk_id": chunk_id,
                "original_length": len(original_text),
                "extracted_length": len(extracted),
                "used_fallback": used_fallback,
                "extracted_preview": extracted[:200] + "..." if len(extracted) > 200 else extracted
            })

            print(f"📝 [Extract] {chunk_id}: {len(original_text)} → {len(extracted)} chars {'(fallback)' if used_fallback else ''}")

        except Exception as e:
            fallback = original_text[:300] + "..."
            extracted_parts.append(fallback)
            extraction_logs.append({
                "chunk_id": chunk_id,
                "error": str(e),
                "used_fallback": True
            })
            print(f"⚠️ [Extract] {chunk_id}: Error - using fallback")

    extracted_content = "\n\n---\n\n".join(extracted_parts)

    print(f"📊 [Extract] Total: {len(extracted_content)} chars from {len(retrieved_docs)} chunks")

    return {
        "extracted_content": extracted_content,
        "extraction_logs": extraction_logs
    }


# =============================================================================
# NODE 6: Generate Answer
# =============================================================================
def generate_answer_node(state: AgentState):
    """Generates answer using extracted content"""
    extracted_content = state.get("extracted_content", "")
    search_query = state.get("search_query", "")

    last_human_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    if not last_human_msg:
        return {"messages": [AIMessage(content="I couldn't understand your question.")]}

    if extracted_content:
        prompt = f"""Answer ONLY using these extracted document snippets:

EXTRACTED CONTENT:
{extracted_content[:4000]}

QUESTION: {last_human_msg}

Rules:
- Use ONLY information from the extracted content above
- If not in the content, say "The documents don't contain this information."
- Be concise and accurate"""
    else:
        prompt = f"""User asked: "{last_human_msg}"
Searched for: "{search_query}"
No relevant content found in documents.

Respond briefly that you searched but found no relevant information."""

    response = llm.invoke([SystemMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}


# =============================================================================
# LLM JUDGE FUNCTION
# =============================================================================
def llm_judge(question: str, context: str, answer: str) -> dict:
    """LLM-based judge that evaluates answer faithfulness to context."""
    judge_prompt = f"""You are a strict factual judge. Your task is to evaluate if an ANSWER is supported by the given CONTEXT.

QUESTION: {question}

CONTEXT:
{context[:3000]}

ANSWER:
{answer}

INSTRUCTIONS:
1. Break the ANSWER into individual factual statements
2. For EACH statement, determine if it is:
   - SUPPORTED: The statement is directly stated or clearly implied in the CONTEXT
   - PARTIALLY_SUPPORTED: Some parts are in CONTEXT, but not complete
   - UNSUPPORTED: The statement is NOT found in the CONTEXT at all

RULES:
- Do NOT use any external knowledge - ONLY use the CONTEXT provided
- Do NOT paraphrase or infer beyond what is explicitly stated
- Be CONSERVATIVE: if unsure, mark as UNSUPPORTED
- If the answer says "documents don't contain information", mark as SUPPORTED (honest admission)

OUTPUT FORMAT (JSON only, no other text):
{{
  "statements": [
    {{"text": "<exact statement from answer>", "verdict": "SUPPORTED"}},
    {{"text": "<exact statement from answer>", "verdict": "UNSUPPORTED"}}
  ],
  "overall_verdict": "SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED"
}}

OVERALL_VERDICT logic:
- SUPPORTED: ALL statements are SUPPORTED
- PARTIALLY_SUPPORTED: Mix of SUPPORTED and UNSUPPORTED, or any PARTIALLY_SUPPORTED
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
                "parse_error": "Could not extract JSON from response"
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
        "context_preview": extracted_content[:500] + "..." if len(extracted_content) > 500 else extracted_content,
        "answer": answer,
        "raw_judgment": judgment
    }

    verdict = judgment.get("overall_verdict", "UNKNOWN")
    num_statements = len(judgment.get("statements", []))
    print(f"⚖️ [Judge] Verdict: {verdict} ({num_statements} statements analyzed)")

    if judgment.get("statements"):
        for stmt in judgment["statements"]:
            icon = "✅" if stmt["verdict"] == "SUPPORTED" else "⚠️" if stmt["verdict"] == "PARTIALLY_SUPPORTED" else "❌"
            print(f"   {icon} {stmt['verdict']}: {stmt['text'][:80]}...")

    return {
        "judgment": judgment,
        "judgment_log": judgment_log
    }


# =============================================================================
# NODE 8: SMART RETRY DECISION (key improvement over v3)
# =============================================================================
def should_retry(state: AgentState) -> bool:
    """SMART retry decision - does NOT retry on valid 'out of scope' responses.

    KEY DIFFERENCES FROM V3:
    - Recognizes "documents don't contain" as VALID when search returned results
    - Only retries on actual retrieval failure (0 results)
    - Stricter hallucination threshold (70% vs 50%)
    - Does NOT trigger on "no info" phrases if search worked
    """
    judgment = state.get("judgment", {})
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    search_results = state.get("search_results", [])
    extracted = state.get("extracted_content", "")

    # Don't retry if we've hit the limit
    if retry_count >= max_retries:
        print(f"🛑 [Smart-Retry] Max retries ({max_retries}) reached, ending")
        return False

    # Get the answer
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            answer = msg.content.lower()
            break

    # KEY FIX: If answer says "documents don't contain" and we DID search,
    # this is a VALID response - don't retry! The docs just don't have the info.
    if any(phrase in answer for phrase in VALID_NO_INFO_PHRASES):
        if search_results and len(search_results) > 0:
            print(f"✅ [Smart-Retry] Valid 'out of scope' response - no retry needed")
            return False

    # Trigger 1: Retrieval technically failed (no results at all)
    if not search_results or len(search_results) == 0:
        print(f"🔄 [Smart-Retry] Trigger: Search returned 0 results")
        return True

    # Trigger 2: Search found results but extraction failed
    if search_results and len(search_results) > 0 and (not extracted or len(extracted) < 50):
        print(f"🔄 [Smart-Retry] Trigger: Search found {len(search_results)} results but extraction failed")
        return True

    # Trigger 3: High hallucination rate (stricter threshold than v3)
    verdict = judgment.get("overall_verdict", "")
    if verdict == "UNSUPPORTED" and extracted and len(extracted) >= 50:
        statements = judgment.get("statements", [])
        unsupported = sum(1 for s in statements if s.get("verdict") == "UNSUPPORTED")
        total = len(statements)
        if total > 0:
            unsupported_ratio = unsupported / total
            if unsupported_ratio > HALLUCINATION_THRESHOLD:  # 70% threshold
                print(f"🔄 [Smart-Retry] Trigger: High hallucination ({unsupported}/{total} = {unsupported_ratio:.0%} UNSUPPORTED)")
                return True

    # Trigger 4: Explicit retrieval failure phrases
    for phrase in RETRIEVAL_FAILURE_PHRASES:
        if phrase in answer:
            print(f"🔄 [Smart-Retry] Trigger: Retrieval failure phrase '{phrase}'")
            return True

    return False


def self_correction_node(state: AgentState):
    """Prepare state for retry with appropriate strategy."""
    retry_count = state.get("retry_count", 0)
    judgment = state.get("judgment", {})
    extracted_content = state.get("extracted_content", "")

    if not extracted_content or len(extracted_content) < 50:
        strategy = "expand_k"
        reason = "No relevant documents found"
    elif judgment.get("overall_verdict") == "UNSUPPORTED":
        strategy = "rewrite_query"
        reason = "Answer not supported by context"
    else:
        strategy = "expand_k"
        reason = "Low faithfulness score"

    print(f"🔄 [Smart-Retry] Strategy: {strategy} | Reason: {reason}")

    return {
        "retry_count": retry_count + 1,
        "retry_strategy": strategy,
        "retry_reason": reason,
        "search_results": [],
        "retrieved_docs": [],
        "extracted_content": "",
        "judgment": {},
        "messages": [RemoveMessage(id=state["messages"][-1].id)] if state["messages"] and isinstance(state["messages"][-1], AIMessage) else []
    }


# =============================================================================
# ROUTING
# =============================================================================
def route_after_rewrite(state: AgentState):
    if state.get("question_is_clear", True):
        return "forced_search"
    return END


def route_after_judge(state: AgentState):
    """Route to retry or end based on smart retry check."""
    if should_retry(state):
        return "self_correct"
    return END


# =============================================================================
# BUILD GRAPH
# =============================================================================
graph_builder = StateGraph(AgentState)

graph_builder.add_node("summarize", summarization_node)
graph_builder.add_node("rewrite", query_rewriter_node)
graph_builder.add_node("forced_search", forced_search_node)
graph_builder.add_node("retrieve_parents", retrieve_parents_node)
graph_builder.add_node("extract_relevant", extract_relevant_node)
graph_builder.add_node("generate_answer", generate_answer_node)
graph_builder.add_node("judge_answer", judge_answer_node)
graph_builder.add_node("self_correct", self_correction_node)

graph_builder.add_edge(START, "summarize")
graph_builder.add_edge("summarize", "rewrite")
graph_builder.add_conditional_edges("rewrite", route_after_rewrite)
graph_builder.add_edge("forced_search", "retrieve_parents")
graph_builder.add_edge("retrieve_parents", "extract_relevant")
graph_builder.add_edge("extract_relevant", "generate_answer")
graph_builder.add_edge("generate_answer", "judge_answer")
graph_builder.add_conditional_edges("judge_answer", route_after_judge)
graph_builder.add_edge("self_correct", "forced_search")

checkpointer = MemorySaver()
agent_graph = graph_builder.compile(checkpointer=checkpointer)

__all__ = ['agent_graph', 'EXPERIMENT_TAG', 'llm_judge', 'get_cached_retrieval', 'cache_retrieval']
