"""
Shared graph nodes for RAG experiments.

All node functions are defined here to avoid code duplication.
"""
import re
import json
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from core.llm_manager import get_llm, get_llm_small
from core.shared.state import AgentState
from core.shared.tools import search_child_chunks, retrieve_parent_chunks
from core.shared.utils import (
    cache_retrieval,
    get_cached_retrieval,
    HALLUCINATION_THRESHOLD,
    VALID_NO_INFO_PHRASES
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _get_last_human_message(messages: list) -> str | None:
    """Extract the last human message from message list."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _get_last_ai_message(messages: list) -> str | None:
    """Extract the last AI message from message list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return None


def _get_retrieval_k(state: AgentState) -> int:
    """Calculate retrieval k based on retry count."""
    retry_count = state.get("retry_count", 0)
    if retry_count > 0:
        k = min(10, 5 + (retry_count * 3))
        print(f"[Retry #{retry_count}] Expanding k: {k}")
        return k
    return state.get("retrieval_k", 5)


def _extract_parent_ids(results: list) -> list:
    """Extract unique parent IDs from search results."""
    return list(set([
        r["parent_id"] for r in results
        if isinstance(r, dict) and "parent_id" in r
    ]))


# =============================================================================
# NODE 1: Summarization
# =============================================================================
def summarization_node(state: AgentState):
    """Summarize and prune old messages to manage context window."""
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

    response = get_llm_small().invoke([SystemMessage(content=prompt)])

    messages_to_keep = human_ai_msgs[-4:]
    messages_to_delete = [
        RemoveMessage(id=m.id)
        for m in messages
        if m not in messages_to_keep and not isinstance(m, SystemMessage)
    ]

    return {
        "conversation_summary": response.content,
        "messages": messages_to_delete
    }


# =============================================================================
# NODE 2: Query Rewriter
# =============================================================================
def query_rewriter_node(state: AgentState):
    """Rewrite questions with pronouns using conversation context."""
    messages = state.get("messages", [])
    if not messages:
        return {"question_is_clear": True}

    last_message = messages[-1].content
    summary = state.get("conversation_summary", "")

    # Check for pronouns that need resolution
    pronouns = ["it", "that", "this", "them", "those", "these"]
    has_pronoun = any(f" {p} " in f" {last_message.lower()} " for p in pronouns)

    if not has_pronoun or not summary:
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
# NODE 3a: Single Query Search
# =============================================================================
def single_query_search_node(state: AgentState):
    """Simple single-query search."""
    messages = state.get("messages", [])
    query = _get_last_human_message(messages)

    if not query:
        return {"search_results": [], "search_queries": []}

    retry_count = state.get("retry_count", 0)
    k = _get_retrieval_k(state)

    # Check cache first (only on first attempt)
    cached = get_cached_retrieval(query)
    if cached and retry_count == 0:
        results, _ = cached
    else:
        results = search_child_chunks.invoke({"query": query, "k": k}) or []
        if retry_count == 0 and results:
            cache_retrieval(query, results, _extract_parent_ids(results))

    print(f"[Search] {len(results)} results for: {query[:50]}...")

    return {
        "search_results": results,
        "search_queries": [query],
        "original_query": query,
        "retrieval_k": k
    }


# =============================================================================
# NODE 3b: Multi-Query Search
# =============================================================================
def multi_query_search_node(state: AgentState):
    """Generate multiple search queries for better coverage."""
    messages = state.get("messages", [])
    query = _get_last_human_message(messages)

    if not query:
        return {"search_results": [], "search_queries": []}

    retry_count = state.get("retry_count", 0)
    retry_strategy = state.get("retry_strategy", "")
    k = _get_retrieval_k(state)

    # Generate queries
    if retry_count > 0 and retry_strategy == "rewrite_query":
        prompt = f"""Generate 3 ALTERNATIVE search queries for: "{query}"
Use different keywords, synonyms, or phrasings.
Return ONLY the queries, one per line."""
    else:
        prompt = f"""Generate 2-3 search queries for this question:
"{query}"
If comparing two things, include a query for each.
Return ONLY the queries, one per line."""

    response = get_llm_small().invoke([SystemMessage(content=prompt)])

    # Clean and parse queries
    queries = []
    for q in response.content.strip().split('\n'):
        cleaned = re.sub(r'^[\d]+[.\)]\s*', '', q.strip())
        cleaned = re.sub(r'^[-*•]\s*', '', cleaned).strip()
        if cleaned and len(cleaned) > 2:
            queries.append(cleaned)

    # Always include original query
    if query not in queries:
        queries.insert(0, query)
    queries = queries[:3]

    print(f"[Multi-Query] {len(queries)} queries: {queries}")

    # Execute searches and merge
    all_results = []
    seen_ids = set()

    for q in queries:
        # Check cache
        cached = get_cached_retrieval(q) if retry_count == 0 else None
        if cached:
            results, _ = cached
        else:
            results = search_child_chunks.invoke({"query": q, "k": k}) or []
            if retry_count == 0 and results:
                cache_retrieval(q, results, _extract_parent_ids(results))

        # Deduplicate by parent_id
        for r in results:
            if isinstance(r, dict) and r.get("parent_id") and r["parent_id"] not in seen_ids:
                seen_ids.add(r["parent_id"])
                all_results.append(r)

    print(f"[Multi-Query] {len(all_results)} unique results")

    return {
        "search_results": all_results,
        "search_queries": queries,
        "original_query": state.get("original_query") or query,
        "retrieval_k": k
    }


# =============================================================================
# NODE 4: Retrieve Parent Chunks
# =============================================================================
def retrieve_parents_node(state: AgentState):
    """Retrieve full parent documents from search results."""
    search_results = state.get("search_results", [])

    if not search_results:
        return {"retrieved_docs": []}

    parent_ids = _extract_parent_ids(search_results)
    if not parent_ids:
        return {"retrieved_docs": []}

    print(f"[Retrieve] {len(parent_ids)} parents: {parent_ids}")
    docs = retrieve_parent_chunks.invoke({"parent_ids": parent_ids})

    return {"retrieved_docs": docs}


# =============================================================================
# NODE 5: Extract Relevant Content (Batched)
# =============================================================================
def extract_relevant_batch_node(state: AgentState):
    """Extract relevant content from all chunks in ONE LLM call."""
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])
    query = _get_last_human_message(messages)

    if not retrieved_docs or not query:
        return {"extracted_content": ""}

    # Combine all documents
    combined = ""
    for i, doc in enumerate(retrieved_docs):
        chunk_id = doc.get("parent_id", f"doc_{i}") if isinstance(doc, dict) else f"doc_{i}"
        content = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
        combined += f"\n=== DOC {i+1} ({chunk_id}) ===\n{content[:1500]}\n"

    prompt = f"""Extract sentences that answer this question:

QUESTION: {query}

DOCUMENTS:
{combined[:6000]}

Copy exact relevant sentences as bullet points.
If nothing relevant, return "- No relevant information found"

RELEVANT:"""

    try:
        response = get_llm_small().invoke([SystemMessage(content=prompt)])
        extracted = response.content.strip()

        # Fallback if extraction fails
        normalized = re.sub(r'[^a-z]', '', extracted.lower())
        if normalized in ["", "none", "norelevantinformationfound", "nothing"] or len(extracted) < 50:
            extracted = combined[:2000] + "..."
            print(f"[Extract] Fallback used")
        else:
            print(f"[Extract] {len(extracted)} chars extracted")

    except Exception as e:
        extracted = combined[:2000] + "..."
        print(f"[Extract] Error: {e}")

    return {"extracted_content": extracted}


# =============================================================================
# NODE 6: Generate Answer
# =============================================================================
def generate_answer_node(state: AgentState):
    """Generate answer using extracted content."""
    extracted = state.get("extracted_content", "")
    messages = state.get("messages", [])
    query = _get_last_human_message(messages)

    if not query:
        return {"messages": [AIMessage(content="I couldn't understand your question.")]}

    if extracted and len(extracted) > 50:
        prompt = f"""Answer using ONLY the excerpts below.

EXCERPTS:
{extracted[:4000]}

QUESTION: {query}

If excerpts are relevant, give a clear answer.
If not relevant, say "The documents don't contain this information."

ANSWER:"""
    else:
        prompt = f"""No relevant documents found for: "{query}"
Briefly explain that no relevant information was found."""

    response = get_llm().invoke([SystemMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}


def generate_answer_simple_node(state: AgentState):
    """Generate answer directly from docs (no extraction)."""
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])
    query = _get_last_human_message(messages)

    if not query:
        return {"messages": [AIMessage(content="I couldn't understand your question.")]}

    # Build context from raw docs
    context = ""
    for i, doc in enumerate(retrieved_docs[:5]):
        content = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
        context += f"\n--- Doc {i+1} ---\n{content[:1500]}\n"

    if context:
        prompt = f"""Answer using ONLY the documents below.

DOCUMENTS:
{context[:5000]}

QUESTION: {query}

ANSWER:"""
    else:
        prompt = f"""No documents found for: "{query}"
Briefly explain that no relevant documents were retrieved."""

    response = get_llm().invoke([SystemMessage(content=prompt)])
    return {
        "messages": [AIMessage(content=response.content)],
        "extracted_content": context  # For judge compatibility
    }


# =============================================================================
# NODE 7: Judge Answer
# =============================================================================
def llm_judge(question: str, context: str, answer: str) -> dict:
    """Evaluate if answer is supported by context."""
    prompt = f"""Evaluate if ANSWER is supported by CONTEXT.

QUESTION: {question}
CONTEXT: {context[:2500]}
ANSWER: {answer}

If answer says "documents don't contain info", mark SUPPORTED.

Return JSON:
{{"overall_verdict": "SUPPORTED|UNSUPPORTED", "statements": [{{"text": "...", "verdict": "SUPPORTED"}}]}}"""

    try:
        response = get_llm_small().invoke([SystemMessage(content=prompt)])
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            return json.loads(json_match.group())
        return {"overall_verdict": "SUPPORTED", "statements": []}
    except Exception as e:
        return {"overall_verdict": "ERROR", "error": str(e), "statements": []}


def judge_answer_node(state: AgentState):
    """Judge if the generated answer is faithful."""
    extracted = state.get("extracted_content", "")
    messages = state.get("messages", [])

    question = _get_last_human_message(messages)
    answer = _get_last_ai_message(messages)

    if not answer or not extracted:
        return {"judgment": {"overall_verdict": "NO_CONTEXT", "statements": []}}

    judgment = llm_judge(question, extracted, answer)
    print(f"[Judge] {judgment.get('overall_verdict', 'UNKNOWN')}")

    return {"judgment": judgment}


# =============================================================================
# NODE 8: Self-Correction
# =============================================================================
def should_retry(state: AgentState) -> bool:
    """Decide if we should retry (smart - skips valid 'no info' responses)."""
    judgment = state.get("judgment", {})
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    search_results = state.get("search_results", [])
    extracted = state.get("extracted_content", "")

    if retry_count >= max_retries:
        print(f"[Retry] Max retries reached")
        return False

    answer = (_get_last_ai_message(messages) or "").lower()

    # Don't retry valid "out of scope" responses
    if any(phrase in answer for phrase in VALID_NO_INFO_PHRASES) and search_results:
        print(f"[Retry] Valid 'out of scope' - skip")
        return False

    # Retry triggers
    if not search_results:
        print(f"[Retry] No search results")
        return True

    if not extracted or len(extracted) < 50:
        print(f"[Retry] Extraction failed")
        return True

    verdict = judgment.get("overall_verdict", "")
    if verdict == "UNSUPPORTED":
        statements = judgment.get("statements", [])
        unsupported = sum(1 for s in statements if s.get("verdict") == "UNSUPPORTED")
        if statements and (unsupported / len(statements)) > HALLUCINATION_THRESHOLD:
            print(f"[Retry] Hallucination detected")
            return True

    return False


def self_correction_node(state: AgentState):
    """Prepare state for retry."""
    retry_count = state.get("retry_count", 0)
    judgment = state.get("judgment", {})
    messages = state.get("messages", [])

    strategy = "rewrite_query" if judgment.get("overall_verdict") == "UNSUPPORTED" else "expand_k"
    print(f"[Retry] Strategy: {strategy}")

    # Remove last AI message
    messages_to_remove = []
    if messages and isinstance(messages[-1], AIMessage):
        messages_to_remove = [RemoveMessage(id=messages[-1].id)]

    return {
        "retry_count": retry_count + 1,
        "retry_strategy": strategy,
        "search_results": [],
        "retrieved_docs": [],
        "extracted_content": "",
        "judgment": {},
        "messages": messages_to_remove
    }
