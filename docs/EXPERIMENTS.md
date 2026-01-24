# Experiments Documentation

This document details all RAG experiments, their approaches, and key differences.

---

## Overview

| Experiment | Key Approach | Search Enforcement | Compression | Judge | Self-Correction |
|------------|--------------|-------------------|-------------|-------|-----------------|
| baseline | Full chunks via tool calls | None (LLM decides) | None | No | No |
| opt-v1-minimal-chunks | Minimal snippets + parent retrieval | None (LLM decides) | Snippet truncation | No | No |
| opt-v2-strict-search | Prompt-based mandatory search | System prompt | None | No | No |
| opt-v3-forced-search | Graph-enforced search + extraction + judge + retry | Graph structure | Query-focused extraction | Yes (qwen2.5:3b) | Yes (automatic) |
| opt-v4-smart-retry | Smart retry - doesn't over-trigger on valid responses | Graph structure | Query-focused extraction | Yes (qwen2.5:3b) | Yes (smart) |
| opt-v5-reliable-extraction | Multi-query search + stronger LLM extraction | Graph structure | Query-focused extraction (main LLM) | Yes (qwen2.5:3b) | Yes (smart) |
| opt-v6-fast-multiquery | Multi-query + batched extraction | Graph structure | Batched extraction (llm_small) | Yes (qwen2.5:3b) | Yes (smart) |

---

## Experiment 1: Baseline

**Directory:** `experiments/baseline/`

**Description:** Original implementation with full chunk retrieval. The LLM decides when to search.

### Flow
```
START → summarize → rewrite → retrieve_agent (LLM decides tools) → END
```

### How It Works
1. User asks a question
2. LLM can choose to call `search_and_retrieve` tool or answer from memory
3. Tool returns full document chunks
4. LLM generates answer

### Files
- `agent_state.py` - Basic state with conversation_summary, question_is_clear
- `graph.py` - ReAct-style agent with tool calling
- `tools.py` - `search_and_retrieve` returns full chunks

### Pros
- Simple implementation
- LLM has flexibility

### Cons
- LLM may skip search and hallucinate
- Full chunks waste tokens
- No control over retrieval behavior

---

## Experiment 2: Optimized V1 (Minimal Chunks)

**Directory:** `experiments/opt_v1_minimal_chunks/`

**Description:** Returns only parent IDs + 100-char snippets to save tokens. Uses two-step retrieval.

### Flow
```
START → summarize → rewrite → retrieve_agent (LLM decides tools) → END
```

### Key Changes from Baseline
1. **Two-step retrieval:**
   - `search_child_chunks` - Returns parent_ids + short snippets
   - `retrieve_parent_chunks` - Fetches full parent docs by ID

2. **Token optimization:**
   ```python
   # Returns minimal data
   {
       "parent_id": "doc_parent_5",
       "source": "javascript_tutorial.md",
       "snippet": "JavaScript is a dynamic..."  # 100 chars
   }
   ```

### Files
- `tools.py` - Split into `search_child_chunks` and `retrieve_parent_chunks`

### Pros
- Saves ~2,400 tokens per query
- Hierarchical retrieval (child → parent)

### Cons
- LLM still decides when to search
- 100-char snippets may be too short for context

---

## Experiment 3: Optimized V2 (Strict Search)

**Directory:** `experiments/opt_v2_strict_search/`

**Description:** Enforces mandatory search via system prompt. LLM is instructed to never use training knowledge.

### Flow
```
START → summarize → rewrite → retrieve_agent (prompted to ALWAYS search) → END
```

### Key Changes from V1
1. **Strict system prompt:**
   ```python
   system_prompt = """You are a document retrieval assistant.

   CRITICAL RULES:
   1. ALWAYS search the documents before answering
   2. NEVER use your training knowledge
   3. If no documents found, say "I couldn't find information about this"
   """
   ```

2. **No fallback to training data** - Refuses to answer without retrieval

### Files
- `graph.py` - Updated system prompt with strict rules

### Pros
- More consistent retrieval behavior
- Reduces hallucination from training data

### Cons
- LLM can still ignore the prompt
- No structural guarantee of search
- Prompt-based enforcement is fragile

---

## Experiment 4: Optimized V3 (Forced Search + Query-Focused Extraction + LLM Judge + Self-Correction)

**Directory:** `experiments/opt_v3_forced_search/`

**Description:** Graph-based enforcement - search is structurally required via graph edges. Adds query-focused extraction to compress retrieved docs. Includes LLM judge to evaluate answer faithfulness. **NEW:** Automatic self-correction with retry loop when faithfulness is low.

### Flow
```
START → summarize → rewrite → forced_search → retrieve_parents → extract_relevant
      → generate_answer → judge_answer → [self_correct → forced_search] or END
```

The flow now includes a **self-correction loop** that automatically retries when:
- Faithfulness is low (>50% statements unsupported)
- Answer contains "no information" phrases
- Retrieval returned insufficient content

### Key Changes from V2

#### 1. Graph-Enforced Search
Search is not a tool the LLM calls - it's a mandatory node in the graph:

```python
# V2: LLM decides whether to call tools
retrieve_agent → (LLM picks: search? or answer?)

# V3: Graph FORCES search
forced_search → retrieve_parents → extract → answer
```

The LLM cannot skip search because graph edges physically route through it.

#### 2. Increased Snippet Size
```python
# V1/V2: 100 chars (too short)
"snippet": doc.page_content[:100] + "..."

# V3: 250 chars (better context)
"snippet": doc.page_content[:250] + "..."
```

#### 3. Query-Focused Extraction (NEW)

**Problem:** Raw parent chunks are 2000+ chars, wasting tokens and diluting relevance.

**Solution:** Extract only sentences that answer the question.

```python
def extract_relevant_node(state: AgentState):
    """Extract relevant sentences from each chunk"""

    extraction_prompt = f"""Extract ONLY sentences that directly answer the question.

    QUESTION: {question}
    TEXT: {chunk_content}

    RULES:
    - Copy exact sentences - do NOT paraphrase
    - Only include directly relevant sentences
    - Return as bullet points
    - If nothing relevant, return "NONE"
    """

    # Fallback if extraction too short
    if len(extracted) < 50:
        extracted = original_text[:300] + "..."
```

#### 4. New State Fields
```python
class AgentState(MessagesState):
    # ... existing fields ...

    # Forced search flow
    search_results: List[Any] = []
    search_performed: bool = False
    search_query: str = ""
    retrieved_docs: List[Any] = []
    parent_ids: List[str] = []

    # Query-focused extraction
    extracted_content: str = ""      # Compressed content
    extraction_logs: List[Dict] = [] # Debug info
```

#### 5. Debug Logging
Each extraction is logged for debugging:
```python
extraction_logs.append({
    "chunk_id": "doc_parent_5",
    "original_length": 2150,
    "extracted_length": 342,
    "used_fallback": False,
    "extracted_preview": "JavaScript is a dynamic..."
})
```

Console output:
```
🔍 [Forced Search] Query: javascript variables
📄 [Retrieve] Parent IDs: ['doc_parent_5', 'doc_parent_8']
📝 [Extract] doc_parent_5: 2150 → 342 chars
📝 [Extract] doc_parent_8: 1890 → 156 chars (fallback)
📊 [Extract] Total: 498 chars from 2 chunks
⚖️ [Judge] Verdict: SUPPORTED (3 statements analyzed)
   ✅ SUPPORTED: JavaScript variables store data values...
   ✅ SUPPORTED: You can declare variables using var, let, or const...
   ✅ SUPPORTED: The let keyword was introduced in ES6...
```

#### 6. LLM Judge (Faithfulness Evaluation)

**Problem:** How do we know if the generated answer is actually supported by the retrieved context?

**Solution:** Add an LLM judge that evaluates each statement in the answer.

```python
def llm_judge(question: str, context: str, answer: str) -> dict:
    """
    LLM-based judge that evaluates answer faithfulness.
    Uses qwen2.5:3b (small, fast model).
    """
    # Returns JSON verdict
    {
        "statements": [
            {"text": "<statement>", "verdict": "SUPPORTED"},
            {"text": "<statement>", "verdict": "UNSUPPORTED"}
        ],
        "overall_verdict": "SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED"
    }
```

**Verdict Types:**
| Verdict | Meaning |
|---------|---------|
| SUPPORTED | Statement is directly found in context |
| PARTIALLY_SUPPORTED | Some parts are in context, but incomplete |
| UNSUPPORTED | Statement is NOT in context (hallucination) |

**Judge Rules:**
- Uses ONLY the provided context (no external knowledge)
- Does NOT paraphrase or infer
- Conservative: if unsure, marks UNSUPPORTED
- "Documents don't contain info" = SUPPORTED (honest admission)

**New State Fields:**
```python
judgment: Dict = {}       # JSON verdict from judge
judgment_log: Dict = {}   # Debug info (context, answer, raw response)
```

### Files
| File | Purpose |
|------|---------|
| `agent_state.py` | Extended state with search + extraction + judgment fields |
| `tools.py` | Header-aware retrieval with boosting + 250-char snippets |
| `graph.py` | Linear flow with extract_relevant_node and judge_answer_node |

#### 7. Cross-Encoder Reranking + Header Boosting

**Problem:** Raw hybrid search scores don't capture fine-grained semantic relevance.

**Solution:** Two-stage reranking pipeline:

```
Hybrid Search → Cross-Encoder Rerank → Header Boost → Top K
```

**Stage 1: Cross-Encoder Reranking (Primary)**

Uses `ms-marco-MiniLM-L-6-v2` (~22M params) for accurate semantic scoring:

```python
# Cross-encoders jointly encode query+doc pairs
# More accurate than bi-encoders, but slower
pairs = [(query, doc.page_content[:512]) for doc in candidates]
scores = reranker.predict(pairs)  # Returns relevance scores
```

**Stage 2: Header Boost (Secondary)**

Lightweight keyword matching on H1/H2/H3 headers:

```python
# Jaccard overlap between query keywords and header keywords
header_score = len(query_keywords & header_keywords) / len(union)
final_score = cross_encoder_score + (header_score * 0.1)
```

**Console output:**
```
🔍 [Search] Retrieved 10 candidates
🎯 [Rerank] Cross-encoder scoring 10 candidates...
   🎯 [CE] #1 score=0.847: JavaScript is a dynamic programming language...
   🎯 [CE] #2 score=0.723: Variables in JavaScript store data values...
   🎯 [CE] #3 score=0.651: The let keyword was introduced in ES6...
   📈 [Header Boost] #2 ce=0.723 + header=0.35×0.1 → 0.758
```

**Why cross-encoder > bi-encoder for reranking:**
| Approach | How it works | Accuracy | Speed |
|----------|--------------|----------|-------|
| Bi-encoder | Separate embeddings, cosine similarity | Good | Fast |
| Cross-encoder | Joint encoding of query+doc pair | Better | Slower |

Cross-encoders see both query and document together, enabling better semantic matching.

**Configuration in `config.py`:**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Lightweight, fast
    max_length=512
)
# Alternative: "BAAI/bge-reranker-base" for better quality
```

#### 8. Self-Correction (Automatic Retry)

**Problem:** The system is named "self-correcting" but correction was manual.

**Solution:** Automatic retry triggers based on judgment results.

**Retry Triggers:**
| Trigger | Condition | Strategy |
|---------|-----------|----------|
| No information | Answer contains "couldn't find", "no relevant", etc. | `expand_k` |
| Low faithfulness | >50% statements UNSUPPORTED | `rewrite_query` |
| Empty retrieval | `extracted_content` < 50 chars | `expand_k` |

**Retry Strategies:**
```python
# Strategy 1: Expand k (retrieve more documents)
if strategy == "expand_k":
    current_k = min(10, 5 + (retry_count * 3))  # 5 → 8 → 10

# Strategy 2: Rewrite query (use different keywords)
if strategy == "rewrite_query":
    prompt = f'Generate ALTERNATIVE search keywords for: "{original_query}"'
```

**Self-Correction Node:**
```python
def self_correction_node(state: AgentState):
    """Prepare state for retry with appropriate strategy."""
    # Choose strategy based on failure mode
    if not extracted_content:
        strategy = "expand_k"
    elif judgment["overall_verdict"] == "UNSUPPORTED":
        strategy = "rewrite_query"

    return {
        "retry_count": retry_count + 1,
        "retry_strategy": strategy,
        "search_results": [],  # Clear to force fresh search
        "messages": [RemoveMessage(id=last_ai_msg.id)]  # Remove bad answer
    }
```

**Console output:**
```
🔍 [Forced Search] Query: javascript variables (k=5)
⚖️ [Judge] Verdict: UNSUPPORTED (2 statements analyzed)
   ❌ UNSUPPORTED: JavaScript uses dynamic typing...
   ❌ UNSUPPORTED: Variables are hoisted to the top...
🔄 [Self-Correct] Trigger: 2/2 statements UNSUPPORTED (100%)
🔄 [Self-Correct] Strategy: rewrite_query | Reason: Answer not supported by context
🔄 [Retry #1] Rewritten query: js variable declaration types
🔍 [Forced Search] Query: js variable declaration types (k=5)
⚖️ [Judge] Verdict: SUPPORTED (2 statements analyzed)
   ✅ SUPPORTED: JavaScript variables can be declared using var, let, or const...
```

**New State Fields:**
```python
# Self-correction fields
retry_count: int = 0          # Number of retry attempts
max_retries: int = 2          # Maximum retry attempts allowed
retry_reason: str = ""        # Why retry was triggered
retry_strategy: str = ""      # Which strategy was used
original_query: str = ""      # Store original query for rewrites
retrieval_k: int = 5          # Current k value for retrieval
```

**Max Retries:** Limited to 2 retries to prevent infinite loops.

---

#### 9. Retrieval Caching

**Problem:** Follow-up queries often search for similar content, wasting compute.

**Solution:** Cache `query → (search_results, parent_ids)` mapping.

```python
# In-memory cache (query_hash → results)
_retrieval_cache: Dict[str, Tuple[list, list]] = {}
CACHE_MAX_SIZE = 100

def cache_retrieval(query: str, search_results: list, parent_ids: list):
    """Cache retrieval results for a query"""
    key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    _retrieval_cache[key] = (search_results, parent_ids)
    print(f"💾 [Cache] Stored: {query[:30]}... → {len(parent_ids)} parents")

def get_cached_retrieval(query: str) -> Tuple[list, list] | None:
    """Get cached retrieval results if available"""
    key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    if key in _retrieval_cache:
        print(f"⚡ [Cache HIT] {query[:30]}... → {len(parent_ids)} parents")
        return _retrieval_cache[key]
    return None
```

**Cache Behavior:**
- Normalized query (lowercase, stripped) used as key
- Skipped on retry attempts (to get fresh results)
- Simple LRU-style eviction when full (clear oldest half)
- Session-scoped (cleared on restart)

**Console output:**
```
🔍 [Forced Search] Query: javascript variables
💾 [Cache] Stored: javascript variables... → 3 parents
...
🔍 [Forced Search] Query: javascript variables
⚡ [Cache HIT] javascript variables... → 3 parents
```

**Benefits:**
- Instant retrieval for repeated/follow-up queries
- Reduces Qdrant queries and embedding computations
- Especially effective in conversational RAG

---

### Why This Works
1. **Structural enforcement:** Graph edges guarantee search runs
2. **Compression:** 2000-char chunks → 200-500 char extractions
3. **Preserved accuracy:** Original wording, no paraphrasing
4. **Graceful fallback:** Never loses context entirely
5. **Debuggable:** Full logging of extraction behavior
6. **Faithfulness check:** LLM judge catches hallucinations
7. **Cross-encoder reranking:** Accurate semantic re-scoring
8. **Header boosting:** Query-header alignment as secondary signal
9. **Self-correction:** Automatic retry on low faithfulness
10. **Caching:** Fast retrieval for follow-up queries

---

## Experiment 5: Optimized V4 (Smart Retry)

**Directory:** `experiments/opt_v4_smart_retry/`

**Description:** Improved self-correction logic that doesn't over-trigger on valid "out of scope" responses. Fixes the answer_quality drop issue from v3.

### Problem with V3

V3's self-correction was too aggressive:
- Triggered retry on every "documents don't contain" response
- This was wasteful for genuinely out-of-scope questions (e.g., Python questions when only JS/Blockchain docs exist)
- Caused answer_quality to drop from 94% to 81%
- Wasted compute cycles on retries that couldn't succeed

### Solution in V4

**Smart Retry Logic:**

```python
# KEY FIX: If answer says "documents don't contain" and we DID search,
# this is a VALID response - don't retry!
VALID_NO_INFO_PHRASES = [
    "documents don't contain",
    "not covered in the documents",
    "no relevant information found",
    "outside the scope"
]

if any(phrase in answer for phrase in VALID_NO_INFO_PHRASES):
    if search_results and len(search_results) > 0:
        # We searched, found docs, but they don't answer the question
        # This is VALID - DON'T retry
        return False
```

### Key Differences from V3

| Aspect | V3 | V4 |
|--------|----|----|
| "No info" handling | Triggers retry | Recognizes as valid response |
| Hallucination threshold | 50% | 70% (stricter) |
| Retry on out-of-scope | Yes (wasteful) | No (smart) |
| answer_quality impact | Drops to ~81% | Should stay ~94% |

### Retry Triggers (V4)

Only retries when:
1. **Retrieval truly failed** - Search returned 0 results
2. **Extraction failed** - Search found results but extraction was empty
3. **High hallucination** - >70% statements UNSUPPORTED (stricter than v3's 50%)
4. **Explicit retrieval errors** - Phrases like "couldn't find any documents"

Does NOT retry when:
- Answer correctly says "documents don't contain" AND search returned results
- This is a valid "out of scope" response, not a failure

### Console Output Example

```
🔍 [Forced Search] Query: python machine learning (k=5)
📄 [Retrieve] Parent IDs: [javascript_tutorial_parent_4, ...]
📝 [Extract] javascript_tutorial_parent_4: 1500 → 300 chars (fallback)
⚖️ [Judge] Verdict: SUPPORTED (1 statement analyzed)
   ✅ SUPPORTED: The documents don't contain this information...
✅ [Smart-Retry] Valid 'out of scope' response - no retry needed
```

### Files

- `agent_state.py` - Same as v3
- `tools.py` - Same as v3 (cross-encoder + header boost)
- `graph.py` - Smart retry logic in `should_retry()` function

---

## Experiment 6: Optimized V5 (Reliable Extraction)

**Directory:** `experiments/opt_v5_reliable_extraction/`

**Description:** Fixes the issue where answers said "documents don't contain" even when relevant documents WERE retrieved. Uses multi-query search for better retrieval coverage and the main LLM for more reliable extraction.

### Problems with V4

1. **Extraction using `llm_small` (qwen2.5:3b) was too weak** - Often returned "NONE" and fell back to raw text
2. **Single keyword search missed conceptual questions** - "What's the difference between X and Y" needs searches for both X and Y
3. **"documents don't contain" even when docs had answers** - Extraction failed to find relevant sentences

### Solution in V5

#### 1. Multi-Query Search

Generates 2-3 search queries per question for better retrieval coverage:

```python
def multi_query_search_node(state: AgentState):
    """Generate multiple search queries for better retrieval coverage."""

    multi_query_prompt = f"""Given this question, generate 2-3 different search queries.

    QUESTION: {last_human_msg}

    Generate queries that:
    1. Extract key terms/concepts (keywords style)
    2. Rephrase as a factual lookup (what is X, how does X work)
    3. If comparing things, search for each thing separately

    Return ONLY the queries, one per line."""

    # Execute all searches and merge results (dedupe by parent_id)
```

**Example:**

Question: "What's the difference between blockchain and traditional databases?"

Generated queries:
1. "blockchain traditional databases difference"
2. "What is blockchain"
3. "blockchain vs database"

**Console output:**
```
[Multi-Query] Generated 3 queries:
   1. blockchain traditional databases difference
   2. What is blockchain
   3. blockchain vs database
[Multi-Query] Merged 8 unique results from 3 queries
```

#### 2. Stronger LLM for Extraction

Uses the **main LLM** instead of `llm_small` for extraction:

```python
# V4: Used llm_small (qwen2.5:3b) - often failed
response = llm_small.invoke([SystemMessage(content=extraction_prompt)])

# V5: Uses main LLM (qwen2.5:7b or higher) - more reliable
response = llm.invoke([SystemMessage(content=extraction_prompt)])
```

#### 3. Improved Extraction Prompt

More specific instructions for better extraction:

```python
extraction_prompt = f"""Your task is to extract sentences from the TEXT that help answer the QUESTION.

INSTRUCTIONS:
1. Find ALL sentences that contain information related to the question
2. Include definitions, explanations, examples, or facts that answer the question
3. Copy the exact sentences - do NOT paraphrase or summarize
4. Format as bullet points starting with "-"
5. If the text discusses the topic but from a different angle, INCLUDE it
6. If there is truly nothing relevant, return "- NONE"
"""
```

#### 4. Improved Answer Generation

Better prompt that uses extracted content properly:

```python
prompt = f"""You are a helpful assistant that answers questions using ONLY the provided document excerpts.

INSTRUCTIONS:
1. Answer the question using ONLY information from the document excerpts above
2. If the excerpts contain relevant information, provide a clear and complete answer
3. If the excerpts discuss the topic but don't fully answer the question, share what IS available
4. Only say "The documents don't contain this information" if the excerpts are truly unrelated
5. Be concise but thorough
"""
```

### Key Differences from V4

| Aspect | V4 | V5 |
|--------|----|----|
| Search queries | Single keyword extraction | 2-3 multi-query generation |
| Extraction LLM | `llm_small` (weak) | `llm` (main, stronger) |
| Extraction prompt | Basic | More specific instructions |
| Answer generation | Basic | Improved prompt with better fallback |
| State field | `search_query` | `search_queries` (list) |

### Flow

```
START -> summarize -> rewrite -> multi_query_search -> retrieve_parents
      -> extract_relevant -> generate_answer -> judge_answer
      -> [self_correct -> multi_query_search] or END
```

### Console Output Example

```
[Multi-Query] Generated 3 queries:
   1. consensus mechanisms blockchain
   2. What are consensus mechanisms
   3. blockchain consensus types
[Search] Retrieved 10 candidates
[CE] #1 score=0.892: Consensus mechanisms include Proof of Work...
[Multi-Query] Merged 7 unique results from 3 queries
[Retrieve] Parent IDs: ['bc_parent_5', 'bc_parent_7', 'bc_parent_8']
[Extract] bc_parent_5: 2100 -> 450 chars
[Extract] bc_parent_7: 1800 -> 380 chars
[Extract] bc_parent_8: 1600 -> 290 chars
[Extract] Total: 1120 chars from 3 chunks
[Judge] Verdict: SUPPORTED (3 statements)
```

### Files

- `agent_state.py` - Adds `search_queries: List[str]` field
- `tools.py` - Same as v4 (cross-encoder + header boost)
- `graph.py` - Multi-query search node + stronger extraction

---

## Experiment 7: Optimized V6 (Fast Multi-Query)

**Directory:** `experiments/opt_v6_fast_multiquery/`

**Description:** Keeps v5's multi-query search (which solves comparison questions) but drastically reduces latency by using batched extraction with llm_small instead of one-call-per-chunk with the main LLM.

### Problem with V5

V5 had great accuracy but terrible latency (~98 seconds vs v4's ~34 seconds):

| Step | V5 (slow) | Calls |
|------|-----------|-------|
| Multi-query generation | main LLM | 1 |
| Extraction per chunk | **main LLM × 5-7 chunks** | **5-7** |
| Answer generation | main LLM | 1 |
| Judge | llm_small | 1 |
| **Total LLM calls** | | **8-10** |

The bottleneck: **5-7 separate main LLM calls for extraction**.

### Solution in V6

**Batched extraction** - Extract from ALL chunks in ONE call:

| Step | V6 (fast) | Calls |
|------|-----------|-------|
| Multi-query generation | llm_small | 1 |
| **Batched extraction** | **llm_small × 1** | **1** |
| Answer generation | main LLM | 1 |
| Judge | llm_small | 1 |
| **Total LLM calls** | | **4** |

### Key Changes from V5

```python
# V5: One call PER chunk (slow)
for doc in retrieved_docs:
    response = llm.invoke([extraction_prompt_for_one_doc])  # 5-7 calls!

# V6: ONE call for ALL chunks (fast)
combined_docs = "\n".join([doc.content for doc in retrieved_docs])
response = llm_small.invoke([extraction_prompt_for_all_docs])  # 1 call!
```

### Batched Extraction Prompt

```python
extraction_prompt = f"""Extract ALL sentences from the documents below that help answer this question:

QUESTION: {question}

DOCUMENTS:
=== DOCUMENT 1 (chunk_id_1) ===
{content_1}

=== DOCUMENT 2 (chunk_id_2) ===
{content_2}

...

INSTRUCTIONS:
1. Find sentences that contain information about the question topic
2. Include definitions, explanations, facts, or examples
3. Copy exact sentences - do NOT paraphrase
4. If truly nothing relevant exists, return "- No relevant information found"

RELEVANT SENTENCES:"""
```

### Expected Latency Comparison

| Experiment | Latency | Accuracy | Notes |
|------------|---------|----------|-------|
| opt-v4 | ~34s | Low on comparisons | Single keyword search |
| opt-v5 | ~98s | High | Multi-query + per-chunk extraction |
| **opt-v6** | **~35-40s** | **High** | Multi-query + batched extraction |

### Flow

```
START -> summarize -> rewrite -> multi_query_search -> retrieve_parents
      -> extract_relevant_batch -> generate_answer -> judge_answer
      -> [self_correct -> multi_query_search] or END
```

### Console Output Example

```
[Multi-Query] Generated 3 queries:
   1. What is DeFi
   2. DeFi traditional finance difference
   3. decentralized finance
[Multi-Query] Merged 6 unique results from 3 queries
[Retrieve] Parent IDs: ['bc_parent_5', 'bc_parent_7', ...]
[Extract-Batch] 6 docs -> 850 chars
[Judge] Verdict: SUPPORTED
```

### Files

- `agent_state.py` - Same as v5
- `tools.py` - Same as v5
- `graph.py` - Batched extraction node

---

## Experiment 8: Optimized V7 (Temperature Tuning)

**Directory:** `experiments/opt_v7_temperature_tuning/`

**Description:** Builds on V6's architecture (Fast Multi-Query + Batched Extraction) but introduces parametrized temperature control to find the optimal balance between creativity and faithfulness.

### Hypothesis

- **Lower Temperature (0.1)**: Higher faithfulness, less hallucination, but potentially dry/robotic answers.
- **Higher Temperature (0.7)**: More natural/fluent answers, but higher risk of hallucinations.
- **Goal**: Find the "sweet spot" where answer quality is high without sacrificing faithfulness.

### Key Changes from V6

1. **Parametrized Graph**
   The graph now accepts configuration at runtime:
   ```python
   def create_graph_with_config(config_key: str):
       config = TEMPERATURE_CONFIGS[config_key]
       set_temperatures(config["llm_temp"], config["llm_small_temp"])
       # ... builds graph ...
   ```

2. **LangSmith Integration**
   Uses `evaluate()` to systematically test 5 configurations against the same dataset.

3. **Serial Execution**
   Forced `max_concurrency=1` to prevent local Ollama instances from choking RAM/VRAM during parallel evaluation.

### Configurations Tested

| Config | Name | Main LLM Temp | Small LLM Temp | Hypothesis |
|--------|------|---------------|----------------|------------|
| **A** | Original (Strict) | 0.1 | 0.0 | High faithfulness, low creativity |
| **B** | Recommended | 0.3 | 0.1 | Balanced (likely winner) |
| **C** | Creative | 0.5 | 0.2 | Better flow, potential hallucinations |
| **D** | Balanced Both | 0.3 | 0.3 | Relaxed judge (higher small temp) |
| **E** | Creative Main | 0.7 | 0.1 | Very fluent main, strict judge |

### Files

- `run_temperature_comparison.py` - Driver script that runs `evaluate()` for all configs
- `graph.py` - Adapted from V6 to support dynamic temperature setting
- `setup_langsmith_dataset.py` - Creates the `rag-evaluation-tests-to-check-temperatures` dataset

---

## Running Experiments

### Setup (First Time)
```bash
# Create LangSmith dataset
python experiments/setup_langsmith_dataset.py
```

### Run Individual Experiment
```bash
python experiments/run_langsmith_experiment.py baseline
python experiments/run_langsmith_experiment.py opt-v1-minimal-chunks
python experiments/run_langsmith_experiment.py opt-v2-strict-search
python experiments/run_langsmith_experiment.py opt-v3-forced-search
```

### List All Experiments
```bash
python experiments/experiment_registry.py
```

### Compare Results
Results are saved to:
- LangSmith UI: https://smith.langchain.com
- Local: `experiments/results/{experiment_id}_{timestamp}.json`

---

## Evaluation Metrics

| Metric | What It Measures | How |
|--------|-----------------|-----|
| `faithfulness` | Is answer grounded in docs? | LLM judges if answer only uses retrieved info |
| `retrieval_relevance` | Are retrieved docs relevant? | LLM rates doc relevance to question |
| `semantic_similarity` | Does answer match reference? | Cosine similarity of embeddings |
| `answer_quality` | Is answer helpful? | LLM rates clarity, completeness, accuracy |
| `latency` | Response time | Timestamp difference |

---

## Evolution Summary

```
baseline
    ↓ Problem: Full chunks waste tokens
opt-v1-minimal-chunks
    ↓ Problem: LLM can skip search
opt-v2-strict-search
    ↓ Problem: Prompt enforcement is fragile
opt-v3-forced-search
    ✓ Graph enforces search
    ✓ Extraction compresses context
    ✓ LLM judge evaluates faithfulness
    ✓ Cross-encoder reranking improves relevance
    ✓ Self-correction retries on failure
    ✓ Caching speeds up follow-ups
    ✓ Logging enables debugging
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        opt-v3-forced-search                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌─────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │ START    │───▶│summarize│───▶│   rewrite    │───▶│ forced_search  │ │
│  └──────────┘    └─────────┘    └──────────────┘    └───────┬────────┘ │
│                                                              │          │
│                                        ┌─────────────────────┘          │
│                                        ▼                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    RETRIEVAL PIPELINE                             │  │
│  │  ┌─────────────┐   ┌──────────────┐   ┌───────────────────────┐  │  │
│  │  │Hybrid Search│──▶│Cross-Encoder │──▶│   Header Boost        │  │  │
│  │  │(dense+sparse│   │  Reranking   │   │ (Jaccard overlap)     │  │  │
│  │  └─────────────┘   └──────────────┘   └───────────────────────┘  │  │
│  │         │                                        │                │  │
│  │         └──────────── CACHE ◀────────────────────┘                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                        │                                │
│                                        ▼                                │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │retrieve_parents │───▶│ extract_relevant │───▶│ generate_answer  │  │
│  │ (full chunks)   │    │ (qwen2.5:3b)     │    │  (qwen2.5:7b)    │  │
│  └─────────────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                          │             │
│                                                          ▼             │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │                      JUDGE + SELF-CORRECT                      │    │
│  │  ┌──────────────┐         ┌──────────────────────────────┐    │    │
│  │  │ judge_answer │────────▶│      should_retry?           │    │    │
│  │  │ (qwen2.5:3b) │         │  • Low faithfulness (>50%)   │    │    │
│  │  │              │         │  • "No info" phrases         │    │    │
│  │  │ SUPPORTED?   │         │  • Empty extraction          │    │    │
│  │  └──────────────┘         └──────────────┬───────────────┘    │    │
│  │                                          │                     │    │
│  │                           ┌──────────────┴──────────────┐     │    │
│  │                           │                             │     │    │
│  │                           ▼                             ▼     │    │
│  │                    ┌─────────────┐              ┌───────────┐ │    │
│  │                    │self_correct │              │    END    │ │    │
│  │                    │• expand_k   │──────────────│           │ │    │
│  │                    │• rewrite    │  (max 2x)    │           │ │    │
│  │                    └──────┬──────┘              └───────────┘ │    │
│  │                           │                                   │    │
│  │                           └─────────────▶ forced_search ◀─────┘    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps (Future Experiments)

Potential improvements to explore:

1. ~~**Reranking** - Use cross-encoder to rerank retrieved chunks~~ ✅ Done
2. ~~**Iterative retrieval** - If first retrieval fails, try different query~~ ✅ Done (self-correction)
3. ~~**Caching** - Cache frequent queries/extractions~~ ✅ Done
4. **Relevance filtering** - Skip chunks with low cross-encoder scores
5. **Multi-hop** - Follow references in retrieved docs
6. **Semantic caching** - Cache similar (not just identical) queries
7. **Parallel extraction** - Extract from multiple chunks concurrently
8. **Streaming** - Stream answer generation for better UX
