# Bug Fixes & Solutions Log

This document tracks all bugs encountered during development, their root causes, and solutions.

---

## Bug #1: ModuleNotFoundError for 'experiments'

**Date:** 2025-01-18
**File:** `experiments/compare_experiments.py`
**Error:**
```
ModuleNotFoundError: No module named 'experiments'
```

**Root Cause:**
Using `sys.path.append('..')` which doesn't reliably resolve to the project root when running scripts from different directories.

**Solution:**
Changed to absolute path resolution:
```python
# BEFORE (broken)
sys.path.append('..')

# AFTER (fixed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**Why it works:**
`os.path.abspath(__file__)` gives the absolute path of the current script, then we go up two directories to reach the project root and insert it at the beginning of `sys.path`.

---

## Bug #2: LangSmith 401 Unauthorized - Invalid Token

**Date:** 2025-01-18
**File:** `.env`
**Error:**
```
langsmith.utils.LangSmithAuthError: Authentication failed for /datasets.
HTTPError('401 Client Error: Unauthorized', '{"detail":"Invalid token"}')
```

**Root Cause:**
The `.env` file had incorrect formatting:
```
LANGCHAIN_API_KEY = "lsv2_pt_xxxxx"
```
- Spaces around the `=` sign
- Value wrapped in quotes

This caused the environment variable to not be parsed correctly.

**Solution:**
Fixed the `.env` formatting:
```
# BEFORE (broken)
LANGCHAIN_API_KEY = "lsv2_pt_xxxxx"

# AFTER (fixed)
LANGCHAIN_API_KEY=lsv2_pt_xxxxx
```

**Additional Fix:**
Also added `python-dotenv` to load the `.env` file in scripts:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Bug #3: Faithfulness Score Always 0.00

**Date:** 2025-01-18
**File:** `experiments/run_langsmith_experiment.py`
**Symptom:**
All experiments showed `faithfulness: 0.00` even when the agent retrieved documents and gave correct answers.

**Root Cause #1: Thread ID Mismatch**
The `create_target_function` was using different thread IDs for `invoke()` and `get_state()`:

```python
# BEFORE (broken)
result = graph.invoke(
    {"messages": [HumanMessage(content=question)]},
    {"configurable": {"thread_id": f"eval-{datetime.now().timestamp()}"}}  # Thread ID #1
)

# Different timestamp = different thread ID!
config = {"configurable": {"thread_id": f"eval-{datetime.now().timestamp()}"}}  # Thread ID #2
state = graph.get_state(config)  # Returns EMPTY state - wrong thread!
messages = state.values.get("messages", [])  # Empty!
```

Because `datetime.now().timestamp()` was called twice (milliseconds apart), two different thread IDs were generated. The `get_state()` call looked at an empty thread.

**Solution #1:**
Use consistent thread ID and get messages directly from result:

```python
# AFTER (fixed)
thread_id = f"eval-{datetime.now().timestamp()}"
config = {"configurable": {"thread_id": thread_id}}  # Single thread ID

result = graph.invoke(
    {"messages": [HumanMessage(content=question)]},
    config
)

# Get messages directly from invoke result (no need for get_state)
messages = result.get("messages", [])
```

**Root Cause #2: Document Truncation Too Aggressive**
The `extract_retrieved_docs` function truncated content to only 500 characters:

```python
# BEFORE (broken)
content = str(msg.content)[:500] + "..."
```

500 characters wasn't enough context for the LLM evaluator to properly judge faithfulness.

**Solution #2:**
Increased context and added filtering:

```python
# AFTER (fixed)
def extract_retrieved_docs(messages):
    docs = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            content = str(msg.content)
            # Only include substantial content (likely from retrieve_parent_chunks)
            if len(content) > 100:
                docs.append(content[:1500])  # More context for faithfulness check
    return "\n---\n".join(docs) if docs else "No documents retrieved"
```

---

## Bug #4: Local JSON Results All Null Values

**Date:** 2025-01-18
**File:** `experiments/run_langsmith_experiment.py`
**Symptom:**
The local JSON results file had all null values:
```json
{
  "inputs": {},
  "outputs": {},
  "reference": {},
  "evaluations": {
    "faithfulness": null,
    "retrieval_f1": null,
    ...
  }
}
```

**Root Cause:**
The LangSmith `evaluate()` function returns an `ExperimentResults` object, not a list of dictionaries. We were treating it like a dict and calling `.get()` on object attributes:

```python
# BEFORE (broken) - treating object as dict
for result in results:
    local_results.append({
        "inputs": result.get("input", {}),  # Wrong! result is an object, not dict
        "outputs": result.get("output", {}),
        "evaluations": {
            "faithfulness": result.get("feedback", {}).get("faithfulness", {}).get("score"),
            ...
        }
    })
```

The result object has attributes like `result.run`, `result.example`, `result.evaluation_results` - not dictionary keys.

**Solution:**
Access the object attributes properly using `hasattr()` and direct attribute access:

```python
# AFTER (fixed) - properly accessing object attributes
for result in results:
    run_input = {}
    run_output = {}
    reference = {}
    evaluations = {}

    # Extract from the result object attributes
    if hasattr(result, 'run') and result.run:
        run_input = result.run.inputs or {}
        run_output = result.run.outputs or {}

    if hasattr(result, 'example') and result.example:
        reference = result.example.outputs or {}

    # Extract evaluation scores from evaluation_results list
    if hasattr(result, 'evaluation_results') and result.evaluation_results:
        for eval_result in result.evaluation_results:
            if hasattr(eval_result, 'key') and hasattr(eval_result, 'score'):
                key = eval_result.key
                score = eval_result.score
                evaluations[key] = score

    local_results.append({
        "inputs": run_input,
        "outputs": run_output,
        "reference": reference,
        "evaluations": evaluations
    })
```

**Why it works:**
LangSmith's `evaluate()` returns `ExperimentResults` which yields `EvaluationResult` objects when iterated. Each `EvaluationResult` has:
- `.run` - the Run object with `.inputs` and `.outputs`
- `.example` - the Example object with `.outputs` (reference data)
- `.evaluation_results` - list of evaluator results with `.key` and `.score`

---

## Bug #5: Hardcoded Expected Chunks - Unreliable Retrieval Evaluation

**Date:** 2025-01-18
**File:** `experiments/setup_langsmith_dataset.py`, `experiments/run_langsmith_experiment.py`
**Symptom:**
The `retrieval_f1` metric was always 0 because the hardcoded `expected_chunks` in the dataset didn't match actual retrieved chunks.

**Root Cause:**
The evaluation system assumed specific chunk IDs were "correct" without verifying:
```python
# In setup_langsmith_dataset.py
{
    "inputs": {"question": "What is JavaScript?"},
    "outputs": {
        "expected_chunks": ["javascript_tutorial_parent_6"],  # Hardcoded assumption!
        ...
    }
}

# In run_langsmith_experiment.py - comparing against assumed chunks
def retrieval_f1_evaluator(run: Run, example: Example) -> dict:
    retrieved = run.outputs.get("retrieved_chunks", [])
    expected = example.outputs.get("expected_chunks", [])  # Uses hardcoded values
    # ... F1 calculation comparing to assumptions
```

Problems:
1. No verification that assumed chunks actually contain correct info
2. Brittle - chunk IDs change if documents are re-indexed
3. Incomplete - questions may have multiple valid chunks
4. Metric measures match to assumptions, not actual relevance

**Solution:**
Replaced `retrieval_f1_evaluator` with `retrieval_relevance_evaluator` that uses LLM to judge relevance:

```python
def retrieval_relevance_evaluator(run: Run, example: Example) -> dict:
    """LLM judges if retrieved documents are relevant to the question"""
    question = run.inputs.get("question", "")
    retrieved_docs = run.outputs.get("retrieved_docs", "No documents")

    if retrieved_docs == "No documents retrieved" or not retrieved_docs:
        return {"key": "retrieval_relevance", "score": 0.0}

    prompt = f"""You are evaluating if retrieved documents are RELEVANT to answer a question.

Question: {question}

Retrieved Documents:
{retrieved_docs[:3000]}

Rate RETRIEVAL RELEVANCE (0.0 to 1.0):
0.0 = Documents are completely irrelevant to the question
0.3 = Documents are tangentially related but don't contain the answer
0.5 = Documents are somewhat relevant but missing key information
0.7 = Documents are relevant and contain most of the needed information
1.0 = Documents are highly relevant and contain all information needed to answer

Respond ONLY with a number between 0 and 1."""

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except:
        score = 0.0

    return {"key": "retrieval_relevance", "score": score}
```

**Why it works:**
- Evaluates actual content relevance, not chunk ID matching
- Works regardless of how documents are indexed
- Considers whether retrieved docs can answer the question
- More robust and meaningful metric

---

## Bug #6: (Template for Future Bugs)

**Date:**
**File:**
**Error:**
```
(paste error message here)
```

**Root Cause:**
(explain why it happened)

**Solution:**
```python
# BEFORE (broken)

# AFTER (fixed)
```

**Why it works:**
(explain the fix)

---

## Summary Table

| Bug # | Issue | Root Cause | File |
|-------|-------|------------|------|
| 1 | ModuleNotFoundError | Relative path in sys.path | compare_experiments.py |
| 2 | 401 Unauthorized | Spaces/quotes in .env | .env |
| 3 | Faithfulness always 0 | Thread ID mismatch + truncation | run_langsmith_experiment.py |
| 4 | Local JSON all nulls | Treating object as dict | run_langsmith_experiment.py |
| 5 | retrieval_f1 unreliable | Hardcoded chunk IDs | run_langsmith_experiment.py |

---

# Experiment 4: opt-v3-forced-search

**Problem:** In v2, search was "enforced" via system prompt, but LLM could still ignore it.

**Solution:** Graph-based enforcement - search is structurally required.

## Changes Made

### 1. New State Fields (`agent_state.py`)
```python
# Added to track search flow
search_results: List[Any] = []
search_performed: bool = False
search_query: str = ""
retrieved_docs: List[Any] = []
parent_ids: List[str] = []
```

### 2. New Graph Flow (`graph.py`)
```
BEFORE (v2): LLM decides whether to call tools
START → summarize → rewrite → retrieve_agent (LLM picks tools) → END

AFTER (v3): Graph enforces search
START → summarize → rewrite → forced_search → retrieve_parents → generate_answer → END
```

### 3. Key Nodes

| Node | Purpose |
|------|---------|
| `forced_search_node` | ALWAYS runs search, stores results in state |
| `retrieve_parents_node` | ALWAYS retrieves parent docs from search results |
| `generate_answer_node` | ONLY uses retrieved docs, no tool calling |

### 4. Increased Snippet Size
```python
# v1/v2: 100 chars (too short)
"snippet": doc.page_content[:100] + "..."

# v3: 250 chars (better context)
"snippet": doc.page_content[:250] + "..."
```

### 5. Why This Works
- **v2**: LLM sees "ALWAYS search" prompt but can ignore it
- **v3**: Graph edges physically route through search - no way to skip
- **v3**: Larger snippets give better context for relevance judgment

### Files Created
- `experiments/opt_v3_forced_search/agent_state.py`
- `experiments/opt_v3_forced_search/tools.py`
- `experiments/opt_v3_forced_search/graph.py`

### 6. Query-Focused Extraction (Compression Layer)

**Problem:** Raw parent chunks are too long (2000+ chars), wasting tokens and diluting relevance.

**Solution:** Add extraction step between retrieval and answer generation.

**New State Fields:**
```python
extracted_content: str = ""      # Compressed, relevant sentences only
extraction_logs: List[Dict] = [] # Debug info for each chunk
```

**New Node: `extract_relevant_node`**
```python
def extract_relevant_node(state: AgentState):
    """Extract only sentences that directly answer the question"""
    # For each retrieved doc:
    # 1. Send chunk + question to small LLM
    # 2. Extract relevant sentences (preserve original wording)
    # 3. Fallback to 300-char snippet if extraction < 50 chars
    # 4. Log: chunk_id, original_length, extracted_length, used_fallback
```

**Updated Graph Flow:**
```
BEFORE: search → retrieve → generate_answer
AFTER:  search → retrieve → EXTRACT → generate_answer
```

**Why it works:**
- Compresses 2000+ char chunks to ~200-500 char relevant excerpts
- Small LLM (qwen2.5:3b) is fast and cheap
- Preserves original wording - no hallucination risk
- Fallback ensures we never lose context entirely
- Logs enable debugging extraction quality

### Run It
```bash
python experiments/run_langsmith_experiment.py opt-v3-forced-search
```

---

## Prevention Tips

1. **Always use absolute paths** for `sys.path` modifications
2. **No spaces around `=`** in `.env` files, no quotes around values
3. **Reuse variables** instead of calling time-based functions multiple times
4. **Test with debug prints** to verify data is flowing correctly between functions
5. **Check evaluator inputs** - if metrics are always 0 or 1, the input data may be wrong
6. **Use graph structure** to enforce behavior, not just prompts
