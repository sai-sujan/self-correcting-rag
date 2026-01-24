# Self-Correcting Multi-Agent RAG System - Technical Specification
**Version**: 1.0  
**Last Updated**: January 2026

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

### 1.1 High-Level Components
```
┌─────────────────────────────────────────────────────────────────┐
│                     User Query Interface                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              LangGraph State Machine (Orchestrator)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Summarization│  │Query Rewriting│  │  Retrieval   │          │
│  │    Agent     │  │     Agent     │  │    Agent     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Qdrant Vector Database                        │
│     ┌──────────────────┐         ┌──────────────────┐          │
│     │  Dense Vectors   │         │  Sparse Vectors  │          │
│     │  (Semantic)      │         │  (Keyword/BM25)  │          │
│     └──────────────────┘         └──────────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Document Processing Pipeline                        │
│  PDF Loader → Chunking → Embedding → Vector Storage             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. DIMENSIONAL SPECIFICATIONS

### 2.1 Vector Embeddings

**Dense Vector Embeddings (Semantic Search)**
- **Model**: `nomic-embed-text` (via Ollama)
- **Dimension**: `768` (fixed)
- **Data Type**: `float32`
- **Memory per vector**: `768 × 4 bytes = 3,072 bytes = 3 KB`
- **Normalization**: L2 normalized (unit vectors)
- **Distance Metric**: Cosine similarity
- **Value Range**: `[-1.0, 1.0]` after normalization

**Sparse Vector Embeddings (Keyword/BM25)**
- **Dimension**: Variable (vocabulary-dependent)
- **Typical Dimension**: `10,000 - 50,000` terms
- **Data Type**: `float32`
- **Sparsity**: ~95-99% zeros
- **Storage Format**: Sparse representation (only non-zero values stored)
- **Distance Metric**: Dot product

### 2.2 Text Chunking Dimensions

**Parent Chunks (Hierarchical Strategy)**
- **Size**: `1,500 characters`
- **Overlap**: `300 characters` (20%)
- **Estimated Tokens**: ~375 tokens (assuming 4 chars/token)
- **Purpose**: Context preservation

**Child Chunks (Retrieval Units)**
- **Size**: `500 characters`
- **Overlap**: `100 characters` (20%)
- **Estimated Tokens**: ~125 tokens
- **Chunks per Parent**: Typically 3-4 child chunks
- **Purpose**: Precise retrieval

**Justification for Chunk Sizes**:
- 500 chars balances granularity vs context
- Small enough for precise retrieval
- Large enough to contain complete thoughts
- Fits well within LLM context windows with room for query + system prompts

### 2.3 LLM Context Windows

**Model**: `llama3.2` (via Ollama)
- **Maximum Context Length**: `128,000 tokens`
- **Practical Working Limit**: `8,000 - 16,000 tokens` (for speed)
- **Output Token Limit**: `2,048 tokens` (configured)

**Token Budget Allocation**:
```
Total Available: 16,000 tokens
├── System Prompt: ~500 tokens
├── Conversation History: ~2,000 tokens (managed via pruning)
├── Retrieved Context: ~3,000 tokens (6 chunks × 500 chars)
├── User Query: ~100 tokens
└── Response Generation: ~2,048 tokens
───────────────────────────────────
Buffer/Safety Margin: ~8,352 tokens
```

### 2.4 Conversation Memory

**Memory Window**:
- **Last N Messages**: `10 messages` (5 user + 5 assistant)
- **Token Limit**: `2,000 tokens`
- **Pruning Strategy**: Sliding window with summarization
- **Compression**: Older messages summarized at 5:1 ratio

**Token Estimation**:
- Average message length: ~200 tokens
- 10 messages × 200 = 2,000 tokens baseline
- Pruned to maintain under 2,000 token limit

---

## 3. MODEL SPECIFICATIONS

### 3.1 Embedding Model

**nomic-embed-text**
```yaml
Model Architecture: Transformer-based encoder
Parameters: ~137M parameters
Embedding Dimension: 768
Max Input Length: 8,192 tokens
Training Data: ~235M text pairs
Performance:
  - MTEB Average: 62.39
  - Retrieval Tasks: Strong performance on long-context
Latency: ~50-100ms per batch (batch size: 32)
```

### 3.2 Language Model

**llama3.2** (via Ollama)
```yaml
Model Architecture: Decoder-only Transformer
Parameters: 3B parameters
Context Window: 128K tokens
Quantization: Q4_K_M (4-bit quantization)
Memory Footprint: ~2GB RAM
Inference Speed: 
  - Tokens/second: ~30-50 (CPU)
  - Tokens/second: ~100-150 (GPU)
Temperature: 0.1 (low for factual consistency)
Top-p: 0.9
Top-k: 40
Repeat Penalty: 1.1
```

---

## 4. VECTOR DATABASE CONFIGURATION

### 4.1 Qdrant Setup

**Collection Schema**:
```python
Collection Name: "documents_hybrid"
Vector Config:
  dense:
    size: 768
    distance: Cosine
  sparse:
    modifier: Idf  # Inverse Document Frequency weighting
    
Payload Schema:
  - text: string (indexed)
  - metadata: object
      ├── source: string
      ├── page: integer
      ├── chunk_id: string
      ├── parent_chunk_id: string
      └── timestamp: datetime

Index Configuration:
  - HNSW for dense vectors
      ├── m: 16 (connections per node)
      ├── ef_construct: 100
      └── ef: 128
  - Inverted Index for sparse vectors
      └── Full-text search enabled
```

**Storage Requirements**:
```
Per Document (assuming 100-page PDF):
├── Raw Text: ~500 KB
├── Child Chunks: ~200 chunks
├── Dense Vectors: 200 × 3 KB = 600 KB
├── Sparse Vectors: 200 × ~5 KB = 1 MB (compressed)
├── Metadata: ~50 KB
└── HNSW Index Overhead: ~200 KB
─────────────────────────────────────
Total per 100-page doc: ~2.55 MB
```

### 4.2 Hybrid Search Configuration

**Search Parameters**:
```yaml
Top-K Results: 6 chunks
Dense Weight (alpha): 0.7 (70% semantic)
Sparse Weight (beta): 0.3 (30% keyword)

Search Process:
  1. Dense Search: Retrieve top-20 by semantic similarity
  2. Sparse Search: Retrieve top-20 by BM25
  3. Fusion: RRF (Reciprocal Rank Fusion)
     Formula: score = 1/(k + rank_dense) + 1/(k + rank_sparse)
     k = 60 (RRF constant)
  4. Re-ranking: Final top-6 selected
```

**Retrieval Latency**:
- Dense search: ~20-30ms
- Sparse search: ~15-25ms
- Fusion & re-ranking: ~5ms
- **Total**: ~40-60ms per query

---

## 5. AGENT SPECIFICATIONS

### 5.1 Agent Roles & Prompts

**Summarization Agent**:
```yaml
Purpose: Condense conversation history
Input Dimensions:
  - Max History: 10 messages
  - Max Tokens: 2,000
Output Dimensions:
  - Summary Length: ~200-300 tokens
  - Compression Ratio: ~7:1
Activation: Every 5 messages or when token limit exceeded
Model: llama3.2
Temperature: 0.3 (slightly creative for summarization)
```

**Query Rewriting Agent**:
```yaml
Purpose: Expand/clarify user queries for better retrieval
Input Dimensions:
  - User Query: ~50-200 tokens
  - Conversation Summary: ~200 tokens
Output Dimensions:
  - Rewritten Queries: 3 variations
  - Each Query: ~30-100 tokens
Techniques:
  - Synonym expansion
  - Context integration
  - Question decomposition
Model: llama3.2
Temperature: 0.5 (balanced creativity)
```

**Retrieval Agent**:
```yaml
Purpose: Fetch relevant documents and self-correct
Input Dimensions:
  - Queries: 3 rewritten queries
  - Top-K per query: 6 chunks
Output Dimensions:
  - Total Retrieved: Up to 18 chunks initially
  - Deduplicated: ~6-10 unique chunks
  - Final Context: Top 6 chunks by score
Grading Mechanism:
  - Relevance Score: 0-10 (LLM-based)
  - Threshold: Score ≥ 7 to keep
  - Fallback: Web search if <3 relevant chunks
```

### 5.2 LangGraph State Machine

**State Schema**:
```python
class GraphState(TypedDict):
    # Input
    question: str                    # User query (max 512 tokens)
    
    # Conversation Context
    chat_history: List[BaseMessage]  # Last 10 messages
    summary: str                     # Condensed history (max 300 tokens)
    
    # Retrieval
    rewritten_queries: List[str]     # 3 query variations
    documents: List[Document]        # Retrieved chunks (6-10)
    relevance_scores: List[float]    # 0-10 range
    
    # Generation
    generation: str                  # Final answer (max 2048 tokens)
    
    # Metadata
    retrieval_count: int             # Attempt counter
    web_search_used: bool            # Fallback flag

Transitions:
  1. START → summarize_history (conditional)
  2. summarize_history → rewrite_query
  3. rewrite_query → retrieve_documents
  4. retrieve_documents → grade_documents
  5. grade_documents → generate (if relevant)
  6. grade_documents → web_search (if not relevant)
  7. web_search → generate
  8. generate → END

Max Iterations: 3 (prevents infinite loops)
```

---

## 6. PERFORMANCE BENCHMARKS

### 6.1 Latency Breakdown

**End-to-End Query Processing**:
```
Component                    | Latency (ms) | % of Total
─────────────────────────────┼──────────────┼───────────
History Summarization        |     500      |    12%
Query Rewriting (3 queries)  |     800      |    20%
Embedding Generation (3x)    |     150      |     4%
Vector Search (hybrid)       |      60      |     1.5%
Document Grading (6 docs)    |     400      |    10%
Response Generation          |   2,000      |    50%
Overhead & State Management  |     100      |    2.5%
─────────────────────────────┼──────────────┼───────────
TOTAL (Typical)              |   4,010      |   100%
TOTAL (With Web Search)      | 6,000-8,000  |     -
```

### 6.2 Token Usage Per Query

**Baseline (No Conversation History)**:
```
Component              | Tokens
───────────────────────┼────────
System Prompt          |    500
User Query             |    100
Retrieved Context      |  3,000
Response               |  2,000
───────────────────────┼────────
Total                  |  5,600
```

**With Conversation History**:
```
Component              | Tokens
───────────────────────┼────────
System Prompt          |    500
Conversation Summary   |    200
User Query             |    100
Retrieved Context      |  3,000
Response               |  2,000
───────────────────────┼────────
Total                  |  5,800
```

### 6.3 Resource Requirements

**Minimum System Requirements**:
```yaml
CPU: 4 cores (Intel i5 or equivalent)
RAM: 8 GB (4 GB for Ollama models + 2 GB for Qdrant + 2 GB OS)
Storage: 10 GB
  ├── Ollama models: 4 GB
  ├── Qdrant data: 2-5 GB (depends on document corpus)
  └── Application code: 100 MB

Network: Required for initial model downloads
GPU: Optional (improves inference speed 3-5x)
```

**Recommended System Requirements**:
```yaml
CPU: 8 cores
RAM: 16 GB
GPU: NVIDIA with 8GB VRAM (RTX 3060 or better)
Storage: 50 GB SSD
```

---

## 7. DATA FLOW DIAGRAMS

### 7.1 Document Ingestion Flow

```
PDF Document (Input)
│
├─ Step 1: Load PDF
│  ├─ Library: PyPDF2
│  ├─ Output: Raw text string
│  └─ Size: Variable (typically 100-500 KB per doc)
│
├─ Step 2: Hierarchical Chunking
│  ├─ Parent Chunks: 1,500 chars, 300 overlap
│  ├─ Child Chunks: 500 chars, 100 overlap
│  ├─ Output: List of chunk objects
│  └─ Count: ~200 chunks per 100-page document
│
├─ Step 3: Embedding Generation
│  ├─ Model: nomic-embed-text
│  ├─ Input: Child chunk text (500 chars)
│  ├─ Output: Dense vector (768 dims) + Sparse vector
│  ├─ Batch Size: 32 chunks
│  └─ Latency: ~50ms per batch
│
└─ Step 4: Vector Storage
   ├─ Database: Qdrant
   ├─ Index: HNSW + Inverted Index
   ├─ Payload: {text, metadata, parent_id}
   └─ Memory: ~2.5 MB per 100-page doc
```

### 7.2 Query Processing Flow

```
User Query
│
├─ Step 1: Conversation Management
│  ├─ Check: Message count > 10?
│  ├─ Yes → Summarize last 10 messages
│  │   ├─ Input: 10 messages (~2,000 tokens)
│  │   └─ Output: Summary (~200 tokens)
│  └─ No → Skip summarization
│
├─ Step 2: Query Rewriting
│  ├─ Input: Original query + summary
│  ├─ Agent: Query Rewriter
│  ├─ Output: 3 rewritten queries
│  │   ├─ Query 1: Synonym expansion
│  │   ├─ Query 2: Context integration
│  │   └─ Query 3: Decomposed question
│  └─ Latency: ~800ms
│
├─ Step 3: Hybrid Retrieval
│  ├─ Embed each query (3 × 768-dim vectors)
│  ├─ Dense Search: Top-20 per query (Cosine)
│  ├─ Sparse Search: Top-20 per query (BM25)
│  ├─ Fusion: RRF algorithm
│  ├─ Output: Top-6 chunks
│  └─ Latency: ~180ms total
│
├─ Step 4: Relevance Grading
│  ├─ Input: 6 chunks + query
│  ├─ Grader: LLM-based scoring (0-10)
│  ├─ Threshold: Keep if score ≥ 7
│  ├─ Output: Filtered chunks (typically 4-6)
│  └─ Latency: ~400ms
│
├─ Step 5: Self-Correction
│  ├─ Check: Relevant chunks ≥ 3?
│  ├─ Yes → Proceed to generation
│  └─ No → Trigger web search fallback
│      ├─ Search for: Original query
│      ├─ Retrieve: Top-3 web results
│      └─ Latency: +2,000-4,000ms
│
└─ Step 6: Response Generation
   ├─ Input: Query + context + history
   ├─ Context Window: ~5,800 tokens
   ├─ Model: llama3.2
   ├─ Output: Answer (max 2,048 tokens)
   └─ Latency: ~2,000ms
```

---

## 8. QUALITY METRICS & EVALUATION

### 8.1 Retrieval Metrics

**Precision & Recall**:
- **Precision@K**: Relevant chunks / K retrieved
  - Target: >0.7 (70% relevant)
  - Measured at K=6

- **Recall@K**: Retrieved relevant / Total relevant
  - Target: >0.6 (catch 60% of all relevant chunks)

- **MRR (Mean Reciprocal Rank)**: 1 / rank of first relevant
  - Target: >0.7

**Faithfulness**:
- **Metric**: Generated answer grounded in retrieved context
- **Measurement**: LLM-as-judge (0-10 scale)
- **Target**: >8.0

**Answer Relevance**:
- **Metric**: Answer addresses the query
- **Measurement**: LLM-as-judge (0-10 scale)
- **Target**: >8.5

### 8.2 Response Time SLAs

```yaml
P50 (Median): <4 seconds
P90: <6 seconds
P99: <10 seconds
Timeout: 30 seconds

Breakdown by Component:
  - Retrieval: <200ms (P90)
  - Grading: <500ms (P90)
  - Generation: <3,000ms (P90)
```

### 8.3 Cost Analysis (Local Deployment)

**Compute Cost** (per 1,000 queries):
```
Ollama Inference:
  - Embedding: ~3 million tokens
  - LLM Generation: ~8 million tokens
  - Total Tokens: ~11 million
  - Cost: $0 (local inference)

Cloud Alternative (for comparison):
  - OpenAI GPT-4: ~$110
  - Anthropic Claude: ~$90
  - Local Savings: 100%

Infrastructure:
  - Electricity: ~$0.50 (GPU usage)
  - Depreciation: ~$0.20 (hardware amortization)
  - Total: ~$0.70 per 1,000 queries
```

---

## 9. SCALING CONSIDERATIONS

### 9.1 Horizontal Scaling

**Vector Database (Qdrant)**:
```yaml
Single Node Capacity:
  - Documents: ~100,000 (100-page PDFs)
  - Vectors: ~20 million
  - Memory: ~60 GB
  - QPS: ~500 queries/second

Cluster Configuration (for scale):
  - Nodes: 3-5
  - Replication Factor: 2
  - Sharding: By collection
  - Capacity: 5x single node
```

**LLM Inference (Ollama)**:
```yaml
Single Instance:
  - Concurrent Requests: 4-8 (depending on GPU)
  - Throughput: ~10 queries/minute

Load Balancing:
  - Instances: 3-5 replicas
  - Load Balancer: Nginx
  - Total Throughput: ~50 queries/minute
```

### 9.2 Optimization Strategies

**Embedding Caching**:
- Cache frequently queried embeddings
- Hit Rate Target: >30%
- Latency Reduction: ~150ms per cache hit

**Batch Processing**:
- Batch Size: 32 documents
- Throughput Increase: 4x vs sequential

**Quantization**:
- Model: Already Q4 quantized (llama3.2)
- Memory Savings: 75% vs FP32
- Accuracy Loss: <2%

---

## 10. SECURITY & PRIVACY

### 10.1 Data Handling

**Document Storage**:
```yaml
Encryption at Rest: AES-256
Encryption in Transit: TLS 1.3
Access Control: Role-based (if deployed)
Data Retention: Configurable (default: indefinite)
PII Handling: User responsibility (local deployment)
```

### 10.2 API Security

**Ollama**:
```yaml
Default: localhost:11434 (no auth)
Production: Reverse proxy with API keys
Rate Limiting: 100 requests/minute per IP
```

**Qdrant**:
```yaml
Default: localhost:6333 (no auth)
Production: API key authentication
TLS: Enabled for remote access
```

---

## 11. APPENDIX: FORMULAS & ALGORITHMS

### 11.1 Cosine Similarity

```
cos(θ) = (A · B) / (||A|| × ||B||)

Where:
  A, B = vectors (768-dim)
  · = dot product
  ||A|| = L2 norm of A
  
Range: [-1, 1]
  1.0 = identical
  0.0 = orthogonal
 -1.0 = opposite
```

### 11.2 BM25 Score

```
BM25(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))

Where:
  D = document
  Q = query
  qi = query term i
  f(qi, D) = term frequency
  |D| = document length
  avgdl = average document length
  k1 = 1.5 (term saturation)
  b = 0.75 (length normalization)
  IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
```

### 11.3 Reciprocal Rank Fusion (RRF)

```
RRF(d) = Σ 1 / (k + rank_i(d))

Where:
  d = document
  rank_i(d) = rank of d in ranking i
  k = 60 (constant)
  
Combines rankings from dense + sparse search
Higher score = better overall relevance
```

---

## 12. CONFIGURATION FILES

### 12.1 System Configuration

```yaml
# config/system.yaml
embedding:
  model: "nomic-embed-text"
  dimension: 768
  batch_size: 32
  
llm:
  model: "llama3.2"
  temperature: 0.1
  max_tokens: 2048
  context_window: 16000
  
chunking:
  parent_size: 1500
  parent_overlap: 300
  child_size: 500
  child_overlap: 100
  
retrieval:
  top_k: 6
  dense_weight: 0.7
  sparse_weight: 0.3
  relevance_threshold: 7.0
  
memory:
  max_messages: 10
  max_tokens: 2000
  summarize_every: 5
  
performance:
  timeout_seconds: 30
  max_retries: 3
```

---

## VERSION HISTORY

**v1.0** (Current)
- Initial comprehensive specification
- All dimensions explicitly documented
- Performance benchmarks included
- Scaling considerations added

