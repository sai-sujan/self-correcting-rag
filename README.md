# Self-Correcting RAG System

A production-grade **Self-Correcting Retrieval Augmented Generation (RAG)** system designed to eliminate hallucinations and retrieval failures common in standard RAG applications.


<img width="1697" height="765" alt="Screenshot 2026-01-31 at 9 01 11 PM" src="https://github.com/user-attachments/assets/e29aa3f1-52ee-464f-98ae-ed78d64de2de" />


## 🚀 Overview

This project implements a sophisticated RAG architecture that evolved through rigorous experimentation to achieve:
- **94% Faithfulness** (verified by LLM Judge)
- **<5s Latency** (optimized from 15s+)
- **Zero Hallucinations** on out-of-scope queries

The system uses a **Heterogeneous Model Architecture**, deploying `qwen2.5:7b` for complex reasoning and `qwen2.5:3b` for high-speed extraction and judging, orchestrated by `LangGraph`.

## 🛠️ Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | `LangGraph` | State management, cyclic graphs, persistence |
| **LLM Inference** | `Ollama` | Local model serving |
| **Main Model** | `Qwen 2.5 (7b)` | Reasoning, final answer generation |
| **Small Model** | `Qwen 2.5 (3b)` | Rewriting, Extraction, Judging |
| **Vector DB** | `Qdrant` | Hybrid Search (Dense + Sparse) |
| **Embeddings** | `HuggingFace` | Dense vector generation |
| **Reranker** | `CrossEncoder` | Semantic re-scoring |

## 🏗️ Architecture

The system follows a strict "Assembly Line" pipeline:

1.  **Ingestion**:
    - "Parent-Child" Indexing Strategy: Splits documents by headers (Parent) and indexes smaller sliding windows (Child) to maximize retrieval accuracy while preserving context.
2.  **Retrieval**:
    - **Query Rewriting**: Resolves pronouns and ambiguity.
    - **Multi-Query**: Generates multiple search variations.
    - **Hybrid Search**: Combines Dense (Semantic) and Sparse (Keyword) search.
    - **Reranking**: Scores results using Cross-Encoders.
3.  **Generation**:
    - **Batched Extraction**: Compresses retrieved chunks into key bullet points in a single LLM call.
    - **Answering**: Generates answers strictly from extracted points.
4.  **Correction (Self-Correction)**:
    - **Judgement**: An LLM Judge evaluates the faithfulness of the answer.
    - **Retry Loop**: If unfaithful, the system rewrites the query and searches again.

## 📥 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/sai-sujan/self-correcting-rag.git
    cd self-correcting-rag
    ```

2.  **Create a virtual environment**:
    ```bash
    conda create -n rag python=3.11 && conda activate rag
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup Ollama**:
    Ensure you have [Ollama](https://ollama.com/) installed and pull the required models:
    ```bash
    ollama pull qwen2.5:7b
    ollama pull qwen2.5:3b
    ```

## 🏃 Usage

### Running the Application
The `streamlit_app.py` is configured to use the optimized V7 settings by default.

```bash
streamlit run streamlit_app.py
```

### Running Experiments
The project includes a unified runner to execute and compare different RAG strategies.

```bash
# List all available experiments
python run_experiment.py --list

# Run the Best Configuration (V7 Config D)
python run_experiment.py --experiment opt-v7-D
```

## 📂 Project Structure

- `core/`: Production code & Shared Components (Nodes, State, Tools)
- `experiments/`: Historical experiments and research findings
- `parent_store/`: Storage for full text documents
- `qdrant_db/`: Vector database indices
- `docs/`: Input documents (PDFs)
