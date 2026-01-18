#!/usr/bin/env python
"""
Run experiments using LangSmith's evaluate() function.

This properly integrates with LangSmith's Experiments feature:
- Each run creates a new "Experiment" in LangSmith
- Experiments are linked to the dataset
- You can compare experiments side-by-side in LangSmith UI
- Full history is preserved

Usage:
    python experiments/run_langsmith_experiment.py baseline
    python experiments/run_langsmith_experiment.py opt-v1-minimal-chunks
    python experiments/run_langsmith_experiment.py opt-v2-strict-search
    python experiments/run_langsmith_experiment.py --all
    python experiments/run_langsmith_experiment.py --list-experiments
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
from datetime import datetime
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np

from core.config import llm, dense_embeddings
from experiments.experiment_registry import EXPERIMENTS, load_experiment_graph, list_experiments

# LangSmith client
client = Client()

# Dataset name (must match setup_langsmith_dataset.py)
DATASET_NAME = "rag-evaluation-tests"


# ============== EVALUATORS ==============

def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def extract_chunk_ids(messages):
    """Extract parent_ids from tool messages"""
    chunk_ids = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            ids = re.findall(r"'parent_id': '([^']+)'", str(msg.content))
            chunk_ids.extend(ids)
    return list(set(chunk_ids))


def extract_retrieved_docs(messages):
    """Extract retrieved content for faithfulness check"""
    docs = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            content = str(msg.content)[:500] + "..."
            docs.append(content)
    return "\n---\n".join(docs) if docs else "No documents retrieved"


def faithfulness_evaluator(run: Run, example: Example) -> dict:
    """LLM judges if answer is grounded in retrieved docs"""
    answer = run.outputs.get("answer", "")
    retrieved_docs = run.outputs.get("retrieved_docs", "No documents")

    prompt = f"""You are evaluating if an AI answer is grounded in provided documents.

Retrieved Documents:
{retrieved_docs[:2000]}

Generated Answer:
{answer}

Rate FAITHFULNESS (0.0 to 1.0):
0.0 = Makes claims NOT in documents (hallucination)
0.5 = Partially supported
1.0 = Every claim directly from documents

Respond ONLY with a number between 0 and 1."""

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))  # Clamp to 0-1
    except:
        score = 0.0

    return {"key": "faithfulness", "score": score}


def retrieval_f1_evaluator(run: Run, example: Example) -> dict:
    """Measure if we retrieved the RIGHT documents"""
    retrieved = run.outputs.get("retrieved_chunks", [])
    expected = example.outputs.get("expected_chunks", [])

    if not expected:
        return {"key": "retrieval_f1", "score": 1.0}

    retrieved_set = set(retrieved)
    expected_set = set(expected)

    if not retrieved_set:
        return {"key": "retrieval_f1", "score": 0.0}

    intersection = retrieved_set & expected_set
    precision = len(intersection) / len(retrieved_set)
    recall = len(intersection) / len(expected_set)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"key": "retrieval_f1", "score": f1}


def semantic_similarity_evaluator(run: Run, example: Example) -> dict:
    """Compare answer to reference using embeddings"""
    answer = run.outputs.get("answer", "")
    reference = example.outputs.get("reference_answer", "")

    if not reference:
        return {"key": "semantic_similarity", "score": 1.0}

    try:
        answer_emb = dense_embeddings.embed_query(answer)
        reference_emb = dense_embeddings.embed_query(reference)
        score = cosine_similarity(np.array(answer_emb), np.array(reference_emb))
    except:
        score = 0.0

    return {"key": "semantic_similarity", "score": score}


def answer_quality_evaluator(run: Run, example: Example) -> dict:
    """Structural quality checks"""
    answer = run.outputs.get("answer", "")

    checks = [
        len(answer) > 50,  # has content
        len(answer) > 100,  # sufficient length
        len(answer.split()) > 20,  # not too short
        "error" not in answer.lower(),  # no error
        "don't have" not in answer.lower()  # not a cop-out
    ]

    score = sum(checks) / len(checks)
    return {"key": "answer_quality", "score": score}


def latency_evaluator(run: Run, example: Example) -> dict:
    """Track response time"""
    if run.end_time and run.start_time:
        latency = (run.end_time - run.start_time).total_seconds()
    else:
        latency = 0
    return {"key": "latency", "score": latency}


# ============== TARGET FUNCTION ==============

def create_target_function(graph):
    """Create a target function that runs the graph and extracts outputs"""

    def target(inputs: dict) -> dict:
        """Run the RAG graph and return structured outputs"""
        question = inputs.get("question", "")

        # Run the graph
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": f"eval-{datetime.now().timestamp()}"}}
        )

        # Extract answer
        answer = result["messages"][-1].content if result["messages"] else ""

        # Get state to extract tool outputs
        config = {"configurable": {"thread_id": f"eval-{datetime.now().timestamp()}"}}
        try:
            state = graph.get_state(config)
            messages = state.values.get("messages", [])
        except:
            messages = result.get("messages", [])

        # Extract retrieved info
        retrieved_docs = extract_retrieved_docs(messages)
        retrieved_chunks = extract_chunk_ids(messages)

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "retrieved_chunks": retrieved_chunks
        }

    return target


# ============== MAIN FUNCTIONS ==============

def run_experiment(experiment_id: str):
    """Run a single experiment using LangSmith evaluate()"""

    if experiment_id not in EXPERIMENTS:
        print(f"❌ Unknown experiment: {experiment_id}")
        print(f"   Available: {list(EXPERIMENTS.keys())}")
        return

    config = EXPERIMENTS[experiment_id]
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{experiment_id}-{run_id.replace('_', '-')}"

    print(f"\n{'='*70}")
    print(f"🧪 Running Experiment: {config['name']}")
    print(f"📝 Description: {config['description']}")
    print(f"🏷️  Experiment Name: {experiment_name}")
    print(f"📊 Dataset: {DATASET_NAME}")
    print('='*70)

    # Load the graph
    graph = load_experiment_graph(experiment_id)

    # Create target function
    target = create_target_function(graph)

    # Run evaluation using LangSmith
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            faithfulness_evaluator,
            retrieval_f1_evaluator,
            semantic_similarity_evaluator,
            answer_quality_evaluator,
            latency_evaluator
        ],
        experiment_prefix=experiment_id,
        metadata={
            "experiment_id": experiment_id,
            "experiment_name": config["name"],
            "description": config["description"],
            "version": config.get("version", "1.0")
        }
    )

    # === SAVE LOCALLY ===
    local_results = []
    for result in results:
        local_results.append({
            "inputs": result.get("input", {}),
            "outputs": result.get("output", {}),
            "reference": result.get("reference", {}),
            "evaluations": {
                "faithfulness": result.get("feedback", {}).get("faithfulness", {}).get("score"),
                "retrieval_f1": result.get("feedback", {}).get("retrieval_f1", {}).get("score"),
                "semantic_similarity": result.get("feedback", {}).get("semantic_similarity", {}).get("score"),
                "answer_quality": result.get("feedback", {}).get("answer_quality", {}).get("score"),
                "latency": result.get("feedback", {}).get("latency", {}).get("score"),
            }
        })

    # Calculate aggregate metrics
    scores = {
        "faithfulness": [],
        "retrieval_f1": [],
        "semantic_similarity": [],
        "answer_quality": [],
        "latency": []
    }

    for r in local_results:
        for key in scores:
            val = r["evaluations"].get(key)
            if val is not None:
                scores[key].append(val)

    metrics = {}
    for key, vals in scores.items():
        if vals:
            metrics[f"avg_{key}"] = round(sum(vals) / len(vals), 3)

    # Build report
    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_config": config,
        "dataset": DATASET_NAME,
        "metrics": metrics,
        "results": local_results
    }

    # Save to results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(results_dir, f"{experiment_id}_{run_id}.json")
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Experiment Complete: {config['name']}")
    print(f"\n📊 Metrics:")
    for key, val in metrics.items():
        print(f"   {key}: {val}")
    print(f"\n💾 Local: {filename}")
    print(f"\n📋 LangSmith:")
    print(f"   https://smith.langchain.com/")
    print(f"   Go to: Datasets → {DATASET_NAME} → Experiments")
    print(f"   Look for: {experiment_name}")
    print('='*70)

    return results


def run_all_experiments():
    """Run all registered experiments"""
    print("\n🚀 Running ALL Experiments\n")

    for exp_id in EXPERIMENTS.keys():
        run_experiment(exp_id)
        print("\n" + "-"*70 + "\n")


def list_langsmith_experiments():
    """List all experiments in LangSmith for this dataset"""
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)

        # Get experiments (called "sessions" in some API versions)
        print(f"\n📋 Experiments for Dataset: {DATASET_NAME}")
        print("="*70)

        # List recent runs/experiments
        projects = list(client.list_projects())

        # Filter to show self-correcting-rag project info
        for project in projects:
            if "self-correcting-rag" in project.name.lower() or any(
                exp_id in project.name for exp_id in EXPERIMENTS.keys()
            ):
                print(f"\n🧪 {project.name}")
                print(f"   ID: {project.id}")
                if project.extra:
                    print(f"   Runs: {project.extra.get('run_count', 'N/A')}")

        print("\n" + "="*70)
        print("\n💡 To compare experiments:")
        print("   1. Go to https://smith.langchain.com/")
        print(f"   2. Open Datasets → {DATASET_NAME}")
        print("   3. Click 'Experiments' tab")
        print("   4. Select experiments to compare")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run LangSmith experiments")
    parser.add_argument("experiment", nargs="?", help="Experiment ID to run")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--list", action="store_true", help="List registered experiments")
    parser.add_argument("--list-experiments", action="store_true", help="List LangSmith experiments")

    args = parser.parse_args()

    if args.list:
        list_experiments()
    elif args.list_experiments:
        list_langsmith_experiments()
    elif args.all:
        run_all_experiments()
    elif args.experiment:
        run_experiment(args.experiment)
    else:
        parser.print_help()
        print("\n📋 Available experiments:")
        for exp_id in EXPERIMENTS.keys():
            print(f"  - {exp_id}")
        print("\n💡 First run: python experiments/setup_langsmith_dataset.py")


if __name__ == "__main__":
    main()
