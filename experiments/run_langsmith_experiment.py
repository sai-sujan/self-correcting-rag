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
DATASET_NAME = "rag-evaluation-tests-corrected-with-mistral-7b"


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
            content = str(msg.content)
            # Only include substantial content (likely from retrieve_parent_chunks)
            if len(content) > 100:
                docs.append(content[:1500])  # More context for faithfulness check
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


def retrieval_relevance_evaluator(run: Run, example: Example) -> dict:
    """LLM judges if retrieved documents are relevant to the question"""
    question = run.inputs.get("question", "")
    retrieved_docs = run.outputs.get("retrieved_docs", "No documents")

    # If no documents retrieved, score is 0
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

Consider:
- Do the documents discuss the topic asked about?
- Do they contain the specific information needed to answer?
- Would someone be able to answer the question using ONLY these documents?

Respond ONLY with a number between 0 and 1."""

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))  # Clamp to 0-1
    except:
        score = 0.0

    return {"key": "retrieval_relevance", "score": score}


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
    """Structural quality checks - handles both answers AND valid 'no info' responses"""
    answer = run.outputs.get("answer", "")
    answer_lower = answer.lower()

    # Valid "out of scope" phrases - these are CORRECT responses, not failures
    VALID_NO_INFO_PHRASES = [
        "documents don't contain",
        "not covered in the documents",
        "no relevant information found",
        "outside the scope",
        "couldn't find information",
        "searched the documents but found no",
        "don't have information about"
    ]

    # Check if this is a valid "no info" response
    is_valid_no_info = any(phrase in answer_lower for phrase in VALID_NO_INFO_PHRASES)

    if is_valid_no_info:
        # Valid "no info" response should score high - this is correct behavior
        # Check that it's well-formed (has content, no errors)
        checks = [
            len(answer) > 30,  # has some content explaining why
            "error" not in answer_lower,  # no error messages
            True,  # valid no-info is acceptable
            True,  # valid no-info is acceptable
            True   # valid no-info is acceptable
        ]
    else:
        # Normal answer - check quality
        checks = [
            len(answer) > 50,  # has content
            len(answer) > 100,  # sufficient length
            len(answer.split()) > 20,  # not too short
            "error" not in answer_lower,  # no error
            "i don't know" not in answer_lower  # not a lazy cop-out (without searching)
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

        # Use consistent thread_id for both invoke and get_state
        thread_id = f"eval-{datetime.now().timestamp()}"
        config = {"configurable": {"thread_id": thread_id}}

        # Run the graph
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )

        # Extract answer from result
        answer = result["messages"][-1].content if result.get("messages") else ""

        # Get messages from the result directly (more reliable than get_state)
        messages = result.get("messages", [])

        # Try to extract from state first (opt-v3-forced-search style)
        # These fields exist in the forced search experiment
        extracted_content = result.get("extracted_content", "")
        retrieved_docs_list = result.get("retrieved_docs", [])
        parent_ids = result.get("parent_ids", [])

        # If we have extracted content from state, use that
        if extracted_content:
            retrieved_docs = extracted_content
            retrieved_chunks = parent_ids
        elif retrieved_docs_list:
            # Use retrieved_docs from state
            docs_text = []
            for doc in retrieved_docs_list:
                if isinstance(doc, dict):
                    content = doc.get("content", str(doc))
                    docs_text.append(content[:1500])
                else:
                    docs_text.append(str(doc)[:1500])
            retrieved_docs = "\n---\n".join(docs_text) if docs_text else "No documents retrieved"
            retrieved_chunks = parent_ids
        else:
            # Fallback: Extract from tool messages (baseline/v1/v2 style)
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
            retrieval_relevance_evaluator,
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
    # The evaluate() returns an ExperimentResults object
    # Each result is an ExperimentResultRow (TypedDict) with keys: run, example, evaluation_results
    local_results = []
    scores = {
        "faithfulness": [],
        "retrieval_relevance": [],
        "semantic_similarity": [],
        "answer_quality": [],
        "latency": []
    }

    # Iterate through the experiment results
    result_count = 0
    for result in results:
        result_count += 1
        # ExperimentResultRow is a TypedDict - access via dict keys, not attributes
        run_input = {}
        run_output = {}
        reference = {}
        evaluations = {}

        # Debug: print result structure for first result
        if result_count == 1:
            print(f"\n📋 Debug: Result type = {type(result)}")
            print(f"   Keys: {list(result.keys()) if hasattr(result, 'keys') else 'N/A'}")

        # Extract from result (TypedDict with keys: run, example, evaluation_results)
        run_obj = result.get("run")
        example_obj = result.get("example")
        eval_results = result.get("evaluation_results")

        # Debug first result
        if result_count == 1:
            print(f"   run_obj type: {type(run_obj)}")
            print(f"   example_obj type: {type(example_obj)}")
            print(f"   eval_results type: {type(eval_results)}")

        if run_obj:
            run_input = run_obj.inputs or {}
            run_output = run_obj.outputs or {}

        if example_obj:
            reference = example_obj.outputs or {}

        # Extract evaluation scores
        if eval_results:
            # eval_results is a dict with 'results' key containing list of EvaluationResult
            eval_list = eval_results.get("results", []) if isinstance(eval_results, dict) else eval_results
            for eval_result in eval_list:
                # EvaluationResult has key and score attributes
                key = getattr(eval_result, 'key', None) or eval_result.get('key') if isinstance(eval_result, dict) else None
                score = getattr(eval_result, 'score', None) if hasattr(eval_result, 'score') else eval_result.get('score') if isinstance(eval_result, dict) else None

                if key and score is not None:
                    evaluations[key] = score
                    if key in scores:
                        scores[key].append(score)

        local_results.append({
            "inputs": run_input,
            "outputs": run_output,
            "reference": reference,
            "evaluations": evaluations
        })

    print(f"\n📊 Processed {result_count} results")

    # Calculate aggregate metrics
    metrics = {}
    for key, vals in scores.items():
        if vals:
            metrics[f"avg_{key}"] = round(sum(vals) / len(vals), 3)
        print(f"   {key}: {len(vals)} scores collected")

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
