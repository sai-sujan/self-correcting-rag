#!/usr/bin/env python
"""
Run a single experiment and save results separately.

Usage:
    python experiments/run_experiment.py baseline
    python experiments/run_experiment.py opt-v1-minimal-chunks
    python experiments/run_experiment.py opt-v2-strict-search
    python experiments/run_experiment.py --list  # List all experiments
    python experiments/run_experiment.py --all   # Run all experiments
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np

from core.config import llm, dense_embeddings
from experiments.experiment_registry import (
    EXPERIMENTS, TEST_SETS, get_experiment, load_experiment_graph, list_experiments
)

# LangSmith
from langsmith import Client
ls_client = Client()


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def judge_faithfulness(answer, retrieved_docs):
    prompt = f"""You are evaluating if an AI answer is grounded in provided documents.

Retrieved Documents:
{retrieved_docs}

Generated Answer:
{answer}

Rate FAITHFULNESS (1-5):
1 = Makes claims NOT in documents (hallucination)
3 = Partially supported
5 = Every claim directly from documents

Respond ONLY with JSON:
{{"faithfulness": <score>, "hallucination_examples": "<any unsupported claims>"}}"""

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        result = json.loads(content)
        return result.get("faithfulness", 0), result.get("hallucination_examples", "")
    except Exception as e:
        return 0, f"Error: {str(e)}"


def evaluate_retrieval_accuracy(retrieved_chunk_ids, expected_chunk_ids):
    if not expected_chunk_ids:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    retrieved_set = set(retrieved_chunk_ids)
    expected_set = set(expected_chunk_ids)

    if not retrieved_set:
        return {"precision": 0, "recall": 0, "f1": 0}

    intersection = retrieved_set & expected_set
    precision = len(intersection) / len(retrieved_set)
    recall = len(intersection) / len(expected_set)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "retrieved": list(retrieved_set),
        "expected": list(expected_set),
        "correct": list(intersection)
    }


def semantic_similarity_score(answer, reference_answer):
    if not reference_answer:
        return 1.0, ""
    try:
        answer_embedding = dense_embeddings.embed_query(answer)
        reference_embedding = dense_embeddings.embed_query(reference_answer)
        similarity = cosine_similarity(np.array(answer_embedding), np.array(reference_embedding))
        return round(float(similarity), 3), ""
    except Exception as e:
        return 0, f"Error: {str(e)}"


def structural_quality_checks(answer):
    checks = {
        "has_content": len(answer) > 50,
        "sufficient_length": len(answer) > 100,
        "has_details": any(word in answer.lower() for word in ["specifically", "example", "such as", "including"]),
        "not_too_short": len(answer.split()) > 20,
        "not_error": "error" not in answer.lower() and "don't have" not in answer.lower()
    }
    return {"structural_score": round(sum(checks.values()) / len(checks), 3), "checks": checks}


def extract_chunk_ids(messages):
    import re
    chunk_ids = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            ids = re.findall(r"'parent_id': '([^']+)'", str(msg.content))
            chunk_ids.extend(ids)
    return list(set(chunk_ids))


def extract_retrieved_docs(messages):
    docs = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            content = str(msg.content)[:500] + "..."
            docs.append(content)
    return "\n---\n".join(docs) if docs else "No documents retrieved"


def run_single_experiment(experiment_id, test_set_name="default"):
    """Run a single experiment and save results"""

    config = get_experiment(experiment_id)
    test_set = TEST_SETS.get(test_set_name, TEST_SETS["default"])

    print(f"\n{'='*70}")
    print(f"🧪 Running Experiment: {config['name']}")
    print(f"📝 Description: {config['description']}")
    print(f"🏷️  Tag: {config['tag']}")
    print(f"📊 Test Set: {test_set_name} ({len(test_set)} tests)")
    print('='*70)

    # Load graph
    graph = load_experiment_graph(experiment_id)

    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []

    for i, test_case in enumerate(test_set, 1):
        question = test_case["question"]
        reference_answer = test_case.get("reference_answer", "")
        expected_chunks = test_case.get("expected_chunks", [])
        test_id = test_case.get("test_id", f"test-{i}")
        category = test_case.get("category", "general")

        # Config with tags for LangSmith
        invoke_config = {
            "configurable": {"thread_id": f"{experiment_id}-{run_id}-{test_id}"},
            "tags": [experiment_id, category, test_id, f"run-{run_id}"],
            "metadata": {
                "experiment": experiment_id,
                "run_id": run_id,
                "test_id": test_id,
                "category": category
            }
        }

        print(f"\n📝 Test {i}/{len(test_set)}: {question}")

        start_time = time.time()

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                invoke_config
            )

            elapsed = time.time() - start_time
            answer = result["messages"][-1].content

            state = graph.get_state(invoke_config)
            messages = state.values.get("messages", [])

            retrieved_docs = extract_retrieved_docs(messages)
            retrieved_chunk_ids = extract_chunk_ids(messages)

            # Evaluate
            print(f"  🔍 Evaluating...")
            faithfulness, hallucinations = judge_faithfulness(answer, retrieved_docs)
            retrieval_metrics = evaluate_retrieval_accuracy(retrieved_chunk_ids, expected_chunks)
            semantic_sim, _ = semantic_similarity_score(answer, reference_answer)
            structural = structural_quality_checks(answer)

            # Calculate overall score
            overall_score = (
                faithfulness * 0.4 +
                retrieval_metrics["f1"] * 5 * 0.3 +
                semantic_sim * 5 * 0.2 +
                structural["structural_score"] * 5 * 0.1
            )

            tool_msgs = [m for m in messages if hasattr(m, 'type') and getattr(m, 'type', None) == 'tool']
            total_chars = sum(len(str(m.content)) for m in messages if hasattr(m, 'content'))

            result_data = {
                "test_id": test_id,
                "category": category,
                "question": question,
                "answer": answer,
                "reference_answer": reference_answer,
                "time_seconds": round(elapsed, 2),
                "estimated_tokens": total_chars // 4,
                "tool_calls": len(tool_msgs) // 2,
                "faithfulness": faithfulness,
                "hallucinations": hallucinations,
                "retrieval_metrics": retrieval_metrics,
                "semantic_similarity": semantic_sim,
                "structural_quality": structural,
                "overall_score": round(overall_score, 2),
                "success": True
            }

            results.append(result_data)

            print(f"  ⭐ Score: {overall_score:.2f}/5 | ⏱️ {elapsed:.2f}s | 🎯 F1: {retrieval_metrics['f1']}")

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                "test_id": test_id,
                "question": question,
                "error": str(e),
                "success": False,
                "overall_score": 0
            })

    # Calculate aggregate metrics
    successful = [r for r in results if r.get("success", False)]
    metrics = {}
    if successful:
        metrics = {
            "total_tests": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results) * 100,
            "avg_time": round(sum(r["time_seconds"] for r in successful) / len(successful), 2),
            "avg_tokens": round(sum(r["estimated_tokens"] for r in successful) / len(successful), 0),
            "avg_faithfulness": round(sum(r["faithfulness"] for r in successful) / len(successful), 2),
            "avg_retrieval_f1": round(sum(r["retrieval_metrics"]["f1"] for r in successful) / len(successful), 3),
            "avg_semantic_sim": round(sum(r["semantic_similarity"] for r in successful) / len(successful), 3),
            "avg_structural": round(sum(r["structural_quality"]["structural_score"] for r in successful) / len(successful), 3),
            "avg_overall": round(sum(r["overall_score"] for r in successful) / len(successful), 2)
        }

    # Save results
    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "experiment_config": config,
        "test_set": test_set_name,
        "metrics": metrics,
        "results": results
    }

    # Save to results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(results_dir, f"{experiment_id}_{run_id}.json")
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"📊 Results Summary for {config['name']}")
    print('='*70)
    print(f"  ⭐ Overall Score: {metrics.get('avg_overall', 0):.2f}/5")
    print(f"  🎯 Faithfulness: {metrics.get('avg_faithfulness', 0):.2f}/5")
    print(f"  📊 Retrieval F1: {metrics.get('avg_retrieval_f1', 0):.3f}")
    print(f"  🧬 Semantic Sim: {metrics.get('avg_semantic_sim', 0):.3f}")
    print(f"  ⏱️  Avg Time: {metrics.get('avg_time', 0):.2f}s")
    print(f"  💰 Avg Tokens: {metrics.get('avg_tokens', 0):.0f}")
    print(f"\n💾 Saved to: {filename}")
    print('='*70)

    return report


def run_all_experiments(test_set_name="default"):
    """Run all registered experiments"""
    print("\n🚀 Running ALL Experiments\n")

    reports = {}
    for exp_id in EXPERIMENTS.keys():
        reports[exp_id] = run_single_experiment(exp_id, test_set_name)

    return reports


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("experiment", nargs="?", help="Experiment ID to run")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--test-set", default="default", help="Test set to use")

    args = parser.parse_args()

    if args.list:
        list_experiments()
    elif args.all:
        run_all_experiments(args.test_set)
    elif args.experiment:
        run_single_experiment(args.experiment, args.test_set)
    else:
        parser.print_help()
        print("\n📋 Available experiments:")
        for exp_id in EXPERIMENTS.keys():
            print(f"  - {exp_id}")


if __name__ == "__main__":
    main()
