#!/usr/bin/env python3
"""
Unified Experiment Runner - Run any experiment with a simple command.

This replaces the need for separate experiment folders with 600+ lines each.
Now experiments are just configurations, and this runner builds the graph dynamically.

Usage:
    # List all available experiments
    python run_experiment.py --list

    # Run a specific experiment interactively
    python run_experiment.py --experiment opt-v7-D

    # Run with a single question
    python run_experiment.py --experiment opt-v7-D --question "What is blockchain?"

    # Run LangSmith evaluation
    python run_experiment.py --experiment opt-v7-D --evaluate --dataset "rag-evaluation-tests"

    # Compare multiple experiments
    python run_experiment.py --compare opt-v6 opt-v7-D --question "What is JavaScript?"
"""
import sys
import os
import argparse
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage
from core.graph_builder import build_graph, build_graph_by_name
from core.experiment_config import (
    get_experiment,
    list_experiments,
    print_experiments,
    ALL_EXPERIMENTS,
    DEFAULT_CONFIG
)


def run_interactive(experiment_name: str = None):
    """Run an experiment in interactive mode."""
    if experiment_name:
        config = get_experiment(experiment_name)
    else:
        config = DEFAULT_CONFIG

    print(f"\n{'='*60}")
    print(f"INTERACTIVE MODE: {config.name}")
    print(f"{'='*60}")
    print(f"Description: {config.description}")
    print(f"Type 'quit' or 'exit' to stop.\n")

    graph = build_graph(config)
    thread_id = f"interactive-{int(time.time())}"

    while True:
        try:
            question = input("\nYou: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not question:
                continue

            start = time.time()
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                {"configurable": {"thread_id": thread_id}}
            )
            latency = time.time() - start

            # Get answer
            answer = result["messages"][-1].content if result.get("messages") else "No response"
            print(f"\nAssistant: {answer}")
            print(f"\n[Latency: {latency:.2f}s]")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def run_single_question(experiment_name: str, question: str):
    """Run a single question through an experiment."""
    config = get_experiment(experiment_name)
    graph = build_graph(config)

    print(f"\n{'='*60}")
    print(f"Experiment: {config.name}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    start = time.time()
    thread_id = f"single-{int(time.time())}"

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}}
    )

    latency = time.time() - start
    answer = result["messages"][-1].content if result.get("messages") else "No response"

    print(f"Answer: {answer}")
    print(f"\n[Latency: {latency:.2f}s]")

    # Show debug info
    if result.get("search_queries"):
        print(f"[Search Queries: {result['search_queries']}]")
    if result.get("judgment"):
        print(f"[Judgment: {result['judgment'].get('overall_verdict', 'N/A')}]")

    return {"answer": answer, "latency": latency, "result": result}


def compare_experiments(experiment_names: list, question: str):
    """Compare multiple experiments on the same question."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {question}")
    print(f"{'='*70}\n")

    results = []
    for name in experiment_names:
        try:
            config = get_experiment(name)
            graph = build_graph(config)

            start = time.time()
            thread_id = f"compare-{name}-{int(time.time())}"

            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                {"configurable": {"thread_id": thread_id}}
            )

            latency = time.time() - start
            answer = result["messages"][-1].content if result.get("messages") else "No response"
            verdict = result.get("judgment", {}).get("overall_verdict", "N/A")

            results.append({
                "name": name,
                "answer": answer,
                "latency": latency,
                "verdict": verdict
            })

            print(f"\n--- {name} ---")
            print(f"Temps: llm={config.llm_temp}, small={config.llm_small_temp}")
            print(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"Latency: {latency:.2f}s | Verdict: {verdict}")

        except Exception as e:
            print(f"\n--- {name} ---")
            print(f"Error: {e}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<20} {'Latency':<10} {'Verdict':<15}")
    print("-"*50)
    for r in results:
        print(f"{r['name']:<20} {r['latency']:.2f}s{'':<5} {r['verdict']:<15}")


def run_langsmith_evaluation(experiment_name: str, dataset_name: str):
    """Run LangSmith evaluation on an experiment."""
    try:
        from langsmith import Client, evaluate
    except ImportError:
        print("Error: langsmith not installed. Run: pip install langsmith")
        return

    config = get_experiment(experiment_name)
    graph = build_graph(config)

    print(f"\n{'='*60}")
    print(f"LANGSMITH EVALUATION")
    print(f"Experiment: {config.name}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}\n")

    def target(inputs: dict) -> dict:
        question = inputs.get("question", "")
        thread_id = f"eval-{int(time.time())}"

        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": thread_id}}
        )

        answer = result["messages"][-1].content if result.get("messages") else ""
        extracted = result.get("extracted_content", "")

        return {
            "answer": answer,
            "retrieved_docs": extracted
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_prefix = f"{config.name}-{timestamp}"

    evaluate(
        target,
        data=dataset_name,
        experiment_prefix=experiment_prefix,
        metadata={
            "experiment": config.name,
            "llm_temp": config.llm_temp,
            "llm_small_temp": config.llm_small_temp,
            "search_mode": config.search_mode.value,
            "extraction_mode": config.extraction_mode.value,
            "retry_mode": config.retry_mode.value
        },
        max_concurrency=1
    )

    print(f"\nEvaluation complete! Check LangSmith dashboard.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified RAG Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --list
  python run_experiment.py --experiment opt-v7-D
  python run_experiment.py --experiment opt-v7-D --question "What is blockchain?"
  python run_experiment.py --compare opt-v6 opt-v7-D --question "What is JavaScript?"
        """
    )

    parser.add_argument("--list", "-l", action="store_true",
                        help="List all available experiments")
    parser.add_argument("--experiment", "-e", type=str,
                        help="Experiment name to run")
    parser.add_argument("--question", "-q", type=str,
                        help="Single question to ask")
    parser.add_argument("--compare", "-c", nargs="+",
                        help="Compare multiple experiments")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run LangSmith evaluation")
    parser.add_argument("--dataset", "-d", type=str, default="rag-evaluation-tests",
                        help="LangSmith dataset name for evaluation")

    args = parser.parse_args()

    # List experiments
    if args.list:
        print_experiments()
        return

    # Compare experiments
    if args.compare:
        if not args.question:
            print("Error: --compare requires --question")
            return
        compare_experiments(args.compare, args.question)
        return

    # LangSmith evaluation
    if args.evaluate:
        if not args.experiment:
            print("Error: --evaluate requires --experiment")
            return
        run_langsmith_evaluation(args.experiment, args.dataset)
        return

    # Single question
    if args.question:
        exp_name = args.experiment or "opt-v7-D"
        run_single_question(exp_name, args.question)
        return

    # Interactive mode
    run_interactive(args.experiment)


if __name__ == "__main__":
    main()
