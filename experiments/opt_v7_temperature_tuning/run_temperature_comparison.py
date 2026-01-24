#!/usr/bin/env python3
"""
Run temperature comparison experiments using LangSmith.

This script runs the temperature comparison experiment (opt-v7)
by executing parallel experiments in LangSmith (Config A-E).

Usage:
    python experiments/opt_v7_temperature_tuning/run_temperature_comparison.py
    python experiments/opt_v7_temperature_tuning/run_temperature_comparison.py --configs A B
"""
import sys
import os
import argparse
from datetime import datetime
from langsmith import Client, evaluate
from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.opt_v7_temperature_tuning.graph import create_graph_with_config, TEMPERATURE_CONFIGS
from experiments.run_langsmith_experiment import (
    faithfulness_evaluator,
    retrieval_relevance_evaluator,
    semantic_similarity_evaluator,
    answer_quality_evaluator,
    latency_evaluator
)

# Dataset name (must match what was created in setup_langsmith_dataset.py)
DATASET_NAME = "rag-evaluation-tests-to-check-temperatures"

def create_target_function(config_key: str):
    """Create a target function for a specific config."""
    graph = create_graph_with_config(config_key)
    
    def target(inputs: dict) -> dict:
        question = inputs.get("question", "")
        thread_id = f"eval-{config_key}-{datetime.now().timestamp()}"
        config = {"configurable": {"thread_id": thread_id}}
        
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )
        
        answer = result["messages"][-1].content if result.get("messages") else ""
        extracted = result.get("extracted_content", "")
        
        # Helper to format retrieved docs
        retrieved_docs_list = result.get("retrieved_docs", [])
        if extracted:
            retrieved_str = extracted
        elif retrieved_docs_list:
            docs_text = []
            for doc in retrieved_docs_list:
                if isinstance(doc, dict):
                    content = doc.get("content", str(doc))
                    docs_text.append(content[:1500])
                else:
                    docs_text.append(str(doc)[:1500])
            retrieved_str = "\n---\n".join(docs_text)
        else:
            retrieved_str = "No documents retrieved"

        return {
            "answer": answer,
            "retrieved_docs": retrieved_str
        }
        
    return target

def main():
    parser = argparse.ArgumentParser(description="Run LangSmith temperature comparison")
    parser.add_argument("--configs", nargs="+", default=list(TEMPERATURE_CONFIGS.keys()),
                        help="Config keys to test (default: all)")
    args = parser.parse_args()
    
    client = Client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🚀 Starting Temperature Comparison Experiment")
    print(f"📅 Timestamp: {timestamp}")
    print(f"📊 Dataset: {DATASET_NAME}")
    print(f"🧪 Configs: {args.configs}")
    print("="*60)
    
    for config_key in args.configs:
        if config_key not in TEMPERATURE_CONFIGS:
            print(f"⚠️  Skipping unknown config: {config_key}")
            continue
            
        config = TEMPERATURE_CONFIGS[config_key]
        experiment_prefix = f"opt-v7-temp-{config_key}"
        description = f"Config {config_key}: llm={config['llm_temp']}, small={config['llm_small_temp']} ({config['description']})"
        
        print(f"\n▶️  Running Config {config_key}...")
        print(f"   {description}")
        
        try:
            target = create_target_function(config_key)
            
            evaluate(
                target,
                data=DATASET_NAME,
                evaluators=[
                    faithfulness_evaluator,
                    retrieval_relevance_evaluator,
                    semantic_similarity_evaluator,
                    answer_quality_evaluator,
                    latency_evaluator
                ],
                experiment_prefix=experiment_prefix,
                metadata={
                    "config_key": config_key,
                    "llm_temp": config["llm_temp"],
                    "llm_small_temp": config["llm_small_temp"],
                    "description": config["description"],
                    "timestamp": timestamp
                },
                max_concurrency=1
            )
            print(f"✅ Completed Config {config_key}")
            
        except Exception as e:
            print(f"❌ Error running Config {config_key}: {e}")

    print("\n" + "="*60)
    print("🎉 All experiments completed!")
    print(f"View results at: https://smith.langchain.com/datasets")

if __name__ == "__main__":
    main()
