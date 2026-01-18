import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from langchain_core.messages import HumanMessage

def run_experiment(graph, experiment_tag, experiment_name, questions, thread_id):
    """Run experiment with metadata tags"""
    
    print(f"\n{'='*70}")
    print(f"🧪 Running: {experiment_name}")
    print(f"🏷️  Tag: {experiment_tag}")
    print('='*70)
    
    # Add experiment tag to config
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": [experiment_tag],  # This tags the run!
        "metadata": {"experiment": experiment_tag}  # Additional metadata
    }
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        start_time = time.time()
        
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )
        
        elapsed = time.time() - start_time
        answer = result["messages"][-1].content
        
        state = graph.get_state(config)
        msg_count = len([m for m in state.values.get("messages", []) 
                        if hasattr(m, 'content')])
        
        results.append({
            "question": question,
            "answer": answer[:200] + "...",
            "time_seconds": round(elapsed, 2),
            "messages_in_state": msg_count
        })
        
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"💬 Messages in state: {msg_count}")
    
    return results

# Test questions
test_questions = [
    "What is JavaScript?",
    "How do I install it?",
    "What are variables?",
    "What is blockchain?",
    "How does mining work?",
]

print("\n🚀 EXPERIMENT COMPARISON - All in 'self-correcting-rag' project\n")

# Run baseline
from experiments.baseline.graph import agent_graph as baseline_graph, EXPERIMENT_TAG as baseline_tag
baseline_results = run_experiment(
    baseline_graph,
    baseline_tag,
    "BASELINE (Full Chunks)",
    test_questions,
    "baseline-compare"
)

# Run optimized
from experiments.opt_v1_minimal_chunks.graph import agent_graph as optimized_graph, EXPERIMENT_TAG as opt_tag
optimized_results = run_experiment(
    optimized_graph,
    opt_tag,
    "OPTIMIZED (Minimal Chunks)",
    test_questions,
    "optimized-compare"
)

# Compare
print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)

baseline_avg = sum(r["time_seconds"] for r in baseline_results) / len(baseline_results)
optimized_avg = sum(r["time_seconds"] for r in optimized_results) / len(optimized_results)

print(f"\n⏱️  Average Response Time:")
print(f"Baseline:  {baseline_avg:.2f}s")
print(f"Optimized: {optimized_avg:.2f}s")
print(f"Speedup:   {((baseline_avg - optimized_avg) / baseline_avg * 100):.1f}% faster")

print("\n" + "="*70)
print("📋 View in LangSmith:")
print("   Project: self-correcting-rag")
print("   Filter by tags: 'baseline' or 'opt-v1-minimal-chunks'")
print("="*70)
