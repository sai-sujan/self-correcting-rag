import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import llm, dense_embeddings
# sys.path.append('..')

import time
import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from core.config import llm, dense_embeddings
from langsmith import Client
from langsmith.schemas import FeedbackSourceType
import numpy as np

# Initialize LangSmith
ls_client = Client()


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def judge_faithfulness(answer, retrieved_docs):
    """
    LLM judges if answer is grounded in retrieved docs.
    This is RELIABLE - doesn't need external knowledge.
    """
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
    """
    Measure if we retrieved the RIGHT documents.
    No LLM knowledge needed - pure overlap metric.
    """
    if not expected_chunk_ids:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "note": "No ground truth provided"}
    
    retrieved_set = set(retrieved_chunk_ids)
    expected_set = set(expected_chunk_ids)
    
    if not retrieved_set:
        return {"precision": 0, "recall": 0, "f1": 0}
    
    intersection = retrieved_set & expected_set
    
    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0
    recall = len(intersection) / len(expected_set) if expected_set else 0
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
    """
    Compare answer to reference using embeddings.
    Uses semantic similarity, not LLM knowledge.
    """
    if not reference_answer:
        return 1.0, "No reference answer provided"
    
    try:
        answer_embedding = dense_embeddings.embed_query(answer)
        reference_embedding = dense_embeddings.embed_query(reference_answer)
        
        similarity = cosine_similarity(
            np.array(answer_embedding),
            np.array(reference_embedding)
        )
        
        return round(float(similarity), 3), ""
    except Exception as e:
        return 0, f"Error: {str(e)}"


def structural_quality_checks(answer):
    """
    Heuristic checks that don't need LLM knowledge.
    """
    checks = {
        "has_content": len(answer) > 50,
        "sufficient_length": len(answer) > 100,
        "has_details": any(word in answer.lower() for word in ["specifically", "example", "such as", "including"]),
        "not_too_short": len(answer.split()) > 20,
        "not_error": "error" not in answer.lower() and "don't have" not in answer.lower()
    }
    
    score = sum(checks.values()) / len(checks)
    
    return {
        "structural_score": round(score, 3),
        "checks": checks
    }


def extract_chunk_ids(messages):
    """Extract parent_ids from tool messages"""
    chunk_ids = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            content = str(msg.content)
            # Try to extract parent_ids from the content
            if "parent_id" in content:
                import re
                ids = re.findall(r"'parent_id': '([^']+)'", content)
                chunk_ids.extend(ids)
    return list(set(chunk_ids))


def extract_retrieved_docs(messages):
    """Extract retrieved content for faithfulness check"""
    docs = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            content = str(msg.content)
            if len(content) > 500:
                content = content[:500] + "..."
            docs.append(content)
    return "\n---\n".join(docs) if docs else "No documents retrieved"


def hybrid_evaluate(graph, experiment_name, test_set, thread_id):
    """
    Hybrid evaluation combining multiple methods:
    1. Faithfulness (LLM judge)
    2. Retrieval accuracy (metrics)
    3. Semantic similarity (embeddings)
    4. Structural checks (heuristics)
    """
    print(f"\n{'='*70}")
    print(f"🧪 Hybrid Evaluation: {experiment_name}")
    print('='*70)
    
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"experiment": experiment_name}
    }
    
    results = []
    
    for i, test_case in enumerate(test_set, 1):
        question = test_case["question"]
        reference_answer = test_case.get("reference_answer", "")
        expected_chunks = test_case.get("expected_chunks", [])
        
        print(f"\n📝 Test {i}/{len(test_set)}: {question}")
        
        start_time = time.time()
        
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                config
            )
            
            elapsed = time.time() - start_time
            answer = result["messages"][-1].content
            
            # Get state
            state = graph.get_state(config)
            messages = state.values.get("messages", [])
            
            # Extract info
            retrieved_docs = extract_retrieved_docs(messages)
            retrieved_chunk_ids = extract_chunk_ids(messages)
            
            # === HYBRID EVALUATION ===
            
            # 1. Faithfulness (LLM judge - reliable)
            print(f"  🔍 Checking faithfulness...")
            faithfulness, hallucinations = judge_faithfulness(answer, retrieved_docs)
            
            # 2. Retrieval accuracy (no LLM knowledge needed)
            print(f"  📊 Measuring retrieval accuracy...")
            retrieval_metrics = evaluate_retrieval_accuracy(retrieved_chunk_ids, expected_chunks)
            
            # 3. Semantic similarity to reference (embeddings)
            print(f"  🧬 Computing semantic similarity...")
            semantic_sim, sim_error = semantic_similarity_score(answer, reference_answer)
            
            # 4. Structural quality (heuristics)
            print(f"  📏 Checking structural quality...")
            structural = structural_quality_checks(answer)
            
            # Overall score (weighted average)
            overall_score = (
                faithfulness * 0.4 +  # Most important
                retrieval_metrics["f1"] * 5 * 0.3 +  # Convert to 0-5 scale
                semantic_sim * 5 * 0.2 +  # Convert to 0-5 scale
                structural["structural_score"] * 5 * 0.1  # Convert to 0-5 scale
            )
            
            # Performance metrics
            tool_msgs = [m for m in messages if hasattr(m, 'type') and getattr(m, 'type', None) == 'tool']
            total_chars = sum(len(str(m.content)) for m in messages if hasattr(m, 'content'))
            estimated_tokens = total_chars // 4
            
            result_data = {
                "question": question,
                "answer": answer,
                "reference_answer": reference_answer,
                "time_seconds": round(elapsed, 2),
                "estimated_tokens": estimated_tokens,
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
            
            # Print scores
            print(f"\n  📊 Results:")
            print(f"     Faithfulness: {faithfulness}/5")
            print(f"     Retrieval F1: {retrieval_metrics['f1']}")
            print(f"     Semantic Similarity: {semantic_sim}")
            print(f"     Structural Quality: {structural['structural_score']}")
            print(f"     ⭐ Overall: {overall_score:.2f}/5")
            print(f"     ⏱️  Time: {elapsed:.2f}s")
            print(f"     💰 Tokens: ~{estimated_tokens}")
            
            # Send to LangSmith
            try:
                runs = list(ls_client.list_runs(
                    project_name="self-correcting-rag",
                    limit=1
                ))
                
                if runs:
                    run_id = runs[0].id
                    
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="faithfulness",
                        score=faithfulness / 5,
                    )
                    
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="retrieval_f1",
                        score=retrieval_metrics["f1"],
                    )
                    
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="semantic_similarity",
                        score=semantic_sim,
                    )
                    
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="overall_quality",
                        score=overall_score / 5,
                        comment=f"Hybrid: F={faithfulness}, R={retrieval_metrics['f1']:.2f}, S={semantic_sim:.2f}",
                    )
                    
                    print(f"  ✅ Sent to LangSmith!")
            except Exception as e:
                print(f"  ⚠️  LangSmith error: {e}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "success": False,
                "overall_score": 0
            })
    
    return results


def calculate_aggregate_metrics(results):
    """Calculate averages across all tests"""
    successful = [r for r in results if r.get("success", False)]
    
    if not successful:
        return {}
    
    return {
        "total_tests": len(results),
        "successful": len(successful),
        "success_rate": len(successful) / len(results) * 100,
        "avg_time": sum(r["time_seconds"] for r in successful) / len(successful),
        "avg_tokens": sum(r["estimated_tokens"] for r in successful) / len(successful),
        "avg_faithfulness": sum(r["faithfulness"] for r in successful) / len(successful),
        "avg_retrieval_f1": sum(r["retrieval_metrics"]["f1"] for r in successful) / len(successful),
        "avg_semantic_sim": sum(r["semantic_similarity"] for r in successful) / len(successful),
        "avg_structural": sum(r["structural_quality"]["structural_score"] for r in successful) / len(successful),
        "avg_overall": sum(r["overall_score"] for r in successful) / len(successful)
    }


# Test set with reference answers and expected chunks
test_set = [
    {
        "question": "What is JavaScript?",
        "reference_answer": "JavaScript is a dynamic programming language commonly used for web development",
        "expected_chunks": ["javascript_tutorial_parent_6"]
    },
    {
        "question": "How do I install it?",
        "reference_answer": "Install Node.js to use JavaScript",
        "expected_chunks": ["javascript_tutorial_parent_15"]
    },
    {
        "question": "What are JavaScript variables?",
        "reference_answer": "Variables in JavaScript store data values using var, let, or const",
        "expected_chunks": ["javascript_tutorial_parent_8", "javascript_tutorial_parent_9"]
    },
    {
        "question": "What is blockchain?",
        "reference_answer": "Blockchain is a distributed ledger technology for secure transactions",
        "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_2"]
    },
    {
        "question": "How does mining work?",
        "reference_answer": "Mining validates blockchain transactions through solving cryptographic puzzles",
        "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_10"]
    }
]

print("\n🚀 HYBRID EVALUATION SYSTEM\n")
print("Combining:")
print("  1. ✅ Faithfulness (LLM judge)")
print("  2. ✅ Retrieval accuracy (metrics)")
print("  3. ✅ Semantic similarity (embeddings)")
print("  4. ✅ Structural quality (heuristics)\n")

# Import graphs
from experiments.baseline.graph import agent_graph as baseline_graph
from experiments.opt_v1_minimal_chunks.graph import agent_graph as optimized_graph

# Evaluate baseline
baseline_results = hybrid_evaluate(
    baseline_graph,
    "baseline",
    test_set,
    "hybrid-eval-baseline"
)

# Evaluate optimized
optimized_results = hybrid_evaluate(
    optimized_graph,
    "opt-v1-minimal-chunks",
    test_set,
    "hybrid-eval-optimized"
)

# Calculate metrics
baseline_metrics = calculate_aggregate_metrics(baseline_results)
optimized_metrics = calculate_aggregate_metrics(optimized_results)

# Print comparison
print("\n" + "="*70)
print("📊 HYBRID EVALUATION RESULTS")
print("="*70)

print("\n⭐ Overall Quality Score (out of 5):")
print(f"  Baseline:  {baseline_metrics['avg_overall']:.2f}")
print(f"  Optimized: {optimized_metrics['avg_overall']:.2f}")
diff = optimized_metrics['avg_overall'] - baseline_metrics['avg_overall']
print(f"  Difference: {'+' if diff > 0 else ''}{diff:.2f}")

print("\n📈 Detailed Metrics:")
print(f"\n  Faithfulness (grounded in docs):")
print(f"    Baseline:  {baseline_metrics['avg_faithfulness']:.2f}/5")
print(f"    Optimized: {optimized_metrics['avg_faithfulness']:.2f}/5")

print(f"\n  Retrieval F1 (right docs retrieved):")
print(f"    Baseline:  {baseline_metrics['avg_retrieval_f1']:.3f}")
print(f"    Optimized: {optimized_metrics['avg_retrieval_f1']:.3f}")

print(f"\n  Semantic Similarity (vs reference):")
print(f"    Baseline:  {baseline_metrics['avg_semantic_sim']:.3f}")
print(f"    Optimized: {optimized_metrics['avg_semantic_sim']:.3f}")

print(f"\n  Structural Quality:")
print(f"    Baseline:  {baseline_metrics['avg_structural']:.3f}")
print(f"    Optimized: {optimized_metrics['avg_structural']:.3f}")

print("\n⚡ Performance:")
time_improvement = ((baseline_metrics['avg_time'] - optimized_metrics['avg_time']) / baseline_metrics['avg_time'] * 100)
token_savings = ((baseline_metrics['avg_tokens'] - optimized_metrics['avg_tokens']) / baseline_metrics['avg_tokens'] * 100)
print(f"  Time: {baseline_metrics['avg_time']:.2f}s → {optimized_metrics['avg_time']:.2f}s ({time_improvement:.1f}% faster)")
print(f"  Tokens: {baseline_metrics['avg_tokens']:.0f} → {optimized_metrics['avg_tokens']:.0f} ({token_savings:.1f}% savings)")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report = {
    "timestamp": timestamp,
    "evaluation_type": "hybrid",
    "baseline": {
        "metrics": baseline_metrics,
        "results": baseline_results
    },
    "optimized": {
        "metrics": optimized_metrics,
        "results": optimized_results
    }
}

filename = f"experiments/hybrid_evaluation_{timestamp}.json"
with open(filename, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n💾 Results saved to: {filename}")
print("\n📋 View in LangSmith:")
print("   Project: self-correcting-rag")
print("   Feedback scores attached to each trace")
print("="*70)