import sys
sys.path.append('..')

import time
import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from core.config import llm
from langsmith import Client
from langsmith.schemas import FeedbackSourceType

# Initialize LangSmith client
ls_client = Client()

def create_judge_prompt(question, answer, retrieved_docs):
    """Create prompt for LLM judge"""
    return f"""You are an expert evaluator. Score this QA system response on multiple dimensions.

Question: {question}

Retrieved Documents:
{retrieved_docs}

Generated Answer:
{answer}

Evaluate on a scale of 1-5 for each dimension:

1. ACCURACY: Is the answer factually correct?
   1 = Completely wrong
   5 = Perfectly accurate

2. FAITHFULNESS: Is the answer supported by the retrieved documents?
   1 = Contradicts documents or makes things up
   5 = Fully grounded in provided documents

3. COMPLETENESS: Does it fully answer the question?
   1 = Missing critical information
   5 = Comprehensive answer

4. RETRIEVAL_QUALITY: Were the right documents retrieved?
   1 = Irrelevant documents
   5 = Highly relevant documents

Respond ONLY with a JSON object:
{{
  "accuracy": <score>,
  "faithfulness": <score>,
  "completeness": <score>,
  "retrieval_quality": <score>,
  "reasoning": "<brief explanation>"
}}"""


def extract_retrieved_docs(messages):
    """Extract retrieved content from tool messages"""
    docs = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            content = str(msg.content)
            # Limit to first 500 chars per doc to save tokens
            if len(content) > 500:
                content = content[:500] + "..."
            docs.append(content)
    return "\n---\n".join(docs) if docs else "No documents retrieved"


def judge_answer(question, answer, retrieved_docs):
    """Use LLM to judge answer quality"""
    prompt = create_judge_prompt(question, answer, retrieved_docs)
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        
        # Try to parse JSON from response
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        
        scores = json.loads(content)
        return scores
    except Exception as e:
        print(f"  ⚠️  Judge error: {e}")
        return {
            "accuracy": 0,
            "faithfulness": 0,
            "completeness": 0,
            "retrieval_quality": 0,
            "reasoning": f"Error: {str(e)}"
        }


def evaluate_with_quality(graph, experiment_name, test_set, thread_id):
    """Evaluate with quality scoring and send to LangSmith"""
    print(f"\n{'='*70}")
    print(f"🧪 Evaluating: {experiment_name}")
    print('='*70)
    
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": experiment_name,  # Name the run
        "metadata": {"experiment": experiment_name}  # Add metadata
    }
    results = []
    
    for i, test_case in enumerate(test_set, 1):
        question = test_case["question"]
        expected_answer = test_case.get("expected_answer", "")
        
        print(f"\n📝 Test {i}/{len(test_set)}: {question}")
        
        start_time = time.time()
        
        try:
            # Get run_id for this invocation
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                config
            )
            
            elapsed = time.time() - start_time
            answer = result["messages"][-1].content
            
            # Get state and extract docs
            state = graph.get_state(config)
            messages = state.values.get("messages", [])
            retrieved_docs = extract_retrieved_docs(messages)
            
            # Count metrics
            tool_msgs = [m for m in messages if hasattr(m, 'type') and getattr(m, 'type', None) == 'tool']
            total_chars = sum(len(str(m.content)) for m in messages if hasattr(m, 'content'))
            estimated_tokens = total_chars // 4
            
            # Judge the answer
            print(f"  🤔 Judging answer quality...")
            quality_scores = judge_answer(question, answer, retrieved_docs)
            
            # ⭐ NEW: Send scores to LangSmith as feedback
            try:
                # Get the run ID from the last run
                runs = list(ls_client.list_runs(
                    project_name=os.environ.get("LANGCHAIN_PROJECT"),
                    limit=1
                ))
                
                if runs:
                    run_id = runs[0].id
                    
                    # Create feedback for each score
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="accuracy",
                        score=quality_scores.get("accuracy", 0) / 5,  # Normalize to 0-1
                        source_type=FeedbackSourceType.APP
                    )
                    
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="faithfulness",
                        score=quality_scores.get("faithfulness", 0) / 5,
                        source_type=FeedbackSourceType.APP
                    )
                    
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="completeness",
                        score=quality_scores.get("completeness", 0) / 5,
                        source_type=FeedbackSourceType.APP
                    )
                    
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="retrieval_quality",
                        score=quality_scores.get("retrieval_quality", 0) / 5,
                        source_type=FeedbackSourceType.APP
                    )
                    
                    # Overall quality score
                    overall = sum(quality_scores.values()) / (4 * 5)  # Average and normalize
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="overall_quality",
                        score=overall,
                        comment=quality_scores.get("reasoning", ""),
                        source_type=FeedbackSourceType.APP
                    )
                    
                    print(f"  ✅ Scores sent to LangSmith!")
                    
            except Exception as e:
                print(f"  ⚠️  Failed to send to LangSmith: {e}")
            
            results.append({
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "time_seconds": round(elapsed, 2),
                "estimated_tokens": estimated_tokens,
                "tool_calls": len(tool_msgs) // 2,
                "quality_scores": quality_scores,
                "success": True
            })
            
            print(f"  ⏱️  Time: {elapsed:.2f}s")
            print(f"  📊 Tokens: ~{estimated_tokens}")
            print(f"  📈 Quality Scores:")
            print(f"     Accuracy: {quality_scores.get('accuracy', 0)}/5")
            print(f"     Faithfulness: {quality_scores.get('faithfulness', 0)}/5")
            print(f"     Completeness: {quality_scores.get('completeness', 0)}/5")
            print(f"     Retrieval: {quality_scores.get('retrieval_quality', 0)}/5")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "success": False,
                "quality_scores": {
                    "accuracy": 0,
                    "faithfulness": 0,
                    "completeness": 0,
                    "retrieval_quality": 0
                }
            })
    
    return results

def calculate_quality_metrics(results):
    """Calculate aggregate quality metrics"""
    successful = [r for r in results if r.get("success", False)]
    
    if not successful:
        return {}
    
    # Average quality scores
    avg_accuracy = sum(r["quality_scores"]["accuracy"] for r in successful) / len(successful)
    avg_faithfulness = sum(r["quality_scores"]["faithfulness"] for r in successful) / len(successful)
    avg_completeness = sum(r["quality_scores"]["completeness"] for r in successful) / len(successful)
    avg_retrieval = sum(r["quality_scores"]["retrieval_quality"] for r in successful) / len(successful)
    
    # Overall quality score (average of all dimensions)
    overall_quality = (avg_accuracy + avg_faithfulness + avg_completeness + avg_retrieval) / 4
    
    return {
        "total_tests": len(results),
        "successful": len(successful),
        "success_rate": len(successful) / len(results) * 100,
        "avg_time": sum(r["time_seconds"] for r in successful) / len(successful),
        "avg_tokens": sum(r["estimated_tokens"] for r in successful) / len(successful),
        "avg_accuracy": avg_accuracy,
        "avg_faithfulness": avg_faithfulness,
        "avg_completeness": avg_completeness,
        "avg_retrieval_quality": avg_retrieval,
        "overall_quality": overall_quality
    }


# Test set with expected answers for validation
test_set = [
    {
        "question": "What is JavaScript?",
        "expected_answer": "JavaScript is a programming language used for web development"
    },
    {
        "question": "How do I install it?",
        "expected_answer": "Information about installing JavaScript/Node.js"
    },
    {
        "question": "What are JavaScript variables?",
        "expected_answer": "Variables in JavaScript store data values"
    },
    {
        "question": "What is blockchain?",
        "expected_answer": "Blockchain is a distributed ledger technology"
    },
    {
        "question": "How does mining work in blockchain?",
        "expected_answer": "Mining validates transactions through solving cryptographic puzzles"
    }
]

print("\n🚀 QUALITY EVALUATION WITH LLM-AS-A-JUDGE\n")

# Import graphs
from experiments.baseline.graph import agent_graph as baseline_graph
from experiments.opt_v1_minimal_chunks.graph import agent_graph as optimized_graph

# Run evaluations
baseline_results = evaluate_with_quality(
    baseline_graph,
    "BASELINE (Full Chunks)",
    test_set,
    "quality-eval-baseline"
)

optimized_results = evaluate_with_quality(
    optimized_graph,
    "OPTIMIZED (Minimal Chunks)",
    test_set,
    "quality-eval-optimized"
)

# Calculate metrics
baseline_metrics = calculate_quality_metrics(baseline_results)
optimized_metrics = calculate_quality_metrics(optimized_results)

# Print comparison
print("\n" + "="*70)
print("📊 QUALITY EVALUATION RESULTS")
print("="*70)

print("\n🎯 Overall Quality Score (out of 5):")
print(f"  Baseline:  {baseline_metrics['overall_quality']:.2f}")
print(f"  Optimized: {optimized_metrics['overall_quality']:.2f}")

print("\n📈 Detailed Quality Scores:")
print(f"\n  Accuracy:")
print(f"    Baseline:  {baseline_metrics['avg_accuracy']:.2f}/5")
print(f"    Optimized: {optimized_metrics['avg_accuracy']:.2f}/5")

print(f"\n  Faithfulness (grounded in docs):")
print(f"    Baseline:  {baseline_metrics['avg_faithfulness']:.2f}/5")
print(f"    Optimized: {optimized_metrics['avg_faithfulness']:.2f}/5")

print(f"\n  Completeness:")
print(f"    Baseline:  {baseline_metrics['avg_completeness']:.2f}/5")
print(f"    Optimized: {optimized_metrics['avg_completeness']:.2f}/5")

print(f"\n  Retrieval Quality:")
print(f"    Baseline:  {baseline_metrics['avg_retrieval_quality']:.2f}/5")
print(f"    Optimized: {optimized_metrics['avg_retrieval_quality']:.2f}/5")

print("\n⚡ Performance Metrics:")
print(f"  Time: {baseline_metrics['avg_time']:.2f}s vs {optimized_metrics['avg_time']:.2f}s")
print(f"  Tokens: ~{baseline_metrics['avg_tokens']:.0f} vs ~{optimized_metrics['avg_tokens']:.0f}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report = {
    "timestamp": timestamp,
    "baseline": {
        "metrics": baseline_metrics,
        "results": baseline_results
    },
    "optimized": {
        "metrics": optimized_metrics,
        "results": optimized_results
    }
}

filename = f"experiments/quality_evaluation_{timestamp}.json"
with open(filename, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n💾 Results saved to: {filename}")
print("="*70)