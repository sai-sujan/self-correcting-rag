#!/usr/bin/env python
"""
Compare experiment results from stored JSON files.

Usage:
    python experiments/compare_results.py                     # Compare latest of each experiment
    python experiments/compare_results.py --list              # List all stored results
    python experiments/compare_results.py --experiment baseline  # Show all runs of an experiment
    python experiments/compare_results.py --runs baseline_20260118_120000 opt-v1_20260118_120500
    python experiments/compare_results.py --latest 3          # Compare latest 3 experiments
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import glob
from datetime import datetime
from experiments.experiment_registry import EXPERIMENTS


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def get_all_results():
    """Get all result files"""
    if not os.path.exists(RESULTS_DIR):
        return []

    files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    results = []

    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                data['_file'] = os.path.basename(f)
                results.append(data)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return results


def list_all_results():
    """List all stored results"""
    results = get_all_results()

    if not results:
        print("\n❌ No results found in experiments/results/")
        print("   Run: python experiments/run_experiment.py <experiment_id>")
        return

    print(f"\n📋 Stored Results ({len(results)} total)")
    print("="*80)

    # Group by experiment
    by_experiment = {}
    for r in results:
        exp_id = r.get('experiment_id', 'unknown')
        if exp_id not in by_experiment:
            by_experiment[exp_id] = []
        by_experiment[exp_id].append(r)

    for exp_id, runs in by_experiment.items():
        exp_name = EXPERIMENTS.get(exp_id, {}).get('name', exp_id)
        print(f"\n🧪 {exp_name} ({len(runs)} runs)")
        print("-"*60)

        for run in runs[:5]:  # Show latest 5
            timestamp = run.get('timestamp', 'unknown')
            metrics = run.get('metrics', {})
            overall = metrics.get('avg_overall', 0)
            time_avg = metrics.get('avg_time', 0)

            print(f"   📁 {run['_file']}")
            print(f"      ⭐ Score: {overall:.2f} | ⏱️ {time_avg:.2f}s | 📅 {timestamp[:19]}")

        if len(runs) > 5:
            print(f"   ... and {len(runs) - 5} more")

    print("\n" + "="*80)


def get_latest_per_experiment():
    """Get the latest result for each experiment"""
    results = get_all_results()

    latest = {}
    for r in results:
        exp_id = r.get('experiment_id')
        if exp_id and exp_id not in latest:
            latest[exp_id] = r

    return latest


def compare_results(result_list):
    """Compare multiple results"""
    if len(result_list) < 2:
        print("❌ Need at least 2 results to compare")
        return

    print(f"\n{'='*80}")
    print("📊 EXPERIMENT COMPARISON")
    print("="*80)

    # Header
    headers = ["Metric"]
    for r in result_list:
        exp_name = r.get('experiment_config', {}).get('name', r.get('experiment_id', 'Unknown'))
        # Truncate long names
        if len(exp_name) > 20:
            exp_name = exp_name[:17] + "..."
        headers.append(exp_name)

    col_width = 22
    print("\n" + "".join(h.ljust(col_width) for h in headers))
    print("-" * (col_width * len(headers)))

    # Metrics to compare
    metrics_to_show = [
        ("avg_overall", "⭐ Overall Score", "/5"),
        ("avg_faithfulness", "🎯 Faithfulness", "/5"),
        ("avg_retrieval_f1", "📊 Retrieval F1", ""),
        ("avg_semantic_sim", "🧬 Semantic Sim", ""),
        ("avg_structural", "📏 Structural", ""),
        ("avg_time", "⏱️  Avg Time", "s"),
        ("avg_tokens", "💰 Avg Tokens", ""),
        ("success_rate", "✅ Success Rate", "%"),
    ]

    for metric_key, metric_name, suffix in metrics_to_show:
        row = [metric_name]
        for r in result_list:
            val = r.get('metrics', {}).get(metric_key, 0)
            if metric_key == "avg_tokens":
                row.append(f"{val:.0f}{suffix}")
            else:
                row.append(f"{val:.2f}{suffix}")
        print("".join(s.ljust(col_width) for s in row))

    # Run info
    print("\n" + "-" * (col_width * len(headers)))
    row = ["📅 Run ID"]
    for r in result_list:
        row.append(r.get('run_id', 'unknown'))
    print("".join(s.ljust(col_width) for s in row))

    row = ["📁 File"]
    for r in result_list:
        filename = r.get('_file', 'unknown')
        if len(filename) > 20:
            filename = filename[:17] + "..."
        row.append(filename)
    print("".join(s.ljust(col_width) for s in row))

    print("\n" + "="*80)

    # Find best
    if result_list:
        best = max(result_list, key=lambda x: x.get('metrics', {}).get('avg_overall', 0))
        best_name = best.get('experiment_config', {}).get('name', best.get('experiment_id'))
        best_score = best.get('metrics', {}).get('avg_overall', 0)
        print(f"\n🏆 Best Overall: {best_name} ({best_score:.2f}/5)")

    print("="*80)


def show_experiment_history(experiment_id):
    """Show all runs of a specific experiment"""
    results = get_all_results()
    exp_results = [r for r in results if r.get('experiment_id') == experiment_id]

    if not exp_results:
        print(f"\n❌ No results found for experiment: {experiment_id}")
        return

    exp_name = EXPERIMENTS.get(experiment_id, {}).get('name', experiment_id)

    print(f"\n📈 History for: {exp_name}")
    print("="*80)

    print(f"\n{'Run ID':<20} {'Score':>8} {'Faith':>8} {'F1':>8} {'Time':>8} {'Timestamp':<20}")
    print("-"*80)

    for r in exp_results:
        m = r.get('metrics', {})
        print(f"{r.get('run_id', 'unknown'):<20} "
              f"{m.get('avg_overall', 0):>7.2f} "
              f"{m.get('avg_faithfulness', 0):>7.2f} "
              f"{m.get('avg_retrieval_f1', 0):>7.3f} "
              f"{m.get('avg_time', 0):>7.2f}s "
              f"{r.get('timestamp', 'unknown')[:19]:<20}")

    print("="*80)

    # Show trend
    if len(exp_results) > 1:
        scores = [r.get('metrics', {}).get('avg_overall', 0) for r in exp_results]
        latest = scores[0]
        oldest = scores[-1]
        trend = latest - oldest
        print(f"\n📊 Trend: {'+' if trend > 0 else ''}{trend:.2f} (latest vs oldest)")


def load_specific_runs(run_filenames):
    """Load specific result files"""
    results = []
    for filename in run_filenames:
        # Add .json if not present
        if not filename.endswith('.json'):
            filename += '.json'

        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['_file'] = filename
                results.append(data)
        else:
            print(f"⚠️  File not found: {filename}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--list", action="store_true", help="List all stored results")
    parser.add_argument("--experiment", type=str, help="Show history of specific experiment")
    parser.add_argument("--runs", nargs="+", help="Specific run files to compare")
    parser.add_argument("--latest", type=int, default=0, help="Compare latest N experiments")

    args = parser.parse_args()

    if args.list:
        list_all_results()
    elif args.experiment:
        show_experiment_history(args.experiment)
    elif args.runs:
        results = load_specific_runs(args.runs)
        if results:
            compare_results(results)
    elif args.latest:
        latest = get_latest_per_experiment()
        result_list = list(latest.values())[:args.latest]
        if result_list:
            compare_results(result_list)
        else:
            print("❌ No results found")
    else:
        # Default: compare latest of each experiment
        latest = get_latest_per_experiment()
        if latest:
            compare_results(list(latest.values()))
        else:
            print("❌ No results found. Run experiments first:")
            print("   python experiments/run_experiment.py baseline")
            print("   python experiments/run_experiment.py --all")


if __name__ == "__main__":
    main()
