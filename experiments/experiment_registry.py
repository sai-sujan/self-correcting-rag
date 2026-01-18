"""
Experiment Registry - Central place to define and manage all experiments
"""

# All registered experiments
EXPERIMENTS = {
    "baseline": {
        "name": "Baseline (Full Chunks)",
        "description": "Original implementation with full chunk retrieval",
        "module": "experiments.baseline.graph",
        "graph_var": "agent_graph",
        "tag": "baseline",
        "version": "1.0"
    },
    "opt-v1-minimal-chunks": {
        "name": "Optimized V1 (Minimal Chunks)",
        "description": "Returns only parent IDs + 100 char snippets to save tokens",
        "module": "experiments.opt_v1_minimal_chunks.graph",
        "graph_var": "agent_graph",
        "tag": "opt-v1-minimal-chunks",
        "version": "1.0"
    },
    "opt-v2-strict-search": {
        "name": "Optimized V2 (Strict Search)",
        "description": "Enforces mandatory search before answering, no training knowledge",
        "module": "experiments.opt_v2_strict_search.graph",
        "graph_var": "agent_graph",
        "tag": "opt-v2-strict-search",
        "version": "1.0"
    }
}

# Test sets - can add more test sets here
TEST_SETS = {
    "default": [
        {
            "test_id": "test-js-basics",
            "category": "javascript",
            "question": "What is JavaScript?",
            "reference_answer": "JavaScript is a dynamic programming language commonly used for web development",
            "expected_chunks": ["javascript_tutorial_parent_6"]
        },
        {
            "test_id": "test-js-install",
            "category": "javascript",
            "question": "How do I install it?",
            "reference_answer": "Install Node.js to use JavaScript",
            "expected_chunks": ["javascript_tutorial_parent_15"]
        },
        {
            "test_id": "test-js-variables",
            "category": "javascript",
            "question": "What are JavaScript variables?",
            "reference_answer": "Variables in JavaScript store data values using var, let, or const",
            "expected_chunks": ["javascript_tutorial_parent_8", "javascript_tutorial_parent_9"]
        },
        {
            "test_id": "test-blockchain-basics",
            "category": "blockchain",
            "question": "What is blockchain?",
            "reference_answer": "Blockchain is a distributed ledger technology for secure transactions",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_2"]
        },
        {
            "test_id": "test-blockchain-mining",
            "category": "blockchain",
            "question": "How does mining work?",
            "reference_answer": "Mining validates blockchain transactions through solving cryptographic puzzles",
            "expected_chunks": ["Blockchain_For_Beginners_A_EUBOF_Guide_parent_10"]
        }
    ]
}


def get_experiment(experiment_id):
    """Get experiment config by ID"""
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_id}. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[experiment_id]


def load_experiment_graph(experiment_id):
    """Dynamically load experiment graph"""
    import importlib
    config = get_experiment(experiment_id)
    module = importlib.import_module(config["module"])
    return getattr(module, config["graph_var"])


def list_experiments():
    """List all registered experiments"""
    print("\n📋 Registered Experiments:")
    print("="*60)
    for exp_id, config in EXPERIMENTS.items():
        print(f"\n  🧪 {exp_id}")
        print(f"     Name: {config['name']}")
        print(f"     Description: {config['description']}")
        print(f"     Tag: {config['tag']}")
    print("\n" + "="*60)


def add_experiment(experiment_id, name, description, module, graph_var="agent_graph", tag=None, version="1.0"):
    """Add a new experiment to registry (runtime only, won't persist)"""
    EXPERIMENTS[experiment_id] = {
        "name": name,
        "description": description,
        "module": module,
        "graph_var": graph_var,
        "tag": tag or experiment_id,
        "version": version
    }
    print(f"✅ Added experiment: {experiment_id}")


if __name__ == "__main__":
    list_experiments()
