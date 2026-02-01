"""
Core module - Shared components for all RAG experiments.

This module contains:
- config.py: LLM, embeddings, vector DB configuration
- experiment_config.py: Experiment definitions and registry
- graph_builder.py: Dynamic graph construction
- llm_manager.py: Runtime LLM temperature management
- shared/: Reusable nodes, tools, state, utilities
"""
from core.config import llm, llm_small, dense_embeddings, sparse_embeddings, reranker, client
from core.experiment_config import (
    ExperimentConfig,
    SearchMode,
    ExtractionMode,
    RetryMode,
    DEFAULT_CONFIG,
    ALL_EXPERIMENTS,
    get_experiment,
    list_experiments
)
from core.graph_builder import build_graph, build_graph_by_name
from core.llm_manager import get_llm, get_llm_small, set_temperatures, reset_llm_config

__all__ = [
    # Config
    "llm", "llm_small", "dense_embeddings", "sparse_embeddings", "reranker", "client",
    # Experiment Config
    "ExperimentConfig", "SearchMode", "ExtractionMode", "RetryMode",
    "DEFAULT_CONFIG", "ALL_EXPERIMENTS", "get_experiment", "list_experiments",
    # Graph Builder
    "build_graph", "build_graph_by_name",
    # LLM Manager
    "get_llm", "get_llm_small", "set_temperatures", "reset_llm_config",
]
