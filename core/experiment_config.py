"""
Experiment Configuration - Defines all configurable parameters for experiments.

Experiments are defined as simple config objects instead of duplicated code.
Each experiment is just a different combination of settings.
"""
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class SearchMode(Enum):
    """Search strategy."""
    SINGLE = "single"           # Single query search
    MULTI_QUERY = "multi_query" # Generate 2-3 queries (better for comparisons)


class ExtractionMode(Enum):
    """Content extraction strategy."""
    NONE = "none"       # No extraction, use raw chunks
    BATCHED = "batched" # One LLM call for all chunks (fast)


class RetryMode(Enum):
    """Self-correction strategy."""
    NONE = "none"   # No retry
    SMART = "smart" # Smart retry - skip valid "out of scope" responses


@dataclass
class ExperimentConfig:
    """Configuration for a RAG experiment."""
    name: str
    description: str

    # LLM Temperatures
    llm_temp: float = 0.3       # Main LLM (answer generation)
    llm_small_temp: float = 0.3 # Small LLM (extraction, judge)

    # Feature flags
    search_mode: SearchMode = SearchMode.MULTI_QUERY
    extraction_mode: ExtractionMode = ExtractionMode.BATCHED
    retry_mode: RetryMode = RetryMode.SMART

    # Retrieval settings
    retrieval_k: int = 5
    max_retries: int = 2


# =============================================================================
# PRE-DEFINED CONFIGS
# =============================================================================

# Baseline - Simple approach (no extraction, no retry)
BASELINE = ExperimentConfig(
    name="baseline",
    description="Simple search + answer (no extraction/retry)",
    search_mode=SearchMode.SINGLE,
    extraction_mode=ExtractionMode.NONE,
    retry_mode=RetryMode.NONE
)

# V4: Smart Retry (single query + smart retry)
OPT_V4 = ExperimentConfig(
    name="opt-v4",
    description="Single query + smart retry",
    search_mode=SearchMode.SINGLE,
    extraction_mode=ExtractionMode.BATCHED,
    retry_mode=RetryMode.SMART
)

# V6: Fast Multi-Query
OPT_V6 = ExperimentConfig(
    name="opt-v6",
    description="Multi-query search + batched extraction",
    llm_temp=0.3,
    llm_small_temp=0.1,
    search_mode=SearchMode.MULTI_QUERY,
    extraction_mode=ExtractionMode.BATCHED,
    retry_mode=RetryMode.SMART
)

# V7 Temperature Configs
OPT_V7_A = ExperimentConfig(
    name="opt-v7-A",
    description="Strict temps (0.1/0.0)",
    llm_temp=0.1,
    llm_small_temp=0.0
)

OPT_V7_B = ExperimentConfig(
    name="opt-v7-B",
    description="Balanced v1 (0.3/0.1)",
    llm_temp=0.3,
    llm_small_temp=0.1
)

OPT_V7_C = ExperimentConfig(
    name="opt-v7-C",
    description="Creative (0.5/0.2)",
    llm_temp=0.5,
    llm_small_temp=0.2
)

OPT_V7_D = ExperimentConfig(
    name="opt-v7-D",
    description="WINNER - Balanced (0.3/0.3)",
    llm_temp=0.3,
    llm_small_temp=0.3
)

OPT_V7_E = ExperimentConfig(
    name="opt-v7-E",
    description="Creative main (0.7/0.1)",
    llm_temp=0.7,
    llm_small_temp=0.1
)

# Default config (winner)
DEFAULT_CONFIG = OPT_V7_D


# =============================================================================
# REGISTRY
# =============================================================================
ALL_EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "baseline": BASELINE,
    "opt-v4": OPT_V4,
    "opt-v6": OPT_V6,
    "opt-v7-A": OPT_V7_A,
    "opt-v7-B": OPT_V7_B,
    "opt-v7-C": OPT_V7_C,
    "opt-v7-D": OPT_V7_D,
    "opt-v7-E": OPT_V7_E,
}


def get_experiment(name: str) -> ExperimentConfig:
    """Get experiment config by name."""
    if name not in ALL_EXPERIMENTS:
        available = ", ".join(ALL_EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {name}. Available: {available}")
    return ALL_EXPERIMENTS[name]


def list_experiments() -> List[str]:
    """List all available experiment names."""
    return list(ALL_EXPERIMENTS.keys())


def print_experiments():
    """Print all available experiments."""
    print("\n" + "="*60)
    print("AVAILABLE EXPERIMENTS")
    print("="*60)

    for name, cfg in ALL_EXPERIMENTS.items():
        print(f"\n  {name}")
        print(f"    {cfg.description}")
        print(f"    Temps: llm={cfg.llm_temp}, small={cfg.llm_small_temp}")

    print("\n" + "="*60)
