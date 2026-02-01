"""
Configuration Loader Module
Loads and validates system configuration with all dimensions specified.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration with explicit dimensions."""
    model: str = "nomic-embed-text"
    dimension: int = 768  # EXPLICIT: Vector dimension
    batch_size: int = 32
    normalize: bool = True
    max_input_length: int = 8192
    cache_enabled: bool = True
    cache_size: int = 1000
    model_params: int = 137_000_000
    memory_footprint_mb: int = 548
    latency_ms_per_batch: int = 75


@dataclass
class LLMConfig:
    """Language model configuration with context windows."""
    model: str = "llama3.2"
    provider: str = "ollama"
    max_context_length: int = 131_072  # EXPLICIT: 128K tokens
    working_context_limit: int = 16_384  # EXPLICIT: Practical limit
    max_output_tokens: int = 2048  # EXPLICIT: Max response
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    
    # Model specs
    architecture: str = "decoder-only-transformer"
    parameters: int = 3_000_000_000  # EXPLICIT: 3B params
    quantization: str = "Q4_K_M"
    memory_footprint_gb: float = 2.0
    tokens_per_second_cpu: int = 40
    tokens_per_second_gpu: int = 125
    
    # Token budget breakdown
    token_budget: Dict[str, int] = field(default_factory=lambda: {
        "system_prompt": 500,
        "conversation_history": 2000,
        "retrieved_context": 3000,
        "user_query": 100,
        "response_generation": 2048,
        "safety_buffer": 8736
    })


@dataclass
class ChunkConfig:
    """Individual chunk configuration."""
    size_chars: int
    overlap_chars: int
    overlap_percent: int
    estimated_tokens: int


@dataclass
class ChunkingConfig:
    """Text chunking configuration with explicit sizes."""
    strategy: str = "hierarchical"
    
    # Parent chunks - EXPLICIT dimensions
    parent: ChunkConfig = field(default_factory=lambda: ChunkConfig(
        size_chars=1500,
        overlap_chars=300,
        overlap_percent=20,
        estimated_tokens=375
    ))
    
    # Child chunks - EXPLICIT dimensions
    child: ChunkConfig = field(default_factory=lambda: ChunkConfig(
        size_chars=500,
        overlap_chars=100,
        overlap_percent=20,
        estimated_tokens=125
    ))
    
    children_per_parent: int = 3
    separators: list = field(default_factory=lambda: [
        "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " "
    ])


@dataclass
class VectorDBConfig:
    """Vector database configuration with HNSW parameters."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "documents_hybrid"
    
    # Dense vector config - EXPLICIT dimension
    dense_size: int = 768  # MUST match embedding dimension
    dense_distance: str = "Cosine"
    dense_on_disk: bool = False
    
    # Sparse vector config
    sparse_modifier: str = "Idf"
    sparse_on_disk: bool = False
    
    # HNSW parameters - EXPLICIT
    hnsw_m: int = 16  # Connections per node
    hnsw_ef_construct: int = 100
    hnsw_ef: int = 128
    hnsw_full_scan_threshold: int = 10000
    
    # Storage estimates per 100-page doc
    storage_per_doc: Dict[str, Any] = field(default_factory=lambda: {
        "raw_text_kb": 500,
        "num_chunks": 200,
        "dense_vectors_kb": 600,
        "sparse_vectors_kb": 1000,
        "metadata_kb": 50,
        "index_overhead_kb": 200,
        "total_mb": 2.55
    })


@dataclass
class RetrievalConfig:
    """Hybrid search and retrieval configuration."""
    top_k: int = 6  # EXPLICIT: Final chunks
    top_k_dense: int = 20  # EXPLICIT: Initial dense results
    top_k_sparse: int = 20  # EXPLICIT: Initial sparse results
    
    # Fusion weights - EXPLICIT percentages
    dense_weight: float = 0.7  # 70% semantic
    sparse_weight: float = 0.3  # 30% keyword
    
    # RRF constant
    rrf_constant: int = 60
    
    # Grading thresholds
    relevance_threshold: float = 7.0  # 0-10 scale
    min_relevant_chunks: int = 3
    
    # Performance targets (milliseconds)
    latency_targets_ms: Dict[str, int] = field(default_factory=lambda: {
        "dense_search": 30,
        "sparse_search": 25,
        "fusion_reranking": 5,
        "total": 60
    })


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    purpose: str
    max_input_tokens: int = 0
    output_tokens: int = 0
    temperature: float = 0.3


@dataclass
class AgentsConfig:
    """All agent configurations with explicit dimensions."""
    summarizer: AgentConfig = field(default_factory=lambda: AgentConfig(
        purpose="condense_conversation_history",
        max_input_tokens=2000,  # EXPLICIT: Input limit
        output_tokens=250,  # EXPLICIT: Summary length
        temperature=0.3
    ))
    
    query_rewriter: AgentConfig = field(default_factory=lambda: AgentConfig(
        purpose="expand_and_clarify_queries",
        max_input_tokens=400,
        output_tokens=75,
        temperature=0.5
    ))
    
    retriever: AgentConfig = field(default_factory=lambda: AgentConfig(
        purpose="fetch_and_grade_documents",
        max_input_tokens=0,
        output_tokens=0,
        temperature=0.0
    ))
    
    # Additional parameters
    summarizer_compression_ratio: int = 7
    summarizer_activation_frequency: int = 5
    rewriter_num_variations: int = 3
    retriever_max_queries: int = 3
    retriever_grading_scale: int = 10


@dataclass
class MemoryConfig:
    """Conversation memory configuration with explicit limits."""
    max_messages: int = 10  # EXPLICIT: Last N messages
    max_tokens: int = 2000  # EXPLICIT: Token limit
    pruning_strategy: str = "sliding_window_with_summary"
    compression_ratio: int = 5
    avg_message_tokens: int = 200
    persist_conversations: bool = True
    conversation_ttl_hours: int = 168


@dataclass
class PerformanceConfig:
    """Performance targets and monitoring."""
    # SLA targets (milliseconds) - EXPLICIT
    sla_targets: Dict[str, int] = field(default_factory=lambda: {
        "p50_ms": 4000,
        "p90_ms": 6000,
        "p99_ms": 10000,
        "timeout_ms": 30000
    })
    
    # Component latency breakdown - EXPLICIT
    component_latency_ms: Dict[str, int] = field(default_factory=lambda: {
        "history_summarization": 500,
        "query_rewriting": 800,
        "embedding_generation": 150,
        "vector_search": 60,
        "document_grading": 400,
        "response_generation": 2000,
        "overhead": 100
    })


@dataclass
class SystemConfig:
    """
    Complete system configuration with ALL dimensions explicitly specified.
    This is the master configuration class that holds all subsystem configs.
    """
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/system_config.yaml") -> "SystemConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            SystemConfig instance with all dimensions loaded
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return cls()
        
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Build configuration from YAML
        return cls(
            embedding=cls._build_embedding_config(yaml_config.get('embedding', {})),
            llm=cls._build_llm_config(yaml_config.get('llm', {})),
            chunking=cls._build_chunking_config(yaml_config.get('chunking', {})),
            vector_db=cls._build_vector_db_config(yaml_config.get('vector_db', {})),
            retrieval=cls._build_retrieval_config(yaml_config.get('retrieval', {})),
            agents=cls._build_agents_config(yaml_config.get('agents', {})),
            memory=cls._build_memory_config(yaml_config.get('memory', {})),
            performance=cls._build_performance_config(yaml_config.get('performance', {}))
        )
    
    @staticmethod
    def _build_embedding_config(config: Dict) -> EmbeddingConfig:
        """Build embedding configuration."""
        return EmbeddingConfig(
            model=config.get('model', 'nomic-embed-text'),
            dimension=config.get('dimension', 768),
            batch_size=config.get('batch_size', 32),
            normalize=config.get('normalize', True),
            max_input_length=config.get('max_input_length', 8192),
            cache_enabled=config.get('cache_enabled', True),
            cache_size=config.get('cache_size', 1000),
            model_params=config.get('model_params', 137_000_000),
            memory_footprint_mb=config.get('memory_footprint_mb', 548),
            latency_ms_per_batch=config.get('latency_ms_per_batch', 75)
        )
    
    @staticmethod
    def _build_llm_config(config: Dict) -> LLMConfig:
        """Build LLM configuration."""
        return LLMConfig(
            model=config.get('model', 'llama3.2'),
            provider=config.get('provider', 'ollama'),
            max_context_length=config.get('max_context_length', 131_072),
            working_context_limit=config.get('working_context_limit', 16_384),
            max_output_tokens=config.get('max_output_tokens', 2048),
            temperature=config.get('temperature', 0.1),
            top_p=config.get('top_p', 0.9),
            top_k=config.get('top_k', 40),
            repeat_penalty=config.get('repeat_penalty', 1.1),
            token_budget=config.get('token_budget', {})
        )
    
    @staticmethod
    def _build_chunking_config(config: Dict) -> ChunkingConfig:
        """Build chunking configuration."""
        parent_config = config.get('parent', {})
        child_config = config.get('child', {})
        
        return ChunkingConfig(
            strategy=config.get('strategy', 'hierarchical'),
            parent=ChunkConfig(
                size_chars=parent_config.get('size_chars', 1500),
                overlap_chars=parent_config.get('overlap_chars', 300),
                overlap_percent=parent_config.get('overlap_percent', 20),
                estimated_tokens=parent_config.get('estimated_tokens', 375)
            ),
            child=ChunkConfig(
                size_chars=child_config.get('size_chars', 500),
                overlap_chars=child_config.get('overlap_chars', 100),
                overlap_percent=child_config.get('overlap_percent', 20),
                estimated_tokens=child_config.get('estimated_tokens', 125)
            ),
            children_per_parent=config.get('children_per_parent', 3),
            separators=config.get('separators', ["\n\n", "\n", ". "])
        )
    
    @staticmethod
    def _build_vector_db_config(config: Dict) -> VectorDBConfig:
        """Build vector database configuration."""
        dense = config.get('dense', {})
        hnsw = config.get('hnsw', {})
        
        return VectorDBConfig(
            host=config.get('host', 'localhost'),
            port=config.get('port', 6333),
            collection_name=config.get('collection_name', 'documents_hybrid'),
            dense_size=dense.get('size', 768),
            dense_distance=dense.get('distance', 'Cosine'),
            hnsw_m=hnsw.get('m', 16),
            hnsw_ef_construct=hnsw.get('ef_construct', 100),
            hnsw_ef=hnsw.get('ef', 128)
        )
    
    @staticmethod
    def _build_retrieval_config(config: Dict) -> RetrievalConfig:
        """Build retrieval configuration."""
        return RetrievalConfig(
            top_k=config.get('top_k', 6),
            top_k_dense=config.get('top_k_dense', 20),
            top_k_sparse=config.get('top_k_sparse', 20),
            dense_weight=config.get('dense_weight', 0.7),
            sparse_weight=config.get('sparse_weight', 0.3),
            rrf_constant=config.get('rrf_constant', 60),
            relevance_threshold=config.get('relevance_threshold', 7.0),
            min_relevant_chunks=config.get('min_relevant_chunks', 3)
        )
    
    @staticmethod
    def _build_agents_config(config: Dict) -> AgentsConfig:
        """Build agents configuration."""
        summarizer = config.get('summarizer', {})
        query_rewriter = config.get('query_rewriter', {})
        
        agents = AgentsConfig()
        agents.summarizer = AgentConfig(
            purpose=summarizer.get('purpose', 'condense_conversation_history'),
            max_input_tokens=summarizer.get('max_input_tokens', 2000),
            output_tokens=summarizer.get('output_tokens', 250),
            temperature=summarizer.get('temperature', 0.3)
        )
        agents.query_rewriter = AgentConfig(
            purpose=query_rewriter.get('purpose', 'expand_and_clarify_queries'),
            max_input_tokens=query_rewriter.get('max_input_tokens', 400),
            output_tokens=query_rewriter.get('output_tokens', 75),
            temperature=query_rewriter.get('temperature', 0.5)
        )
        return agents
    
    @staticmethod
    def _build_memory_config(config: Dict) -> MemoryConfig:
        """Build memory configuration."""
        return MemoryConfig(
            max_messages=config.get('max_messages', 10),
            max_tokens=config.get('max_tokens', 2000),
            pruning_strategy=config.get('pruning_strategy', 'sliding_window_with_summary'),
            compression_ratio=config.get('compression_ratio', 5)
        )
    
    @staticmethod
    def _build_performance_config(config: Dict) -> PerformanceConfig:
        """Build performance configuration."""
        return PerformanceConfig(
            sla_targets=config.get('sla_targets', {}),
            component_latency_ms=config.get('component_latency_ms', {})
        )
    
    def validate(self) -> bool:
        """
        Validate configuration for consistency.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Validate embedding dimension matches vector DB
        if self.embedding.dimension != self.vector_db.dense_size:
            raise ValueError(
                f"Embedding dimension ({self.embedding.dimension}) must match "
                f"vector DB dense size ({self.vector_db.dense_size})"
            )
        
        # Validate chunk overlap percentages
        if self.chunking.parent.overlap_chars >= self.chunking.parent.size_chars:
            raise ValueError("Parent chunk overlap must be less than chunk size")
        
        if self.chunking.child.overlap_chars >= self.chunking.child.size_chars:
            raise ValueError("Child chunk overlap must be less than chunk size")
        
        # Validate retrieval weights sum to 1.0
        weight_sum = self.retrieval.dense_weight + self.retrieval.sparse_weight
        if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Dense and sparse weights must sum to 1.0 (got {weight_sum})"
            )
        
        # Validate token budget
        total_budget = sum(self.llm.token_budget.values())
        if total_budget > self.llm.working_context_limit:
            raise ValueError(
                f"Token budget ({total_budget}) exceeds working limit "
                f"({self.llm.working_context_limit})"
            )
        
        logger.info("✓ Configuration validation passed")
        return True
    
    def print_summary(self) -> None:
        """Print a summary of key dimensions."""
        print("\n" + "="*70)
        print(" SYSTEM CONFIGURATION SUMMARY")
        print("="*70)
        
        print(f"\n📊 EMBEDDING")
        print(f"  Model: {self.embedding.model}")
        print(f"  Dimension: {self.embedding.dimension}")
        print(f"  Memory per vector: {(self.embedding.dimension * 4) / 1024:.2f} KB")
        
        print(f"\n🤖 LANGUAGE MODEL")
        print(f"  Model: {self.llm.model}")
        print(f"  Parameters: {self.llm.parameters:,}")
        print(f"  Context Window: {self.llm.working_context_limit:,} tokens")
        print(f"  Max Output: {self.llm.max_output_tokens} tokens")
        
        print(f"\n📝 CHUNKING")
        print(f"  Parent: {self.chunking.parent.size_chars} chars "
              f"(~{self.chunking.parent.estimated_tokens} tokens)")
        print(f"  Child: {self.chunking.child.size_chars} chars "
              f"(~{self.chunking.child.estimated_tokens} tokens)")
        print(f"  Overlap: {self.chunking.child.overlap_percent}%")
        
        print(f"\n🔍 RETRIEVAL")
        print(f"  Top-K: {self.retrieval.top_k} chunks")
        print(f"  Dense Weight: {self.retrieval.dense_weight:.0%}")
        print(f"  Sparse Weight: {self.retrieval.sparse_weight:.0%}")
        print(f"  Relevance Threshold: {self.retrieval.relevance_threshold}/10")
        
        print(f"\n💾 MEMORY")
        print(f"  Max Messages: {self.memory.max_messages}")
        print(f"  Max Tokens: {self.memory.max_tokens}")
        print(f"  Compression: {self.memory.compression_ratio}:1")
        
        print(f"\n⚡ PERFORMANCE TARGETS")
        print(f"  P50 Latency: {self.performance.sla_targets['p50_ms']}ms")
        print(f"  P90 Latency: {self.performance.sla_targets['p90_ms']}ms")
        print(f"  Timeout: {self.performance.sla_targets['timeout_ms']}ms")
        
        print("\n" + "="*70 + "\n")


# Convenience function
def load_config(config_path: str = "config/system_config.yaml") -> SystemConfig:
    """
    Load and validate system configuration.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Validated SystemConfig instance
    """
    config = SystemConfig.from_yaml(config_path)
    config.validate()
    return config


if __name__ == "__main__":
    # Test configuration loading
    print("Loading configuration...")
    config = load_config()
    config.print_summary()
