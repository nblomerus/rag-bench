"""
strategies — Swappable text-splitting strategies for PaperChunker.

Each strategy implements the ChunkingStrategy protocol (split_text method).
Use get_strategy() to instantiate by name from a config dict.
"""

from rag_bench.core.strategies.recursive import RecursiveStrategy
from rag_bench.core.strategies.semantic import SemanticStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "recursive": RecursiveStrategy,
    "semantic": SemanticStrategy,
}


def get_strategy(name: str, config: dict | None = None):
    """Instantiate a chunking strategy by name.

    Args:
        name: Strategy key (e.g. "recursive", "semantic").
        config: Strategy-specific parameters passed as keyword arguments.

    Raises:
        ValueError: If the strategy name is not registered.
    """
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Unknown chunking strategy '{name}'. Available: {available}")

    cls = STRATEGY_REGISTRY[name]
    return cls(**(config or {}))
