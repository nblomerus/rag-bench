"""
Citation quality evaluation queries for benchmarking.

These queries test whether primary sources rank appropriately for definition/fundamental questions.
Used to measure improvement in citation quality after dataset/pipeline changes.
"""

CITATION_QUALITY_QUERIES = [
    # Transformer & Attention Mechanisms
    {
        "question": "What is the scaled dot-product attention formula?",
        "primary_source": "1706.03762",  # Attention Is All You Need
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "transformers",
    },
    {
        "question": "Who introduced the Transformer architecture?",
        "primary_source": "1706.03762",
        "expected_rank": 1,
        "query_type": "original_source",
        "topic": "transformers",
    },
    {
        "question": "What is multi-head attention?",
        "primary_source": "1706.03762",
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "transformers",
    },
    # BERT & Masked Language Modeling
    {
        "question": "What is BERT and how does it work?",
        "primary_source": "1810.04805",  # BERT
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "language_models",
    },
    {
        "question": "What is masked language modeling?",
        "primary_source": "1810.04805",
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "language_models",
    },
    # GPT-3 & Few-Shot Learning
    {
        "question": "What are the capabilities of GPT-3?",
        "primary_source": "2005.14165",  # GPT-3
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "language_models",
    },
    {
        "question": "What is in-context learning?",
        "primary_source": "2005.14165",
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "language_models",
    },
    # Scaling Laws
    {
        "question": "What are scaling laws for neural language models?",
        "primary_source": "2001.08361",  # Scaling Laws
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "scaling",
    },
    {
        "question": "What is compute-optimal training?",
        "primary_source": "2203.15556",  # Chinchilla
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "scaling",
    },
    # RLHF & Alignment
    {
        "question": "What is RLHF in the context of language models?",
        "primary_source": "2203.02155",  # InstructGPT
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "alignment",
    },
    {
        "question": "How does InstructGPT work?",
        "primary_source": "2203.02155",
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "alignment",
    },
    # LoRA & Parameter-Efficient Fine-Tuning
    {
        "question": "What is LoRA low-rank adaptation?",
        "primary_source": "2106.09685",  # LoRA
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "fine_tuning",
    },
    {
        "question": "How does parameter-efficient fine-tuning work?",
        "primary_source": "2106.09685",
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "fine_tuning",
    },
    # RAG & Retrieval
    {
        "question": "What is retrieval-augmented generation?",
        "primary_source": "2005.11401",  # RAG
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "retrieval",
    },
    {
        "question": "How does RAG combine retrieval and generation?",
        "primary_source": "2005.11401",
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "retrieval",
    },
    # Diffusion Models
    {
        "question": "What are denoising diffusion probabilistic models?",
        "primary_source": "2006.11239",  # DDPM
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "diffusion",
    },
    {
        "question": "How does Stable Diffusion work?",
        "primary_source": "2112.10752",  # Stable Diffusion
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "diffusion",
    },
    # Vision-Language Models
    {
        "question": "What is CLIP and how does it work?",
        "primary_source": "2103.00020",  # CLIP
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "multimodal",
    },
    {
        "question": "How does contrastive language-image pretraining work?",
        "primary_source": "2103.00020",
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "multimodal",
    },
    # Mixture of Experts
    {
        "question": "What is the Switch Transformer?",
        "primary_source": "2101.03961",  # Switch Transformer
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "moe",
    },
    # Positional Encodings
    {
        "question": "What is Rotary Position Embedding (RoPE)?",
        "primary_source": "2104.09864",  # RoPE
        "expected_rank": 1,
        "query_type": "definition",
        "topic": "positional_encoding",
    },
]


def get_queries_by_topic(topic: str) -> list[dict]:
    """Get all queries for a specific topic"""
    return [q for q in CITATION_QUALITY_QUERIES if q["topic"] == topic]


def get_queries_by_type(query_type: str) -> list[dict]:
    """Get all queries of a specific type"""
    return [q for q in CITATION_QUALITY_QUERIES if q["query_type"] == query_type]


def get_all_primary_sources() -> set[str]:
    """Get unique set of all primary source ArXiv IDs"""
    return set(q["primary_source"] for q in CITATION_QUALITY_QUERIES)
