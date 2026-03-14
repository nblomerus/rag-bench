"""
GaRAGe dataset loader.

Loads the GaRAGe (General RAG Evaluation) benchmark from HuggingFace
and maps it to an internal format for evaluation.

Source: AmazonScience/GaRAGe (ACL 2025)
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

try:
    from datasets import load_dataset

    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

logger = logging.getLogger(__name__)

GARAGE_CACHE_DIR = Path(__file__).parent / "cache"
GARAGE_CACHE_FILE = GARAGE_CACHE_DIR / "garage_dataset.json"

# HuggingFace dataset identifier
HF_DATASET_NAME = "AmazonScience/GaRAGe"


@dataclass
class GaRAGePassage:
    """A single passage with relevance annotation."""

    text: str
    is_relevant: bool
    passage_id: str = ""


@dataclass
class GaRAGeEntry:
    """A single GaRAGe benchmark entry."""

    id: str
    question: str
    gold_answer: str
    passages: list[GaRAGePassage] = field(default_factory=list)
    should_deflect: bool = False
    question_tag: str = ""  # e.g., "answerable", "unanswerable"
    topic_tag: str = ""  # topic category if available
    metadata: dict = field(default_factory=dict)


def _try_load_from_huggingface() -> list[dict]:
    """Attempt to load GaRAGe from HuggingFace datasets library."""
    if not _HAS_DATASETS:
        logger.warning("datasets library not installed. Install with: pip install datasets")
        raise ImportError("datasets library not installed. Install with: pip install datasets")
    try:
        logger.info("Loading GaRAGe from HuggingFace: %s", HF_DATASET_NAME)
        ds = load_dataset(HF_DATASET_NAME)

        # GaRAGe typically has train/test splits
        entries = []
        for split_name in ds:
            split = ds[split_name]
            for row in split:
                entries.append(dict(row))

        logger.info("Loaded %d entries from HuggingFace (%s)", len(entries), ", ".join(ds.keys()))
        return entries
    except Exception as e:
        logger.error("Failed to load from HuggingFace: %s", e)
        raise


def _parse_entry(raw: dict, idx: int) -> GaRAGeEntry:
    """Parse a raw HuggingFace row into a GaRAGeEntry."""
    entry_id = raw.get("id") or raw.get("example_id") or f"garage_{idx}"

    question = raw.get("question", raw.get("query", ""))
    gold_answer = raw.get("answer", raw.get("gold_answer", raw.get("reference_answer", "")))

    # Parse passages — GaRAGe uses various field names
    passages = []
    raw_passages = raw.get("passages", raw.get("contexts", raw.get("documents", [])))
    raw_labels = raw.get("passage_labels", raw.get("relevance_labels", []))

    if isinstance(raw_passages, list):
        for i, p in enumerate(raw_passages):
            if isinstance(p, dict):
                text = p.get("text", p.get("content", str(p)))
                is_relevant = p.get("is_relevant", p.get("relevant", False))
                pid = p.get("id", p.get("passage_id", f"p_{i}"))
            else:
                text = str(p)
                is_relevant = bool(raw_labels[i]) if i < len(raw_labels) else False
                pid = f"p_{i}"
            passages.append(GaRAGePassage(text=text, is_relevant=is_relevant, passage_id=pid))

    # Determine if the question should be deflected
    question_tag = raw.get("question_tag", raw.get("type", raw.get("category", "")))
    should_deflect = (
        question_tag.lower() in ("unanswerable", "no_answer", "deflect")
        or raw.get("should_deflect", False)
        or raw.get("answerable", True) is False
    )

    topic_tag = raw.get("topic", raw.get("domain", raw.get("topic_tag", "")))

    return GaRAGeEntry(
        id=str(entry_id),
        question=question,
        gold_answer=gold_answer,
        passages=passages,
        should_deflect=should_deflect,
        question_tag=str(question_tag),
        topic_tag=str(topic_tag),
        metadata={
            k: v
            for k, v in raw.items()
            if k
            not in (
                "question",
                "query",
                "answer",
                "gold_answer",
                "reference_answer",
                "passages",
                "contexts",
                "documents",
                "passage_labels",
                "relevance_labels",
            )
        },
    )


def _cache_entries(entries: list[GaRAGeEntry]) -> None:
    """Cache parsed entries to disk."""
    GARAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = []
    for e in entries:
        d = {
            "id": e.id,
            "question": e.question,
            "gold_answer": e.gold_answer,
            "passages": [{"text": p.text, "is_relevant": p.is_relevant, "passage_id": p.passage_id} for p in e.passages],
            "should_deflect": e.should_deflect,
            "question_tag": e.question_tag,
            "topic_tag": e.topic_tag,
            "metadata": e.metadata,
        }
        data.append(d)
    with open(GARAGE_CACHE_FILE, "w") as f:
        json.dump(data, f)
    logger.info("Cached %d GaRAGe entries to %s", len(data), GARAGE_CACHE_FILE)


def _load_from_cache() -> list[GaRAGeEntry] | None:
    """Load entries from disk cache if available."""
    if not GARAGE_CACHE_FILE.exists():
        return None
    try:
        with open(GARAGE_CACHE_FILE) as f:
            data = json.load(f)
        entries = []
        for d in data:
            passages = [GaRAGePassage(**p) for p in d.get("passages", [])]
            entries.append(
                GaRAGeEntry(
                    id=d["id"],
                    question=d["question"],
                    gold_answer=d["gold_answer"],
                    passages=passages,
                    should_deflect=d.get("should_deflect", False),
                    question_tag=d.get("question_tag", ""),
                    topic_tag=d.get("topic_tag", ""),
                    metadata=d.get("metadata", {}),
                )
            )
        logger.info("Loaded %d GaRAGe entries from cache", len(entries))
        return entries
    except Exception as e:
        logger.warning("Cache load failed: %s", e)
        return None


def load_garage(
    sample_size: int = 0,
    seed: int = 42,
    force_download: bool = False,
) -> list[GaRAGeEntry]:
    """
    Load the GaRAGe benchmark dataset.

    Args:
        sample_size: Number of entries to sample (0 = all).
        seed: Random seed for reproducible sampling.
        force_download: Skip cache and re-download from HuggingFace.

    Returns:
        List of GaRAGeEntry objects.
    """
    entries = None

    if not force_download:
        entries = _load_from_cache()

    if entries is None:
        raw_data = _try_load_from_huggingface()
        entries = [_parse_entry(raw, i) for i, raw in enumerate(raw_data)]
        _cache_entries(entries)

    # Sample if requested
    if sample_size > 0 and sample_size < len(entries):
        rng = random.Random(seed)
        entries = rng.sample(entries, sample_size)

    logger.info("GaRAGe dataset: %d entries loaded (sample_size=%d)", len(entries), sample_size)
    return entries
