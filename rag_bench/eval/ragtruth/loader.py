"""
RAGTruth dataset loader.

Loads the RAGTruth hallucination corpus from GitHub and maps it
to an internal format for evaluation.

Source: ParticleMedia/RAGTruth (ACL 2024)
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RAGTRUTH_CACHE_DIR = Path(__file__).parent / "cache"
RAGTRUTH_CACHE_FILE = RAGTRUTH_CACHE_DIR / "ragtruth_dataset.json"

# GitHub raw URLs for RAGTruth data files
RAGTRUTH_BASE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
SOURCE_INFO_URL = f"{RAGTRUTH_BASE_URL}/source_info.jsonl"
RESPONSE_URL = f"{RAGTRUTH_BASE_URL}/response.jsonl"


@dataclass
class HallucinationSpan:
    """A single hallucinated span in a response."""

    text: str
    start: int = -1
    end: int = -1
    label_type: str = ""  # "Evident Conflict", "Subtle Conflict", "Evident Baseless", "Subtle Baseless"


@dataclass
class RAGTruthEntry:
    """A single RAGTruth benchmark entry."""

    id: str
    source_id: str
    task_type: str  # "QA", "Summarization", "Data-to-Text"
    source_info: str  # The context/source material
    prompt: str  # The question or instruction
    reference_response: str  # Model-generated response with annotations
    hallucination_spans: list[HallucinationSpan] = field(default_factory=list)
    has_hallucination: bool = False
    metadata: dict = field(default_factory=dict)


def _download_jsonl(url: str) -> list[dict]:
    """Download and parse a JSONL file from a URL."""
    import urllib.request

    logger.info("Downloading: %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RAG-Bench/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read().decode("utf-8")
        entries = []
        for line in content.strip().split("\n"):
            if line.strip():
                entries.append(json.loads(line))
        return entries
    except Exception as e:
        logger.error("Download failed for %s: %s", url, e)
        raise


def _try_load_from_github() -> tuple[list[dict], list[dict]]:
    """Download RAGTruth source_info and response JSONL files."""
    source_info = _download_jsonl(SOURCE_INFO_URL)
    responses = _download_jsonl(RESPONSE_URL)
    logger.info("Downloaded %d source entries and %d response entries", len(source_info), len(responses))
    return source_info, responses


def _try_load_from_huggingface() -> tuple[list[dict], list[dict]]:
    """Fallback: try loading from HuggingFace datasets."""
    try:
        from datasets import load_dataset

        ds = load_dataset("ParticleMedia/RAGTruth")
        source_info = []
        responses = []
        for split in ds:
            for row in ds[split]:
                # Map HuggingFace schema to our expected format
                responses.append(dict(row))
        return source_info, responses
    except Exception as e:
        logger.warning("HuggingFace fallback failed: %s", e)
        raise


def _parse_hallucination_spans(raw_spans: list) -> list[HallucinationSpan]:
    """Parse raw hallucination span annotations."""
    spans = []
    if not raw_spans:
        return spans

    for s in raw_spans:
        if isinstance(s, dict):
            spans.append(
                HallucinationSpan(
                    text=s.get("text", s.get("hallucinated_text", "")),
                    start=s.get("start", s.get("start_idx", -1)),
                    end=s.get("end", s.get("end_idx", -1)),
                    label_type=s.get("label_type", s.get("type", s.get("hallucination_type", ""))),
                )
            )
        elif isinstance(s, str):
            spans.append(HallucinationSpan(text=s))
    return spans


def _extract_source_text(data: dict) -> str:
    """Extract source text from a data dict, handling nested dicts."""
    for key in ("source_info", "context", "text"):
        val = data.get(key, "")
        if isinstance(val, dict):
            # RAGTruth QA entries store {"question": ..., "passages": ...}
            return val.get("passages", "") or val.get("text", "") or val.get("context", "")
        if isinstance(val, str) and val:
            return val
    return ""


def _merge_and_parse(
    source_info: list[dict],
    responses: list[dict],
) -> list[RAGTruthEntry]:
    """Merge source info with responses and parse into RAGTruthEntry objects."""
    # Build source_id → source_info lookup
    source_map = {}
    for s in source_info:
        sid = s.get("source_id", s.get("id", ""))
        source_map[sid] = s

    entries = []
    for i, resp in enumerate(responses):
        source_id = resp.get("source_id", resp.get("id", f"ragtruth_{i}"))
        entry_id = resp.get("response_id", resp.get("id", f"ragtruth_{i}"))

        # Get source text
        source_data = source_map.get(source_id, {})
        source_text = _extract_source_text(source_data) or _extract_source_text(resp)

        task_type = resp.get("task_type", "") or source_data.get("task_type", "") or resp.get("type", "")

        prompt = resp.get("prompt", "") or resp.get("question", "") or source_data.get("prompt", "")

        reference_response = resp.get("response", resp.get("generated_text", ""))

        # Parse hallucination annotations
        raw_spans = resp.get("hallucination_spans", []) or resp.get("hallucinations", []) or resp.get("spans", [])
        hallucination_spans = _parse_hallucination_spans(raw_spans)

        has_hallucination = (
            resp.get("has_hallucination", False) or resp.get("is_hallucinated", False) or len(hallucination_spans) > 0
        )

        entries.append(
            RAGTruthEntry(
                id=str(entry_id),
                source_id=str(source_id),
                task_type=task_type,
                source_info=source_text,
                prompt=prompt,
                reference_response=reference_response,
                hallucination_spans=hallucination_spans,
                has_hallucination=has_hallucination,
                metadata={
                    k: v
                    for k, v in resp.items()
                    if k
                    not in (
                        "response",
                        "generated_text",
                        "hallucination_spans",
                        "hallucinations",
                        "spans",
                        "source_info",
                        "context",
                        "prompt",
                        "question",
                    )
                },
            )
        )

    return entries


def _cache_entries(entries: list[RAGTruthEntry]) -> None:
    """Cache parsed entries to disk."""
    RAGTRUTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = []
    for e in entries:
        data.append(
            {
                "id": e.id,
                "source_id": e.source_id,
                "task_type": e.task_type,
                "source_info": e.source_info,
                "prompt": e.prompt,
                "reference_response": e.reference_response,
                "hallucination_spans": [
                    {"text": s.text, "start": s.start, "end": s.end, "label_type": s.label_type}
                    for s in e.hallucination_spans
                ],
                "has_hallucination": e.has_hallucination,
                "metadata": e.metadata,
            }
        )
    with open(RAGTRUTH_CACHE_FILE, "w") as f:
        json.dump(data, f)
    logger.info("Cached %d RAGTruth entries to %s", len(data), RAGTRUTH_CACHE_FILE)


def _load_from_cache() -> list[RAGTruthEntry] | None:
    """Load entries from disk cache if available."""
    if not RAGTRUTH_CACHE_FILE.exists():
        return None
    try:
        with open(RAGTRUTH_CACHE_FILE) as f:
            data = json.load(f)
        entries = []
        for d in data:
            spans = [HallucinationSpan(**s) for s in d.get("hallucination_spans", [])]
            entries.append(
                RAGTruthEntry(
                    id=d["id"],
                    source_id=d.get("source_id", ""),
                    task_type=d.get("task_type", ""),
                    source_info=d.get("source_info", ""),
                    prompt=d.get("prompt", ""),
                    reference_response=d.get("reference_response", ""),
                    hallucination_spans=spans,
                    has_hallucination=d.get("has_hallucination", False),
                    metadata=d.get("metadata", {}),
                )
            )
        logger.info("Loaded %d RAGTruth entries from cache", len(entries))
        return entries
    except Exception as e:
        logger.warning("Cache load failed: %s", e)
        return None


def load_ragtruth(
    sample_size: int = 0,
    task_type: str = "QA",
    seed: int = 42,
    force_download: bool = False,
) -> list[RAGTruthEntry]:
    """
    Load the RAGTruth benchmark dataset.

    Args:
        sample_size: Number of entries to sample (0 = all).
        task_type: Filter by task type ("QA", "Summarization", "Data-to-Text", or "" for all).
        seed: Random seed for reproducible sampling.
        force_download: Skip cache and re-download.

    Returns:
        List of RAGTruthEntry objects.
    """
    entries = None

    if not force_download:
        entries = _load_from_cache()

    if entries is None:
        try:
            source_info, responses = _try_load_from_github()
        except Exception:
            logger.warning("GitHub download failed, trying HuggingFace fallback")
            source_info, responses = _try_load_from_huggingface()

        entries = _merge_and_parse(source_info, responses)
        _cache_entries(entries)

    # Filter by task type
    if task_type:
        entries = [e for e in entries if e.task_type.lower() == task_type.lower()]

    # Sample if requested
    if sample_size > 0 and sample_size < len(entries):
        rng = random.Random(seed)
        entries = rng.sample(entries, sample_size)

    logger.info(
        "RAGTruth dataset: %d entries loaded (task_type=%s, sample_size=%d)", len(entries), task_type, sample_size
    )
    return entries
