"""
hybrid_ingest.py — Hybrid paper ingestion combining scraped PDFs + HuggingFace dataset.

This module provides intelligent merging of two paper sources:
1. Locally scraped PDFs (full text, high quality)
2. HuggingFace jamescalam/ai-arxiv2 dataset (curated selection)

Key features:
- Deduplication by arXiv ID
- Prefers scraped papers when available (better full text)
- Falls back to HF dataset for missing papers
- Preserves all metadata and paper structure
"""

import json
import logging
import re
from pathlib import Path

from tqdm import tqdm

from rag_bench.core.ingest import (
    load_arxiv_dataset,
    parse_paper,
)
from rag_bench.utils.text import extract_sections_from_pdf

logger = logging.getLogger(__name__)


def extract_arxiv_id(doc_id: str) -> str:
    """
    Extract clean arXiv ID from doc_id.

    Examples:
        arxiv_2106.09685 -> 2106.09685
        arxiv_1706_03762 -> 1706.03762
        2106.09685 -> 2106.09685
    """
    # Remove arxiv_ prefix if present
    clean_id = doc_id.replace("arxiv_", "")

    # Convert underscores back to dots (scraped format)
    clean_id = clean_id.replace("_", ".")

    return clean_id


def normalize_arxiv_id(arxiv_id: str) -> str:
    """
    Normalize arXiv ID to a standard format for comparison.

    Handles various formats:
        - 2106.09685 (standard)
        - arxiv:2106.09685
        - 1706.03762v1 (with version)
        - cs/0506023 (old format)
    """
    # Remove arxiv: prefix
    arxiv_id = re.sub(r"^arxiv:", "", arxiv_id, flags=re.IGNORECASE)

    # Remove version number (v1, v2, etc.)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    # Normalize old format (cs/0506023 -> 0506.023)
    if "/" in arxiv_id:
        parts = arxiv_id.split("/")
        if len(parts) == 2 and len(parts[1]) >= 7:
            # Old format: category/YYMMNNN
            num = parts[1]
            arxiv_id = f"{num[:4]}.{num[4:]}"

    return arxiv_id.strip()


def load_scraped_papers(scraped_path: Path) -> dict[str, dict]:
    """
    Load scraped papers and return as dict keyed by normalized arXiv ID.

    Returns:
        Dict mapping normalized arXiv ID -> paper document
    """
    if not scraped_path.exists():
        logger.warning(f"Scraped papers not found: {scraped_path}")
        return {}

    logger.info(f"Loading scraped papers from {scraped_path}")
    with open(scraped_path) as f:
        scraped_docs = json.load(f)

    # Index by normalized arXiv ID, re-extracting sections from full_text
    # so that sub-section headers (3.2.1, etc.) are properly detected
    scraped_by_id = {}
    re_sectioned = 0
    for doc in scraped_docs:
        arxiv_id = extract_arxiv_id(doc["doc_id"])
        normalized_id = normalize_arxiv_id(arxiv_id)
        # Re-extract sections from full_text using updated PDF patterns
        if doc.get("full_text"):
            new_sections = extract_sections_from_pdf(doc["full_text"])
            if len(new_sections) > len(doc.get("sections", {})):
                doc["sections"] = new_sections
                re_sectioned += 1
        scraped_by_id[normalized_id] = doc

    logger.info(f"Loaded {len(scraped_by_id)} scraped papers ({re_sectioned} gained sub-sections)")
    return scraped_by_id


def load_hf_papers(dataset_name: str = "jamescalam/ai-arxiv2", split: str = "train") -> dict[str, dict]:
    """
    Load HuggingFace dataset papers and return as dict keyed by normalized arXiv ID.

    Returns:
        Dict mapping normalized arXiv ID -> paper document
    """
    logger.info(f"Loading HuggingFace dataset: {dataset_name}")
    raw_data = load_arxiv_dataset(dataset_name, split)

    hf_by_id = {}
    skipped = 0

    for row in tqdm(raw_data, desc="Parsing HF papers"):
        doc = parse_paper(row)

        # Skip papers with no meaningful text
        if len(doc["full_text"]) < 100:
            skipped += 1
            continue

        # Extract and normalize arXiv ID
        arxiv_id = doc.get("arxiv_id", "")
        if not arxiv_id or arxiv_id == "unknown":
            skipped += 1
            continue

        normalized_id = normalize_arxiv_id(arxiv_id)
        hf_by_id[normalized_id] = doc

    logger.info(f"Loaded {len(hf_by_id)} HF papers ({skipped} skipped)")
    return hf_by_id


def merge_paper_sources(
    scraped_by_id: dict[str, dict],
    hf_by_id: dict[str, dict],
    prefer_scraped: bool = True,
) -> list[dict]:
    """
    Merge scraped papers and HF dataset, deduplicating by arXiv ID.

    Args:
        scraped_by_id: Dict of scraped papers keyed by normalized arXiv ID
        hf_by_id: Dict of HF dataset papers keyed by normalized arXiv ID
        prefer_scraped: If True, use scraped version when duplicate exists
                       (default: True, since scraped has better full text)

    Returns:
        List of merged paper documents
    """
    merged_docs = []
    stats = {
        "scraped_only": 0,
        "hf_only": 0,
        "both_prefer_scraped": 0,
        "both_prefer_hf": 0,
    }

    # Get all unique arXiv IDs
    all_ids = set(scraped_by_id.keys()) | set(hf_by_id.keys())

    logger.info(f"Merging {len(scraped_by_id)} scraped + {len(hf_by_id)} HF papers")
    logger.info(f"Total unique arXiv IDs: {len(all_ids)}")

    for arxiv_id in sorted(all_ids):
        scraped_doc = scraped_by_id.get(arxiv_id)
        hf_doc = hf_by_id.get(arxiv_id)

        if scraped_doc and hf_doc:
            # Both sources have this paper
            if prefer_scraped:
                merged_docs.append(scraped_doc)
                stats["both_prefer_scraped"] += 1
            else:
                merged_docs.append(hf_doc)
                stats["both_prefer_hf"] += 1

        elif scraped_doc:
            # Only scraped has this paper
            merged_docs.append(scraped_doc)
            stats["scraped_only"] += 1

        elif hf_doc:
            # Only HF has this paper
            merged_docs.append(hf_doc)
            stats["hf_only"] += 1

    # Log statistics
    logger.info("=" * 60)
    logger.info("📊 Hybrid Ingestion Statistics")
    logger.info("=" * 60)
    logger.info(f"Total merged papers: {len(merged_docs)}")
    logger.info(f"  • Scraped only: {stats['scraped_only']}")
    logger.info(f"  • HF only: {stats['hf_only']}")
    logger.info(f"  • Both (using scraped): {stats['both_prefer_scraped']}")
    logger.info(f"  • Both (using HF): {stats['both_prefer_hf']}")

    overlap = stats["both_prefer_scraped"] + stats["both_prefer_hf"]
    if len(scraped_by_id) > 0 and len(hf_by_id) > 0:
        overlap_pct = (overlap / min(len(scraped_by_id), len(hf_by_id))) * 100
        logger.info(f"  • Overlap: {overlap} papers ({overlap_pct:.1f}%)")

    logger.info("=" * 60)

    return merged_docs


def hybrid_ingest(
    scraped_path: Path,
    dataset_name: str = "jamescalam/ai-arxiv2",
    split: str = "train",
    save_path: Path | None = None,
    prefer_scraped: bool = True,
) -> list[dict]:
    """
    Full hybrid ingestion pipeline.

    Loads papers from both scraped PDFs and HuggingFace dataset,
    deduplicates by arXiv ID, and returns merged list.

    Args:
        scraped_path: Path to scraped_papers.json
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        save_path: Optional path to save merged papers JSON
        prefer_scraped: If True, prefer scraped version for duplicates

    Returns:
        List of merged paper documents
    """
    logger.info("=" * 60)
    logger.info("🔄 Starting Hybrid Paper Ingestion")
    logger.info("=" * 60)

    # Load both sources
    scraped_by_id = load_scraped_papers(scraped_path)
    hf_by_id = load_hf_papers(dataset_name, split)

    # Merge with deduplication
    merged_docs = merge_paper_sources(
        scraped_by_id,
        hf_by_id,
        prefer_scraped=prefer_scraped,
    )

    # Optionally cache to disk
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(merged_docs, f, indent=2, default=str)
        logger.info(f"💾 Saved merged papers to {save_path}")

    # Log year range
    years = [d["year"] for d in merged_docs if d.get("year")]
    if years:
        logger.info(f"📅 Year range: {min(years)} - {max(years)}")

    logger.info("=" * 60)
    logger.info("✅ Hybrid ingestion complete!")
    logger.info("=" * 60)

    return merged_docs
