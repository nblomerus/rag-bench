"""
Pure metric computation functions for RAG-Bench evaluation.

All functions are deterministic with no LLM calls — fast and fully unit-testable.
Three categories: retrieval metrics, citation quality metrics, and completeness.
"""

import math
import re

# ═══════════════════════════════════════════════════════════════════════════
# 2a. Retrieval Metrics
# ═══════════════════════════════════════════════════════════════════════════


def extract_paper_ids(results: list[dict]) -> list[str]:
    """
    Extract unique paper ArXiv IDs from retrieval results in rank order.

    Handles multiple metadata formats:
      - arxiv_id: "arxiv_2301_12345" → "2301.12345"
      - arxiv_id: "2301.12345" (already normalized)
      - paper_id field as fallback

    Returns deduplicated list preserving first-occurrence order.
    """
    seen = set()
    ids = []
    for r in results:
        raw = r.get("arxiv_id") or r.get("paper_id") or ""
        if not raw:
            metadata = r.get("metadata", {})
            if isinstance(metadata, dict):
                raw = metadata.get("arxiv_id") or metadata.get("paper_id") or ""
        normalized = _normalize_arxiv_id(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ids.append(normalized)
    return ids


def _normalize_arxiv_id(raw: str) -> str:
    """Normalize arxiv ID: 'arxiv_2301_12345' → '2301.12345', etc."""
    raw = raw.strip()
    if not raw:
        return ""
    # Strip 'arxiv_' prefix
    if raw.lower().startswith("arxiv_"):
        raw = raw[6:]
    # Convert underscore format: '2301_12345' → '2301.12345'
    # Only convert the first underscore between digit groups
    m = re.match(r"^(\d{4})_(\d{4,5})$", raw)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return raw


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:k]| / k
    """
    if k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(expected_ids)
    return len(top_k & relevant) / k


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|
    """
    if not expected_ids:
        return 1.0  # Nothing expected → perfect recall
    top_k = set(retrieved_ids[:k])
    relevant = set(expected_ids)
    return len(top_k & relevant) / len(relevant)


def mean_reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """
    MRR = 1 / rank_of_first_relevant_result
    Returns 0.0 if no expected source found.
    """
    expected_set = set(expected_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """
    NDCG@K with binary relevance (1 if in expected_ids, 0 otherwise).
    """
    if not expected_ids:
        return 1.0
    expected_set = set(expected_ids)

    # DCG@K
    dcg = 0.0
    for i in range(min(k, len(retrieved_ids))):
        rel = 1.0 if retrieved_ids[i] in expected_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # IDCG@K: best possible — all relevant docs at top
    n_relevant = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """1.0 if any expected source appears in top-K, else 0.0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(expected_ids) else 0.0


def compute_retrieval_metrics(
    results: list[dict],
    expected_sources: list[str],
    acceptable_sources: list[str] | None = None,
    k: int = 5,
) -> dict:
    """
    Compute all retrieval metrics at once.

    Uses acceptable_sources for precision (broader set),
    expected_sources for recall (strict set).
    """
    retrieved_ids = extract_paper_ids(results)
    precision_ids = acceptable_sources if acceptable_sources else expected_sources

    return {
        "precision_at_k": precision_at_k(retrieved_ids, precision_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, expected_sources, k),
        "mrr": mean_reciprocal_rank(retrieved_ids, expected_sources),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, expected_sources, k),
        "hit_rate": hit_rate(retrieved_ids, expected_sources, k),
        "retrieved_papers": retrieved_ids,
        "expected_papers": list(expected_sources),
        "k": k,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2b. Citation Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════


def extract_cited_source_numbers(answer: str) -> list[int]:
    """
    Parse citation references from the answer, preferring the body over the footer.

    Handles three formats:
      - Standard inline:  [Source N]  (instruction-following models)
      - Fallback inline:  [N]         (models that use numeric footnote style)
      - Footer-only:      Source: [Source 1] ... (models that only list at the end)

    When a model only cites in a trailing sources block (no inline citations),
    we count those footer references so the metric is not misleadingly zero.

    Returns sorted, deduplicated list of source numbers.
    """
    parts = re.split(r"\n\s*(?:\[Source[^\]]*\]:|Sources?:)", answer, maxsplit=1)
    body = parts[0]
    footer = parts[1] if len(parts) > 1 else ""

    # Standard inline format: [Source N]
    standard = re.findall(r"\[Source\s+(\d+)\]", body)
    if standard:
        return sorted(set(int(m) for m in standard))

    # Fallback: bare [N] in body — limit to 1-20
    bare = re.findall(r"\[(\d{1,2})\]", body)
    if bare:
        return sorted(set(int(m) for m in bare if 1 <= int(m) <= 20))

    # Last resort: model only cited in footer — still credit those references
    footer_standard = re.findall(r"\[Source\s+(\d+)\]", footer)
    if footer_standard:
        return sorted(set(int(m) for m in footer_standard))

    footer_bare = re.findall(r"\[(\d{1,2})\]", footer)
    return sorted(set(int(m) for m in footer_bare if 1 <= int(m) <= 20))


def _source_number_to_paper_id(source_num: int, results: list[dict]) -> str | None:
    """Map a 1-based source number to a paper ID from the results list."""
    idx = source_num - 1
    if 0 <= idx < len(results):
        r = results[idx]
        raw = r.get("arxiv_id") or r.get("paper_id") or ""
        if not raw:
            metadata = r.get("metadata", {})
            if isinstance(metadata, dict):
                raw = metadata.get("arxiv_id") or metadata.get("paper_id") or ""
        return _normalize_arxiv_id(raw)
    return None


def citation_precision(
    answer: str,
    results: list[dict],
    expected_sources: list[str],
    acceptable_sources: list[str] | None = None,
) -> float:
    """
    Of the sources cited in the answer, what fraction are from expected/acceptable papers?
    """
    cited_nums = extract_cited_source_numbers(answer)
    if not cited_nums:
        return 0.0

    valid_set = set(acceptable_sources) if acceptable_sources else set(expected_sources)
    correct = 0
    for num in cited_nums:
        pid = _source_number_to_paper_id(num, results)
        if pid and pid in valid_set:
            correct += 1
    return correct / len(cited_nums)


def citation_recall(
    answer: str,
    results: list[dict],
    expected_sources: list[str],
) -> float:
    """
    Of the expected source papers, what fraction are actually cited in the answer?
    """
    if not expected_sources:
        return 1.0
    cited_nums = extract_cited_source_numbers(answer)
    cited_papers = set()
    for num in cited_nums:
        pid = _source_number_to_paper_id(num, results)
        if pid:
            cited_papers.add(pid)
    expected_set = set(expected_sources)
    return len(cited_papers & expected_set) / len(expected_set)


def source_coverage(answer: str, num_sources_provided: int) -> float:
    """
    What fraction of the provided sources are actually cited?
    """
    if num_sources_provided <= 0:
        return 0.0
    cited = extract_cited_source_numbers(answer)
    return len(cited) / num_sources_provided


# Matches any inline citation: [Source N] or bare [N] (1-20)
_CITATION_PAT = re.compile(r"\[Source\s+\d+\]|\[(\d{1,2})\]")


def _strip_sources_block(answer: str) -> str:
    """Remove trailing sources/references block from answer text."""
    return re.split(r"\n\s*(?:\[Source[^\]]*\]:|Sources?:)", answer, maxsplit=1)[0]


def _has_citation(text: str) -> bool:
    """Return True if text contains any inline citation in either supported format."""
    return bool(_CITATION_PAT.search(text))


def citation_density(answer: str) -> float:
    """
    Average number of citations per sentence.
    Handles both [Source N] and bare [N] citation formats.
    """
    text = _strip_sources_block(answer)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return 0.0
    total_citations = len(_CITATION_PAT.findall(text))
    return total_citations / len(sentences)


def count_unsupported_claims(answer: str) -> int:
    """
    Count sentences that make specific factual claims but have no citation.

    A sentence is considered an unsupported claim when it contains BOTH:
      - a number or measurement (e.g. "175 billion", "0.85", "28 BLEU")
      - a technical term indicating it's making a verifiable claim

    Named entities alone are NOT flagged — headlines, topic sentences, and
    introductory statements routinely name things without needing citations.
    Excludes the sources block and sentences shorter than 25 characters.
    """
    text = _strip_sources_block(answer)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    _TECHNICAL = re.compile(
        r"\b(?:parameter|layer|architecture|training|dataset|performance|accuracy|"
        r"score|benchmark|attention|embedding|token|transformer|neural|matrix|vector|"
        r"gradient|loss|weight|dimension|head|softmax|linear|epoch|batch|"
        r"precision|recall|f1|bleu|rouge|perplexity|latency|throughput)\b",
        re.IGNORECASE,
    )
    _NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?(?:\s*(?:billion|million|thousand|%|ms|gb|tb))?\b")

    unsupported = 0
    for sentence in sentences:
        if len(sentence) < 25:
            continue
        if _has_citation(sentence):
            continue
        # Only flag sentences that make measurable/verifiable claims
        if _NUMBER.search(sentence) and _TECHNICAL.search(sentence):
            unsupported += 1

    return unsupported


def detect_hallucinations(answer: str, expected_excludes: list[str]) -> list[str]:
    """
    Check if any hallucination canary terms appear in the answer.
    Returns list of found exclusion terms.
    """
    found = []
    answer_lower = answer.lower()
    for term in expected_excludes:
        if term.lower() in answer_lower:
            found.append(term)
    return found


def compute_citation_metrics(
    answer: str,
    results: list[dict],
    expected_sources: list[str],
    acceptable_sources: list[str] | None = None,
    expected_excludes: list[str] | None = None,
) -> dict:
    """Compute all citation metrics at once."""
    cited_nums = extract_cited_source_numbers(answer)
    cited_paper_ids = []
    for num in cited_nums:
        pid = _source_number_to_paper_id(num, results)
        if pid:
            cited_paper_ids.append(pid)

    return {
        "precision": citation_precision(answer, results, expected_sources, acceptable_sources),
        "recall": citation_recall(answer, results, expected_sources),
        "source_coverage": source_coverage(answer, len(results)),
        "density": citation_density(answer),
        "unsupported_claims": count_unsupported_claims(answer),
        "hallucination_flags": detect_hallucinations(answer, expected_excludes or []),
        "cited_sources": cited_nums,
        "cited_paper_ids": cited_paper_ids,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2c. Completeness Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_completeness(
    answer: str,
    expected_contains: list[str],
) -> dict:
    """
    Check what fraction of expected keywords appear in the answer.
    Case-insensitive substring matching.
    """
    if not expected_contains:
        return {
            "expected_keywords_found": 0,
            "expected_keywords_total": 0,
            "score": 1.0,
            "missing_keywords": [],
        }

    answer_lower = answer.lower()
    found = 0
    missing = []
    for keyword in expected_contains:
        if keyword.lower() in answer_lower:
            found += 1
        else:
            missing.append(keyword)

    return {
        "expected_keywords_found": found,
        "expected_keywords_total": len(expected_contains),
        "score": found / len(expected_contains),
        "missing_keywords": missing,
    }
