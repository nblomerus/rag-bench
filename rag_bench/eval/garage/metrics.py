"""
GaRAGe evaluation metrics.

Implements:
- RAF (Relevance-Aware Factuality): Measures factuality weighted by passage relevance.
- uRAF (Unweighted RAF): Equal-weight version.
- Attribution F1: Token-level attribution correctness.
- Deflection metrics: Whether the system correctly declines unanswerable questions.

All functions are pure and deterministic — no LLM calls.
"""

import logging
import re

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def _token_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level F1 between two texts."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    precision = len(overlap) / len(tokens_a) if tokens_a else 0.0
    recall = len(overlap) / len(tokens_b) if tokens_b else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_raf(
    answer: str,
    passages: list[dict],
    gold_answer: str = "",
) -> dict:
    """
    Compute Relevance-Aware Factuality (RAF).

    RAF measures how well the generated answer aligns with relevant passages,
    penalizing reliance on irrelevant passages.

    Args:
        answer: Generated answer text.
        passages: List of dicts with 'text' and 'is_relevant' keys.
        gold_answer: Reference answer (used for additional comparison).

    Returns:
        Dict with raf_score, relevant_overlap, irrelevant_overlap, and details.
    """
    if not answer or not passages:
        return {"raf_score": 0.0, "relevant_overlap": 0.0, "irrelevant_overlap": 0.0}

    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return {"raf_score": 0.0, "relevant_overlap": 0.0, "irrelevant_overlap": 0.0}

    relevant_tokens = set()
    irrelevant_tokens = set()

    for p in passages:
        p_tokens = set(_tokenize(p.get("text", "")))
        if p.get("is_relevant", False):
            relevant_tokens |= p_tokens
        else:
            irrelevant_tokens |= p_tokens

    # Tokens in the answer that come from relevant vs irrelevant passages
    from_relevant = answer_tokens & relevant_tokens
    from_irrelevant = answer_tokens & irrelevant_tokens - relevant_tokens  # exclusive to irrelevant

    relevant_overlap = len(from_relevant) / len(answer_tokens) if answer_tokens else 0.0
    irrelevant_overlap = len(from_irrelevant) / len(answer_tokens) if answer_tokens else 0.0

    # RAF = relevant_overlap - irrelevant_overlap, clipped to [0, 1]
    raf_score = max(0.0, min(1.0, relevant_overlap - irrelevant_overlap))

    # Boost if answer also matches gold answer well
    if gold_answer:
        gold_f1 = _token_overlap(answer, gold_answer)
        raf_score = 0.7 * raf_score + 0.3 * gold_f1

    return {
        "raf_score": round(raf_score, 4),
        "relevant_overlap": round(relevant_overlap, 4),
        "irrelevant_overlap": round(irrelevant_overlap, 4),
    }


def compute_uraf(
    answer: str,
    passages: list[dict],
    gold_answer: str = "",
) -> dict:
    """
    Compute Unweighted RAF (uRAF).

    Treats all passages equally regardless of relevance annotation.
    Useful as a baseline comparison to RAF.

    Args:
        answer: Generated answer text.
        passages: List of dicts with 'text' key.
        gold_answer: Reference answer.

    Returns:
        Dict with uraf_score and passage_overlap.
    """
    if not answer or not passages:
        return {"uraf_score": 0.0, "passage_overlap": 0.0}

    answer_tokens = set(_tokenize(answer))
    all_passage_tokens = set()
    for p in passages:
        all_passage_tokens |= set(_tokenize(p.get("text", "")))

    if not answer_tokens:
        return {"uraf_score": 0.0, "passage_overlap": 0.0}

    overlap = answer_tokens & all_passage_tokens
    passage_overlap = len(overlap) / len(answer_tokens)

    uraf_score = passage_overlap
    if gold_answer:
        gold_f1 = _token_overlap(answer, gold_answer)
        uraf_score = 0.7 * passage_overlap + 0.3 * gold_f1

    return {
        "uraf_score": round(uraf_score, 4),
        "passage_overlap": round(passage_overlap, 4),
    }


def compute_attribution_f1(
    answer: str,
    passages: list[dict],
) -> dict:
    """
    Compute attribution F1 — how well the answer's content is attributed
    to the correct passages.

    Uses citation detection ([Source N], [N]) to map answer claims to passages,
    then computes precision and recall of attributions.

    Args:
        answer: Generated answer with citations.
        passages: List of dicts with 'text' and 'is_relevant' keys.

    Returns:
        Dict with attribution_f1, attribution_precision, attribution_recall.
    """
    if not answer or not passages:
        return {"attribution_f1": 0.0, "attribution_precision": 0.0, "attribution_recall": 0.0}

    # Extract citation numbers from the answer
    cited_indices = set()
    for m in re.finditer(r"\[(?:Source\s+)?(\d+)\]", answer):
        idx = int(m.group(1)) - 1  # Convert to 0-based
        if 0 <= idx < len(passages):
            cited_indices.add(idx)

    # Identify which passages are actually relevant
    relevant_indices = {i for i, p in enumerate(passages) if p.get("is_relevant", False)}

    if not cited_indices and not relevant_indices:
        return {"attribution_f1": 1.0, "attribution_precision": 1.0, "attribution_recall": 1.0}

    # Attribution precision: % of cited passages that are relevant
    if cited_indices:
        correct_citations = cited_indices & relevant_indices
        precision = len(correct_citations) / len(cited_indices)
    else:
        precision = 0.0

    # Attribution recall: % of relevant passages that are cited
    if relevant_indices:
        correct_citations = cited_indices & relevant_indices
        recall = len(correct_citations) / len(relevant_indices)
    else:
        recall = 1.0  # No relevant passages → nothing to recall

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "attribution_f1": round(f1, 4),
        "attribution_precision": round(precision, 4),
        "attribution_recall": round(recall, 4),
        "cited_count": len(cited_indices),
        "relevant_count": len(relevant_indices),
    }


def compute_deflection_metrics(
    results: list[dict],
) -> dict:
    """
    Compute deflection metrics across a batch of results.

    Each result should have:
        - 'should_deflect': bool (ground truth)
        - 'did_deflect': bool (system behavior)

    Returns:
        Dict with TPR, FPR, accuracy, and counts.
    """
    if not results:
        return {
            "deflection_tpr": 0.0,
            "deflection_fpr": 0.0,
            "deflection_accuracy": 0.0,
            "total": 0,
        }

    tp = fp = tn = fn = 0
    for r in results:
        should = r.get("should_deflect", False)
        did = r.get("did_deflect", False)
        if should and did:
            tp += 1
        elif should and not did:
            fn += 1
        elif not should and did:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "deflection_tpr": round(tpr, 4),
        "deflection_fpr": round(fpr, 4),
        "deflection_accuracy": round(accuracy, 4),
        "total": total,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }
