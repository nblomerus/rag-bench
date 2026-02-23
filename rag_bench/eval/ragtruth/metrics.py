"""
RAGTruth evaluation metrics.

Implements:
- Hallucination rate: Fraction of responses containing hallucinations.
- Span-level F1: Token-level overlap between predicted and annotated hallucination spans.
- Case-level accuracy: Binary classification accuracy (hallucinated vs not).
- Hallucination by type: Breakdown by hallucination category.

All functions are pure and deterministic — no LLM calls.
"""

import logging
import re

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def _extract_spans_from_text(text: str, span_texts: list[str]) -> list[tuple[int, int]]:
    """Find character positions of span texts within the full text."""
    positions = []
    text_lower = text.lower()
    for span_text in span_texts:
        span_lower = span_text.lower().strip()
        if not span_lower:
            continue
        start = text_lower.find(span_lower)
        if start >= 0:
            positions.append((start, start + len(span_lower)))
    return positions


def compute_hallucination_rate(results: list[dict]) -> dict:
    """
    Compute hallucination rate across a batch of results.

    Each result should have:
        - 'has_hallucination_predicted': bool (our system's prediction)
        - 'has_hallucination_gold': bool (ground truth)

    Returns:
        Dict with predicted_rate, gold_rate, and detection stats.
    """
    if not results:
        return {"predicted_rate": 0.0, "gold_rate": 0.0, "total": 0}

    predicted_positives = sum(1 for r in results if r.get("has_hallucination_predicted", False))
    gold_positives = sum(1 for r in results if r.get("has_hallucination_gold", False))
    total = len(results)

    return {
        "predicted_rate": round(predicted_positives / total, 4) if total else 0.0,
        "gold_rate": round(gold_positives / total, 4) if total else 0.0,
        "predicted_count": predicted_positives,
        "gold_count": gold_positives,
        "total": total,
    }


def span_level_f1(
    predicted_spans: list[str],
    gold_spans: list[str],
    full_text: str = "",
) -> dict:
    """
    Compute span-level F1 between predicted and gold hallucination spans.

    Uses token-level overlap for fuzzy matching since exact character
    positions may differ between systems.

    Args:
        predicted_spans: List of predicted hallucinated text segments.
        gold_spans: List of ground-truth hallucinated text segments.
        full_text: Full response text (used for positional matching if available).

    Returns:
        Dict with span_f1, span_precision, span_recall.
    """
    if not predicted_spans and not gold_spans:
        return {"span_f1": 1.0, "span_precision": 1.0, "span_recall": 1.0}

    if not predicted_spans:
        return {"span_f1": 0.0, "span_precision": 0.0, "span_recall": 0.0}

    if not gold_spans:
        return {"span_f1": 0.0, "span_precision": 0.0, "span_recall": 1.0}

    # Token-based overlap approach
    pred_tokens = set()
    for span in predicted_spans:
        pred_tokens.update(_tokenize(span))

    gold_tokens = set()
    for span in gold_spans:
        gold_tokens.update(_tokenize(span))

    if not pred_tokens and not gold_tokens:
        return {"span_f1": 1.0, "span_precision": 1.0, "span_recall": 1.0}

    overlap = pred_tokens & gold_tokens
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(overlap) / len(gold_tokens) if gold_tokens else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "span_f1": round(f1, 4),
        "span_precision": round(precision, 4),
        "span_recall": round(recall, 4),
        "predicted_token_count": len(pred_tokens),
        "gold_token_count": len(gold_tokens),
        "overlap_token_count": len(overlap),
    }


def case_level_accuracy(results: list[dict]) -> dict:
    """
    Compute case-level binary classification accuracy.

    For each response, checks if the system correctly predicted whether
    hallucinations are present or absent.

    Each result should have:
        - 'has_hallucination_predicted': bool
        - 'has_hallucination_gold': bool

    Returns:
        Dict with accuracy, precision, recall, f1, and confusion counts.
    """
    if not results:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "total": 0}

    tp = fp = tn = fn = 0
    for r in results:
        predicted = r.get("has_hallucination_predicted", False)
        gold = r.get("has_hallucination_gold", False)
        if gold and predicted:
            tp += 1
        elif gold and not predicted:
            fn += 1
        elif not gold and predicted:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total": total,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def hallucination_by_type(results: list[dict]) -> dict:
    """
    Break down hallucination detections by type.

    RAGTruth categorizes hallucinations into 4 types:
    - Evident Conflict: Clear contradiction with source
    - Subtle Conflict: Nuanced contradiction
    - Evident Baseless: Clearly unsupported claim
    - Subtle Baseless: Plausible but unsupported claim

    Each result should have:
        - 'gold_span_types': list of str (label types from gold annotations)
        - 'predicted_span_types': list of str (if available)

    Returns:
        Dict with counts and rates per type.
    """
    type_counts = {
        "Evident Conflict": {"gold": 0, "predicted": 0},
        "Subtle Conflict": {"gold": 0, "predicted": 0},
        "Evident Baseless": {"gold": 0, "predicted": 0},
        "Subtle Baseless": {"gold": 0, "predicted": 0},
        "Other": {"gold": 0, "predicted": 0},
    }

    total_gold = 0
    total_predicted = 0

    for r in results:
        for span_type in r.get("gold_span_types", []):
            normalized = _normalize_type(span_type)
            type_counts[normalized]["gold"] += 1
            total_gold += 1

        for span_type in r.get("predicted_span_types", []):
            normalized = _normalize_type(span_type)
            type_counts[normalized]["predicted"] += 1
            total_predicted += 1

    # Compute rates
    breakdown = {}
    for type_name, counts in type_counts.items():
        if counts["gold"] == 0 and counts["predicted"] == 0:
            continue
        breakdown[type_name] = {
            "gold_count": counts["gold"],
            "predicted_count": counts["predicted"],
            "gold_rate": round(counts["gold"] / total_gold, 4) if total_gold else 0.0,
        }

    return {
        "breakdown": breakdown,
        "total_gold_spans": total_gold,
        "total_predicted_spans": total_predicted,
    }


def _normalize_type(span_type: str) -> str:
    """Normalize hallucination type string to standard categories."""
    t = span_type.strip().lower()
    if "evident" in t and "conflict" in t:
        return "Evident Conflict"
    elif "subtle" in t and "conflict" in t:
        return "Subtle Conflict"
    elif "evident" in t and "baseless" in t:
        return "Evident Baseless"
    elif "subtle" in t and "baseless" in t:
        return "Subtle Baseless"
    elif "conflict" in t:
        return "Evident Conflict"
    elif "baseless" in t:
        return "Evident Baseless"
    else:
        return "Other"
