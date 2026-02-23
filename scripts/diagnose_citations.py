#!/usr/bin/env python3
"""Diagnose why citation precision and recall are low.

Loads the latest evaluation results and produces a detailed breakdown
of citation failures: what the LLM cited vs. what was expected, where
the expected paper ranked, and the most common failure patterns.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Locate the latest eval file ──────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval_results"


def find_latest_eval() -> Path:
    files = sorted(EVAL_DIR.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("No eval result files found in", EVAL_DIR)
        sys.exit(1)
    return files[-1]


def load_eval(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Helpers ──────────────────────────────────────────────────────────────────


def fmt_pct(n: float) -> str:
    return f"{n * 100:.1f}%"


def truncate(s: str, n: int = 120) -> str:
    return s[:n] + "…" if len(s) > n else s


# ── Analysis ─────────────────────────────────────────────────────────────────


def analyse(data: dict) -> None:
    results = data["results"]

    # Filter to non-deflection, non-error entries that have citation data
    entries = [
        r
        for r in results
        if not r.get("deflection", {}).get("expected", False) and not r.get("error") and r.get("citation")
    ]

    total = len(entries)
    print(f"\n{'=' * 80}")
    print(f"  CITATION DIAGNOSIS — {total} answerable queries")
    print(f"{'=' * 80}\n")

    # ── 1. Overall stats ─────────────────────────────────────────────────────
    prec_values = [r["citation"]["precision"] for r in entries]
    rec_values = [r["citation"]["recall"] for r in entries]
    avg_prec = sum(prec_values) / len(prec_values) if prec_values else 0
    avg_rec = sum(rec_values) / len(rec_values) if rec_values else 0

    perfect_prec = sum(1 for v in prec_values if v == 1.0)
    zero_prec = sum(1 for v in prec_values if v == 0.0)
    partial_prec = total - perfect_prec - zero_prec

    perfect_rec = sum(1 for v in rec_values if v == 1.0)
    zero_rec = sum(1 for v in rec_values if v == 0.0)
    partial_rec = total - perfect_rec - zero_rec

    print("1. OVERALL DISTRIBUTION")
    print(
        f"   Citation Precision: {fmt_pct(avg_prec)}  (perfect={perfect_prec}, partial={partial_prec}, zero={zero_prec})"
    )
    print(f"   Citation Recall:    {fmt_pct(avg_rec)}  (perfect={perfect_rec}, partial={partial_rec}, zero={zero_rec})")
    print()

    # ── 2. Classify each entry ───────────────────────────────────────────────
    # Failure categories:
    #   A. Expected paper NOT in retrieved set at all (retrieval miss)
    #   B. Expected paper retrieved but NOT cited (LLM ignored it)
    #   C. Expected paper cited but extra wrong citations too (over-citation)
    #   D. Correct — expected paper cited, no wrong citations

    categories = {
        "retrieval_miss": [],  # Expected paper not in top-5
        "not_cited": [],  # Retrieved but LLM didn't cite it
        "wrong_source_num": [],  # LLM cited a different source number
        "over_citation": [],  # Correct citation + extra wrong ones
        "correct": [],  # Perfect citation
    }

    for r in entries:
        cit = r["citation"]
        ret = r.get("retrieval", {})
        expected = set(ret.get("expected_papers", []))
        retrieved = ret.get("retrieved_papers", [])
        cited_papers = set(cit.get("cited_paper_ids", []))
        cited_sources = cit.get("cited_sources", [])
        prec = cit["precision"]
        rec = cit["recall"]

        # Check if expected papers are in the retrieved set
        expected_in_retrieved = expected & set(retrieved)
        expected_missing = expected - set(retrieved)

        if expected_missing and not expected_in_retrieved:
            categories["retrieval_miss"].append(r)
        elif rec == 0.0 and expected_in_retrieved:
            # Expected paper was retrieved but not cited
            categories["not_cited"].append(r)
        elif prec == 1.0 and rec == 1.0:
            categories["correct"].append(r)
        elif rec > 0 and prec < 1.0:
            categories["over_citation"].append(r)
        elif rec == 0.0 and not expected_missing:
            categories["wrong_source_num"].append(r)
        else:
            # Partial cases
            if rec == 0.0:
                categories["not_cited"].append(r)
            else:
                categories["over_citation"].append(r)

    print("2. FAILURE CATEGORY BREAKDOWN")
    print(f"   {'Category':<30} {'Count':>5}  {'% of total':>10}")
    print(f"   {'-' * 50}")
    for cat, items in categories.items():
        label = {
            "retrieval_miss": "A. Retrieval miss",
            "not_cited": "B. Retrieved but not cited",
            "wrong_source_num": "C. Wrong source number",
            "over_citation": "D. Over-citation (extra wrong)",
            "correct": "E. Correct",
        }[cat]
        print(f"   {label:<30} {len(items):>5}  {fmt_pct(len(items) / total):>10}")
    print()

    # ── 3. Position analysis ─────────────────────────────────────────────────
    print("3. EXPECTED PAPER POSITION vs. CITATION RATE")
    print("   (Where does the expected paper rank in the retrieved list?)\n")

    position_cited = defaultdict(lambda: [0, 0])  # [cited_count, total_count]
    for r in entries:
        ret = r.get("retrieval", {})
        cit = r["citation"]
        expected = set(ret.get("expected_papers", []))
        retrieved = ret.get("retrieved_papers", [])
        cited_papers = set(cit.get("cited_paper_ids", []))

        for ep in expected:
            if ep in retrieved:
                pos = retrieved.index(ep) + 1  # 1-based
                was_cited = ep in cited_papers
                position_cited[pos][1] += 1
                if was_cited:
                    position_cited[pos][0] += 1
            else:
                position_cited["not_found"][1] += 1
                if ep in cited_papers:
                    position_cited["not_found"][0] += 1

    print(f"   {'Position':<12} {'Cited':>6} / {'Total':>6}  {'Rate':>8}")
    print(f"   {'-' * 40}")
    for pos in sorted(k for k in position_cited if isinstance(k, int)):
        cited, tot = position_cited[pos]
        rate = cited / tot if tot else 0
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"   Position {pos:<3} {cited:>6} / {tot:>6}  {fmt_pct(rate):>8}  {bar}")
    if "not_found" in position_cited:
        cited, tot = position_cited["not_found"]
        rate = cited / tot if tot else 0
        print(f"   {'Not in top5':<12} {cited:>6} / {tot:>6}  {fmt_pct(rate):>8}")
    print()

    # ── 4. What does the LLM actually cite? ──────────────────────────────────
    print("4. LLM CITATION PATTERNS")

    cite_counts = Counter()
    for r in entries:
        for src in r["citation"].get("cited_sources", []):
            cite_counts[src] += 1

    print("   Source numbers the LLM cites most often:")
    print(f"   {'Source #':<10} {'Times cited':>12}  {'% of entries':>12}")
    print(f"   {'-' * 40}")
    for src, count in cite_counts.most_common(10):
        print(f"   Source {src:<4} {count:>12}  {fmt_pct(count / total):>12}")
    print()

    num_citations = [len(r["citation"].get("cited_sources", [])) for r in entries]
    avg_num = sum(num_citations) / len(num_citations) if num_citations else 0
    print(f"   Avg citations per answer: {avg_num:.1f}")
    dist = Counter(num_citations)
    for n in sorted(dist):
        print(f"     {n} citations: {dist[n]} answers ({fmt_pct(dist[n] / total)})")
    print()

    # ── 5. Detailed examples of each failure category ────────────────────────
    print("5. DETAILED FAILURE EXAMPLES")
    print("=" * 80)

    for cat, items in categories.items():
        if cat == "correct" or not items:
            continue

        label = {
            "retrieval_miss": "A. RETRIEVAL MISS — expected paper not in top 5",
            "not_cited": "B. NOT CITED — expected paper retrieved but LLM ignored it",
            "wrong_source_num": "C. WRONG SOURCE — LLM cited wrong source number",
            "over_citation": "D. OVER-CITATION — correct + extra wrong citations",
        }[cat]

        print(f"\n{'─' * 80}")
        print(f"  {label}")
        print(f"  ({len(items)} entries — showing up to 5)")
        print(f"{'─' * 80}")

        for r in items[:5]:
            ret = r.get("retrieval", {})
            cit = r["citation"]
            expected = ret.get("expected_papers", [])
            retrieved = ret.get("retrieved_papers", [])
            cited_papers = cit.get("cited_paper_ids", [])
            cited_sources = cit.get("cited_sources", [])

            print(f"\n  ID: {r['id']}")
            print(f"  Q:  {truncate(r['question'], 100)}")
            print(f"  Topic: {r.get('topic', '?')}  |  Difficulty: {r.get('difficulty', '?')}")
            print(f"  Expected papers:  {expected}")
            print(f"  Retrieved papers: {retrieved}")

            # Show position of expected papers
            for ep in expected:
                if ep in retrieved:
                    pos = retrieved.index(ep) + 1
                    print(f"    → Expected '{ep}' at position {pos}")
                else:
                    print(f"    → Expected '{ep}' NOT in retrieved set")

            print(f"  LLM cited sources: {cited_sources} → papers: {cited_papers}")
            print(f"  Precision: {fmt_pct(cit['precision'])}  |  Recall: {fmt_pct(cit['recall'])}")
            print(f"  Answer: {truncate(r.get('answer_preview', ''), 150)}")

    # ── 6. Topic-level breakdown ─────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("6. CITATION PRECISION BY TOPIC")
    print(f"{'=' * 80}\n")

    topic_stats = defaultdict(lambda: {"prec": [], "rec": [], "count": 0})
    for r in entries:
        topic = r.get("topic", "unknown")
        topic_stats[topic]["prec"].append(r["citation"]["precision"])
        topic_stats[topic]["rec"].append(r["citation"]["recall"])
        topic_stats[topic]["count"] += 1

    print(f"   {'Topic':<25} {'Count':>5}  {'Avg Prec':>10}  {'Avg Recall':>10}  {'Zero Prec':>10}")
    print(f"   {'-' * 65}")
    for topic in sorted(topic_stats, key=lambda t: sum(topic_stats[t]["prec"]) / len(topic_stats[t]["prec"])):
        s = topic_stats[topic]
        avg_p = sum(s["prec"]) / len(s["prec"])
        avg_r = sum(s["rec"]) / len(s["rec"])
        zeros = sum(1 for v in s["prec"] if v == 0.0)
        print(f"   {topic:<25} {s['count']:>5}  {fmt_pct(avg_p):>10}  {fmt_pct(avg_r):>10}  {zeros:>10}")

    # ── 7. Difficulty-level breakdown ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("7. CITATION PRECISION BY DIFFICULTY")
    print(f"{'=' * 80}\n")

    diff_stats = defaultdict(lambda: {"prec": [], "rec": [], "count": 0})
    for r in entries:
        diff = r.get("difficulty", "unknown")
        diff_stats[diff]["prec"].append(r["citation"]["precision"])
        diff_stats[diff]["rec"].append(r["citation"]["recall"])
        diff_stats[diff]["count"] += 1

    print(f"   {'Difficulty':<15} {'Count':>5}  {'Avg Prec':>10}  {'Avg Recall':>10}")
    print(f"   {'-' * 45}")
    for diff in ["easy", "medium", "hard"]:
        if diff in diff_stats:
            s = diff_stats[diff]
            avg_p = sum(s["prec"]) / len(s["prec"])
            avg_r = sum(s["rec"]) / len(s["rec"])
            print(f"   {diff:<15} {s['count']:>5}  {fmt_pct(avg_p):>10}  {fmt_pct(avg_r):>10}")

    # ── 8. Query-type breakdown ──────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("8. CITATION PRECISION BY QUERY TYPE")
    print(f"{'=' * 80}\n")

    qt_stats = defaultdict(lambda: {"prec": [], "rec": [], "count": 0})
    for r in entries:
        qt = r.get("query_type", "unknown")
        qt_stats[qt]["prec"].append(r["citation"]["precision"])
        qt_stats[qt]["rec"].append(r["citation"]["recall"])
        qt_stats[qt]["count"] += 1

    print(f"   {'Query Type':<20} {'Count':>5}  {'Avg Prec':>10}  {'Avg Recall':>10}")
    print(f"   {'-' * 50}")
    for qt in sorted(qt_stats):
        s = qt_stats[qt]
        avg_p = sum(s["prec"]) / len(s["prec"])
        avg_r = sum(s["rec"]) / len(s["rec"])
        print(f"   {qt:<20} {s['count']:>5}  {fmt_pct(avg_p):>10}  {fmt_pct(avg_r):>10}")

    # ── 9. Root cause summary ────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("9. ROOT CAUSE SUMMARY")
    print(f"{'=' * 80}\n")

    retrieval_miss_n = len(categories["retrieval_miss"])
    not_cited_n = len(categories["not_cited"])
    wrong_src_n = len(categories["wrong_source_num"])
    over_cite_n = len(categories["over_citation"])
    correct_n = len(categories["correct"])
    failed_n = total - correct_n

    print(f"   Total answerable queries:   {total}")
    print(f"   Correct citations:          {correct_n} ({fmt_pct(correct_n / total)})")
    print(f"   Failed citations:           {failed_n} ({fmt_pct(failed_n / total)})")
    print()

    if failed_n > 0:
        print(f"   Failure breakdown (% of {failed_n} failures):")
        print(
            f"     Retrieval miss:           {retrieval_miss_n:>3} ({fmt_pct(retrieval_miss_n / failed_n)}) "
            f"— paper not retrieved at all"
        )
        print(
            f"     Not cited (LLM ignored):  {not_cited_n:>3} ({fmt_pct(not_cited_n / failed_n)}) "
            f"— paper retrieved but LLM didn't cite it"
        )
        print(
            f"     Wrong source number:      {wrong_src_n:>3} ({fmt_pct(wrong_src_n / failed_n)}) "
            f"— LLM cited wrong [Source N]"
        )
        print(
            f"     Over-citation:            {over_cite_n:>3} ({fmt_pct(over_cite_n / failed_n)}) "
            f"— cited correct + extra wrong sources"
        )
    print()

    # Position insight
    if 1 in position_cited and position_cited[1][1] > 0:
        p1_rate = position_cited[1][0] / position_cited[1][1]
        other_cited = sum(v[0] for k, v in position_cited.items() if isinstance(k, int) and k > 1)
        other_total = sum(v[1] for k, v in position_cited.items() if isinstance(k, int) and k > 1)
        other_rate = other_cited / other_total if other_total > 0 else 0
        print("   Position effect:")
        print(f"     Expected paper at position 1 → cited {fmt_pct(p1_rate)} of the time")
        print(f"     Expected paper at position 2-5 → cited {fmt_pct(other_rate)} of the time")
        print("     → LLM strongly prefers Source 1. Getting expected paper to position 1 is key.")
    print()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_eval()

    print(f"Loading: {path.name}")
    data = load_eval(path)
    analyse(data)


if __name__ == "__main__":
    main()
