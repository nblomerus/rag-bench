#!/usr/bin/env python3
"""
scrape_arxiv.py — Download AI/ML research papers from ArXiv API.

Run this on your server (not in a sandbox). It will:
1. Query ArXiv for papers across 10 AI/ML categories
2. Download full PDF text via arxiv2text or fallback to abstracts
3. Save everything as JSON ready for the RAG pipeline

Usage:
    pip install arxiv pymupdf requests tqdm
    python scrape_arxiv.py                          # Default: ~50 landmark papers
    python scrape_arxiv.py --mode extended           # ~500 papers across all topics
    python scrape_arxiv.py --mode abstracts           # ~2000 abstracts only (fast)
    python scrape_arxiv.py --output /path/to/data     # Custom output directory
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import arxiv
import pymupdf
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Landmark papers — the core 50 that every ML engineer should know
# ═══════════════════════════════════════════════════════════════════════════
LANDMARK_PAPERS = [
    # Transformer Architecture
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3 (Language Models are Few-Shot Learners)
    # Scaling Laws & Training
    "2001.08361",  # Scaling Laws for Neural Language Models (Kaplan)
    "2203.15556",  # Chinchilla (Training Compute-Optimal LLMs)
    "2303.08774",  # GPT-4 Technical Report
    # Alignment & RLHF
    "2203.02155",  # InstructGPT
    "2212.08073",  # Constitutional AI
    "2305.18290",  # DPO (Direct Preference Optimization)
    # Efficient Fine-Tuning
    "2106.09685",  # LoRA
    "2305.14314",  # QLoRA
    "2101.00190",  # Prefix Tuning
    # Retrieval-Augmented Generation
    "2005.11401",  # RAG (Lewis et al.)
    "2002.08909",  # REALM
    "2112.04426",  # RETRO
    "2208.03299",  # Atlas
    # Long Context & Memory
    "2104.09864",  # RoPE (RoFormer)
    "2108.12409",  # ALiBi (Train Short, Test Long)
    "2310.01889",  # Ring Attention
    "2312.00752",  # Mamba
    # Diffusion Models
    "2006.11239",  # DDPM
    "2112.10752",  # Latent Diffusion / Stable Diffusion
    "2207.12598",  # Classifier-Free Guidance
    # Multi-Modal Models
    "2103.00020",  # CLIP
    "2304.08485",  # LLaVA
    "2204.14198",  # Flamingo
    # Mixture of Experts
    "2101.03961",  # Switch Transformer
    "2401.04088",  # Mixtral
    # Agents & Reasoning
    "2210.03629",  # ReAct
    "2302.04761",  # Toolformer
    "2201.11903",  # Chain-of-Thought Prompting
    "2305.10601",  # Tree of Thoughts
    # Additional important papers
    "2307.09288",  # Llama 2
    "2205.14135",  # FlashAttention
    "2307.03172",  # Lost in the Middle
    "2401.18059",  # RAPTOR (Hierarchical RAG)
    "2307.08621",  # RetNet (Retentive Network)
]

# ═══════════════════════════════════════════════════════════════════════════
# Extended search queries for broader coverage
# ═══════════════════════════════════════════════════════════════════════════
SEARCH_TOPICS = {
    "transformers": {
        "query": "cat:cs.CL AND (transformer architecture OR attention mechanism OR self-attention)",
        "max_results": 60,
    },
    "scaling_laws": {
        "query": "cat:cs.LG AND (scaling laws OR compute optimal training OR emergent abilities)",
        "max_results": 40,
    },
    "alignment_rlhf": {
        "query": "cat:cs.CL AND (RLHF OR reinforcement learning human feedback OR preference optimization OR alignment)",
        "max_results": 50,
    },
    "efficient_finetuning": {
        "query": "cat:cs.CL AND (LoRA OR parameter efficient fine tuning OR adapter OR quantized fine tuning)",
        "max_results": 40,
    },
    "rag_retrieval": {
        "query": "cat:cs.CL AND (retrieval augmented generation OR dense passage retrieval OR grounded generation)",
        "max_results": 50,
    },
    "long_context": {
        "query": "cat:cs.CL AND (long context OR positional encoding OR state space model OR linear attention)",
        "max_results": 40,
    },
    "diffusion": {
        "query": "cat:cs.LG AND (diffusion model OR denoising score matching OR latent diffusion)",
        "max_results": 40,
    },
    "multimodal": {
        "query": "cat:cs.CV AND (vision language model OR multimodal OR contrastive learning CLIP)",
        "max_results": 40,
    },
    "mixture_of_experts": {
        "query": "cat:cs.LG AND (mixture of experts OR sparse MoE OR expert routing)",
        "max_results": 30,
    },
    "agents_reasoning": {
        "query": "cat:cs.CL AND (LLM agent OR tool use OR chain of thought OR reasoning)",
        "max_results": 50,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PDF download and text extraction
# ═══════════════════════════════════════════════════════════════════════════
def download_pdf(arxiv_id: str, output_dir: Path) -> Path | None:
    """Download a PDF from ArXiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = output_dir / f"{arxiv_id.replace('/', '_')}.pdf"

    if pdf_path.exists():
        return pdf_path

    try:
        resp = requests.get(pdf_url, timeout=30, stream=True)
        resp.raise_for_status()

        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return pdf_path
    except Exception as e:
        logger.warning(f"Failed to download PDF for {arxiv_id}: {e}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = pymupdf.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def extract_sections_from_text(text: str) -> dict[str, str]:
    """
    Split extracted paper text into sections based on common headers.
    Handles both markdown-style and plain text headers.
    """
    if not text or len(text.strip()) < 100:
        return {"full_text": text}

    sections = {}
    current_section = "preamble"
    current_lines = []

    # Patterns for section headers
    header_patterns = [
        re.compile(r"^#{1,4}\s+(.+)$"),  # Markdown: ## Header
        re.compile(r"^(\d+\.?\s+[A-Z][A-Za-z\s]+)$"),  # Numbered: 1. Introduction
        re.compile(r"^([A-Z][A-Z\s]{3,40})$"),  # ALL CAPS: INTRODUCTION
        re.compile(
            r"^(Abstract|Introduction|Related Work|Background|"
            r"Method(?:ology|s)?|Approach|Model|Architecture|"
            r"Experiment(?:s|al)?(?:\s+(?:Setup|Results))?|Results|"
            r"Discussion|Conclusion(?:s)?|Limitation(?:s)?|"
            r"Training|Evaluation|Analysis|Appendix)\s*$",
            re.IGNORECASE,
        ),  # Known section names
    ]

    for line in text.split("\n"):
        stripped = line.strip()
        matched_header = None

        for pattern in header_patterns:
            m = pattern.match(stripped)
            if m:
                matched_header = m.group(1) if m.lastindex else stripped
                break

        if matched_header and len(matched_header) < 80:
            # Save previous section
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text and len(section_text) > 30:
                    sections[current_section] = section_text

            # Normalize section name
            current_section = re.sub(r"^[\d.]+\s*", "", matched_header)
            current_section = re.sub(r"[^a-zA-Z0-9\s]", "", current_section)
            current_section = current_section.lower().strip()
            current_section = re.sub(r"\s+", "_", current_section) or "unnamed"
            current_lines = []
        else:
            current_lines.append(line)

    # Save final section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text and len(section_text) > 30:
            sections[current_section] = section_text

    return sections if sections else {"full_text": text}


def build_acronym_dict(text: str) -> dict[str, str]:
    """Extract acronym definitions from paper text."""
    acronyms = {}
    pattern = r"([A-Za-z][A-Za-z\s\-]{2,50})\s*\(([A-Z][A-Z0-9]{1,10})\)"
    for match in re.finditer(pattern, text):
        full_form = match.group(1).strip()
        acronym = match.group(2).strip()
        words = full_form.split()
        if len(words) >= 2:
            acronyms[acronym] = full_form
    return acronyms


def format_authors(authors: list[str], max_authors: int = 3) -> str:
    """Format author list for display."""
    if not authors:
        return "Unknown"
    last_names = []
    for author in authors[:max_authors]:
        parts = author.strip().split()
        if parts:
            last_names.append(parts[-1])
    if len(authors) > max_authors:
        return f"{last_names[0]} et al."
    elif len(last_names) == 1:
        return last_names[0]
    elif len(last_names) == 2:
        return f"{last_names[0]} and {last_names[1]}"
    else:
        return ", ".join(last_names[:-1]) + f", and {last_names[-1]}"


# ═══════════════════════════════════════════════════════════════════════════
# Main scraping functions
# ═══════════════════════════════════════════════════════════════════════════
def fetch_by_ids(
    arxiv_ids: list[str],
    download_pdfs: bool = True,
    pdf_dir: Path | None = None,
) -> list[dict]:
    """Fetch specific papers by ArXiv ID."""
    logger.info(f"Fetching {len(arxiv_ids)} papers by ID...")

    client = arxiv.Client(
        page_size=20,
        delay_seconds=3.0,  # Be nice to ArXiv API
        num_retries=3,
    )

    docs = []
    search = arxiv.Search(id_list=arxiv_ids)

    for result in tqdm(client.results(search), total=len(arxiv_ids), desc="Fetching papers"):
        arxiv_id = result.entry_id.split("/")[-1]
        # Remove version suffix if present (e.g., v1, v2)
        arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)

        full_text = ""
        sections = {}

        # Try to get full text from PDF
        if download_pdfs and pdf_dir:
            pdf_path = download_pdf(arxiv_id_clean, pdf_dir)
            if pdf_path:
                full_text = extract_text_from_pdf(pdf_path)
                if full_text:
                    sections = extract_sections_from_text(full_text)

        # Fall back to abstract if no full text
        if not sections:
            sections = {"abstract": result.summary.strip()}
            full_text = result.summary.strip()

        year = result.published.year if result.published else None

        doc = {
            "doc_id": f"arxiv_{arxiv_id_clean}",
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors],
            "year": year,
            "arxiv_id": arxiv_id_clean,
            "categories": result.categories,
            "pdf_url": result.pdf_url,
            "full_text": full_text,
            "sections": sections,
            "acronyms": build_acronym_dict(full_text),
        }
        docs.append(doc)

        # Rate limiting — ArXiv asks for 3s between requests
        time.sleep(1)

    logger.info(f"Fetched {len(docs)} papers by ID")
    return docs


def fetch_by_search(
    topics: dict,
    download_pdfs: bool = True,
    pdf_dir: Path | None = None,
) -> list[dict]:
    """Fetch papers by search query across topics."""
    client = arxiv.Client(
        page_size=50,
        delay_seconds=3.0,
        num_retries=3,
    )

    all_docs = []
    seen_ids = set()

    for topic_name, topic_config in topics.items():
        logger.info(f"Searching topic: {topic_name} (max {topic_config['max_results']})")

        search = arxiv.Search(
            query=topic_config["query"],
            max_results=topic_config["max_results"],
            sort_by=arxiv.SortCriterion.Relevance,
        )

        topic_docs = []
        for result in tqdm(
            client.results(search),
            total=topic_config["max_results"],
            desc=f"  {topic_name}",
        ):
            arxiv_id = result.entry_id.split("/")[-1]
            arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)

            if arxiv_id_clean in seen_ids:
                continue
            seen_ids.add(arxiv_id_clean)

            full_text = ""
            sections = {}

            if download_pdfs and pdf_dir:
                pdf_path = download_pdf(arxiv_id_clean, pdf_dir)
                if pdf_path:
                    full_text = extract_text_from_pdf(pdf_path)
                    if full_text:
                        sections = extract_sections_from_text(full_text)

            if not sections:
                sections = {"abstract": result.summary.strip()}
                full_text = result.summary.strip()

            year = result.published.year if result.published else None

            doc = {
                "doc_id": f"arxiv_{arxiv_id_clean}",
                "title": result.title.strip(),
                "authors": [a.name for a in result.authors],
                "year": year,
                "arxiv_id": arxiv_id_clean,
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "full_text": full_text,
                "sections": sections,
                "acronyms": build_acronym_dict(full_text),
                "topic": topic_name,
            }
            topic_docs.append(doc)

            time.sleep(0.5)

        all_docs.extend(topic_docs)
        logger.info(f"  Found {len(topic_docs)} unique papers for {topic_name}")

    logger.info(f"Total unique papers from search: {len(all_docs)}")
    return all_docs


def scrape_core(output_dir: Path, download_pdfs: bool = True) -> list[dict]:
    """Scrape the ~37 landmark papers (core corpus)."""
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    docs = fetch_by_ids(LANDMARK_PAPERS, download_pdfs=download_pdfs, pdf_dir=pdf_dir)
    return docs


def scrape_extended(output_dir: Path, download_pdfs: bool = True) -> list[dict]:
    """Scrape landmark papers + search-based papers (~500 total)."""
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Start with landmarks
    docs = fetch_by_ids(LANDMARK_PAPERS, download_pdfs=download_pdfs, pdf_dir=pdf_dir)
    seen_ids = {d["arxiv_id"] for d in docs}

    # Add search results
    search_docs = fetch_by_search(SEARCH_TOPICS, download_pdfs=download_pdfs, pdf_dir=pdf_dir)
    for doc in search_docs:
        if doc["arxiv_id"] not in seen_ids:
            docs.append(doc)
            seen_ids.add(doc["arxiv_id"])

    return docs


def scrape_abstracts(output_dir: Path) -> list[dict]:
    """Scrape ~2000 abstracts only (fast, no PDF download)."""
    # Increase max_results for abstract-only mode
    abstract_topics = {}
    for name, config in SEARCH_TOPICS.items():
        abstract_topics[name] = {
            "query": config["query"],
            "max_results": config["max_results"] * 5,
        }

    docs = fetch_by_ids(LANDMARK_PAPERS, download_pdfs=False)
    seen_ids = {d["arxiv_id"] for d in docs}

    search_docs = fetch_by_search(abstract_topics, download_pdfs=False)
    for doc in search_docs:
        if doc["arxiv_id"] not in seen_ids:
            docs.append(doc)
            seen_ids.add(doc["arxiv_id"])

    return docs


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Scrape AI/ML papers from ArXiv for RAG-Bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_arxiv.py                             # ~37 landmark papers with full text
  python scrape_arxiv.py --mode extended             # ~500 papers across all topics
  python scrape_arxiv.py --mode abstracts            # ~2000 abstracts only (fast)
  python scrape_arxiv.py --output ~/rag-bench/data   # Custom output directory
  python scrape_arxiv.py --no-pdf                    # Skip PDF download (abstracts only)
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["core", "extended", "abstracts"],
        default="core",
        help="Scraping mode: core (~37 landmark papers), extended (~500), or abstracts (~2000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data"),
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF download; use abstracts only",
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("RAG-Bench ArXiv Scraper")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {args.output}")
    logger.info(f"PDF download: {'disabled' if args.no_pdf else 'enabled'}")

    start = time.time()

    if args.mode == "core":
        docs = scrape_core(args.output, download_pdfs=not args.no_pdf)
    elif args.mode == "extended":
        docs = scrape_extended(args.output, download_pdfs=not args.no_pdf)
    elif args.mode == "abstracts":
        docs = scrape_abstracts(args.output)

    elapsed = time.time() - start

    # Save output
    output_path = args.output / "scraped_papers.json"
    with open(output_path, "w") as f:
        json.dump(docs, f, indent=2, default=str)

    # Also save as parsed_papers.json for direct pipeline compatibility
    compat_path = args.output / "parsed_papers.json"
    with open(compat_path, "w") as f:
        json.dump(docs, f, indent=2, default=str)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SCRAPING COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Papers scraped: {len(docs)}")
    logger.info(f"Time elapsed: {elapsed:.1f}s")
    logger.info(f"Output: {output_path}")
    logger.info(f"Pipeline-ready: {compat_path}")

    # Stats
    with_fulltext = sum(1 for d in docs if len(d.get("full_text", "")) > 500)
    abstract_only = len(docs) - with_fulltext
    years = [d["year"] for d in docs if d.get("year")]

    logger.info("\nStats:")
    logger.info(f"  Full text: {with_fulltext} papers")
    logger.info(f"  Abstract only: {abstract_only} papers")
    if years:
        logger.info(f"  Year range: {min(years)} - {max(years)}")

    # Show topic distribution
    topics = {}
    for d in docs:
        t = d.get("topic", "landmark")
        topics[t] = topics.get(t, 0) + 1
    logger.info(f"  Topics: {json.dumps(topics, indent=4)}")

    logger.info(f"\nNext step: copy {compat_path} to your rag-bench/data/ folder")
    logger.info("Then run: python main.py")


if __name__ == "__main__":
    main()
