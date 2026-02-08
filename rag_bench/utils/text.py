"""
text_utils.py — Shared text processing utilities.

Consolidates functions used across ingest, chunker, generator, and scraper:
- Section extraction from markdown/PDF text
- Acronym dictionary extraction
- Author formatting for citations
- Encoding artifact cleanup from PDF extraction
"""

import re

# ── Section header patterns common in AI/ML papers ──
SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminary",
    "preliminaries",
    "problem setup",
    "problem statement",
    "method",
    "methodology",
    "methods",
    "approach",
    "model",
    "architecture",
    "framework",
    "system",
    "attention",
    "multi-head attention",
    "self-attention",
    "training",
    "training objective",
    "training procedure",
    "optimization",
    "learning",
    "experiment",
    "experiments",
    "experimental setup",
    "experimental results",
    "evaluation",
    "results",
    "main results",
    "analysis",
    "ablation",
    "ablation study",
    "discussion",
    "conclusion",
    "conclusions",
    "limitation",
    "limitations",
    "broader impact",
    "appendix",
    "supplementary",
]

# Compiled pattern for detecting section headers in markdown
HEADER_PATTERN = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# Additional patterns for PDF-extracted text (plain text headers)
PDF_HEADER_PATTERNS = [
    re.compile(r"^(\d+\.?\s+[A-Z][A-Za-z\s]+)$"),  # Numbered: 1. Introduction
    re.compile(r"^([A-Z][A-Z\s]{3,40})$"),  # ALL CAPS: INTRODUCTION
    re.compile(
        r"^(Abstract|Introduction|Related Work|Background|"
        r"Method(?:ology|s)?|Approach|Model|Architecture|"
        r"Experiment(?:s|al)?(?:\s+(?:Setup|Results))?|Results|"
        r"Discussion|Conclusion(?:s)?|Limitation(?:s)?|"
        r"Training|Evaluation|Analysis|Appendix)\s*$",
        re.IGNORECASE,
    ),
]


def normalize_section_name(name: str) -> str:
    """Normalize a section header into a consistent key."""
    # Remove numbering like "3.1", "IV.", etc.
    name = re.sub(r"^[\d.]+\s*", "", name)
    name = re.sub(r"^[IVXLC]+\.?\s*", "", name)

    # Lowercase and strip
    name = name.lower().strip()

    # Remove trailing punctuation
    name = re.sub(r"[:\-–—]+$", "", name).strip()

    # Replace spaces/special chars with underscores
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")

    return name or "unnamed"


def extract_sections(text: str) -> dict[str, str]:
    """
    Split paper text into sections based on markdown headers.

    Returns a dict mapping normalized section names to their text content.
    Handles nested headers (##, ###) by flattening to top-level sections.
    """
    if not text or not text.strip():
        return {"full_text": ""}

    sections = {}
    current_section = "preamble"
    current_lines = []

    for line in text.split("\n"):
        header_match = HEADER_PATTERN.match(line.strip())
        if header_match:
            # Save previous section
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections[current_section] = section_text

            # Normalize section name
            raw_name = header_match.group(2).strip()
            current_section = normalize_section_name(raw_name)
            current_lines = []
        else:
            current_lines.append(line)

    # Save final section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections[current_section] = section_text

    return sections


def extract_sections_from_pdf(text: str) -> dict[str, str]:
    """
    Split PDF-extracted text into sections.

    Handles both markdown-style and plain text headers found in PDF extractions.
    Falls back to markdown extraction first, then tries PDF header patterns.
    """
    if not text or len(text.strip()) < 100:
        return {"full_text": text}

    sections = {}
    current_section = "preamble"
    current_lines = []

    # Combined patterns: markdown + PDF-specific
    all_patterns = [HEADER_PATTERN] + PDF_HEADER_PATTERNS

    for line in text.split("\n"):
        stripped = line.strip()
        matched_header = None

        for pattern in all_patterns:
            m = pattern.match(stripped)
            if m:
                if pattern is HEADER_PATTERN:
                    matched_header = m.group(2).strip()
                else:
                    matched_header = m.group(1) if m.lastindex else stripped
                break

        if matched_header and len(matched_header) < 80:
            # Save previous section
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text and len(section_text) > 30:
                    sections[current_section] = section_text

            current_section = normalize_section_name(matched_header)
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
    """
    Extract acronym definitions from paper text.

    Looks for patterns like:
    - "Maximum Inner Product Search (MIPS)"
    - "reinforcement learning from human feedback (RLHF)"
    """
    acronyms = {}

    # Pattern: "Full Name (ACRONYM)"
    pattern = r"([A-Za-z][A-Za-z\s\-]{2,50})\s*\(([A-Z][A-Z0-9]{1,10})\)"
    for match in re.finditer(pattern, text):
        full_form = match.group(1).strip()
        acronym = match.group(2).strip()

        # Validate: first letters of words should roughly match acronym
        words = [w for w in full_form.split() if w[0].isupper() or w[0].islower()]
        if len(words) >= 2:
            acronyms[acronym] = full_form

    return acronyms


def format_authors(authors: list[str] | str, max_authors: int = 3) -> str:
    """Format author list for citation display."""
    if isinstance(authors, str):
        # Try to parse comma-separated string
        authors = [a.strip() for a in authors.split(",") if a.strip()]

    if not authors:
        return "Unknown"

    # Get last names
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
# Encoding cleanup — fix common artifacts from PDF extraction
# ═══════════════════════════════════════════════════════════════════════════
# Map of garbled UTF-8-as-Latin-1 sequences to correct Unicode characters
_ENCODING_FIXES = {
    # Greek letters (very common in ML papers)
    "Î±": "α",
    "Î²": "β",
    "Î³": "γ",
    "Î´": "δ",
    "Îµ": "ε",
    "Î¶": "ζ",
    "Î·": "η",
    "Î¸": "θ",
    "Î¹": "ι",
    "Îº": "κ",
    "Î»": "λ",
    "Î¼": "μ",
    "Î½": "ν",
    "Î¾": "ξ",
    "Î¿": "ο",
    "Ï€": "π",
    "Ï": "ρ",
    "Ïƒ": "σ",
    "Ï„": "τ",
    "Ï…": "υ",
    "Ï†": "φ",
    "Ï‡": "χ",
    "Ïˆ": "ψ",
    "Ï‰": "ω",
    "Ïµ": "ε",
    "Ï²": "ρ",
    # Math symbols
    "â‰¤": "≤",
    "â‰¥": "≥",
    "â‰ˆ": "≈",
    "â†'": "→",
    "Ã—": "×",
    "Ã·": "÷",
    "Â±": "±",
    "âˆž": "∞",
    "âˆ'": "∑",
    "âˆš": "√",
    "âˆ‚": "∂",
    "âˆ†": "∆",
    "âˆ‡": "∇",
    "âˆˆ": "∈",
    "âˆ©": "∩",
    "âˆª": "∪",
    "âˆ¼": "∼",
    "Â·": "·",
    # Common accented chars & special
    "Âµ": "μ",
    "Ã¡": "á",
    "Ã©": "é",
    "Ã³": "ó",
    "Ã¶": "ö",
    "Ã¼": "ü",
    "Ã±": "ñ",
    "Ã§": "ç",
    # Subscript/superscript
    "Â²": "²",
    "Â³": "³",
}


def fix_encoding(text: str) -> str:
    """Fix common garbled UTF-8-as-Latin-1 encoding artifacts.

    Also attempts the more general fix: try re-encoding as latin-1
    and decoding as utf-8 on segments that look garbled.
    """
    if not text:
        return text

    # First, try the general fix: if the text has telltale garbled patterns
    # (Â followed by a non-ASCII char, or Ã followed by something), try to
    # re-encode the whole thing
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        # If successful and looks cleaner (fewer Â/Ã artifacts), use it
        if fixed.count("Â") + fixed.count("Ã") < text.count("Â") + text.count("Ã"):
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    # Fall back to lookup-table replacement for partial garbling
    for garbled, correct in _ENCODING_FIXES.items():
        if garbled in text:
            text = text.replace(garbled, correct)

    return text
