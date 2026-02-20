"""
generator.py — Answer generation with grounded citations.

Takes retrieved chunks from the hybrid retriever and generates
answers with numbered source citations. Includes a relevance gate
to trigger deflection when the corpus can't answer the question.

Supports multiple LLM backends:
- Local: Mistral-7B via llama-cpp-python or transformers
- API: OpenAI-compatible endpoints (vLLM, Ollama, etc.)
- Fallback: Template-based generation (no LLM needed)
"""

import json
import logging
import re

import requests

from rag_bench.utils.text import clean_latex_artifacts, fix_encoding

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Math post-processing — wrap bare math expressions in LaTeX delimiters
# ═══════════════════════════════════════════════════════════════════════════

# Greek letters that Mistral often outputs as bare Unicode or names
_GREEK_UNICODE_TO_LATEX = {
    "α": "\\alpha",
    "β": "\\beta",
    "γ": "\\gamma",
    "δ": "\\delta",
    "ε": "\\epsilon",
    "ζ": "\\zeta",
    "η": "\\eta",
    "θ": "\\theta",
    "ι": "\\iota",
    "κ": "\\kappa",
    "λ": "\\lambda",
    "μ": "\\mu",
    "ν": "\\nu",
    "ξ": "\\xi",
    "π": "\\pi",
    "ρ": "\\rho",
    "σ": "\\sigma",
    "τ": "\\tau",
    "υ": "\\upsilon",
    "φ": "\\phi",
    "χ": "\\chi",
    "ψ": "\\psi",
    "ω": "\\omega",
    "Γ": "\\Gamma",
    "Δ": "\\Delta",
    "Θ": "\\Theta",
    "Λ": "\\Lambda",
    "Ξ": "\\Xi",
    "Π": "\\Pi",
    "Σ": "\\Sigma",
    "Φ": "\\Phi",
    "Ψ": "\\Psi",
    "Ω": "\\Omega",
}

# Build a regex that matches any Greek letter surrounded by math-ish context
_GREEK_CHARS = "".join(_GREEK_UNICODE_TO_LATEX.keys())
_GREEK_RE = re.compile(f"([{re.escape(_GREEK_CHARS)}])")

# Math symbols that hint at equation context
_MATH_SYMBOLS = set("∑∏∫√∂∇∞≈≤≥≠±×·∈∉⊂⊃∩∪∼∝→←↔⟨⟩‖")


def _split_math_segments(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_math, content) segments preserving $...$ and $$...$$ blocks."""
    segments = []
    pattern = re.compile(r"\$\$[\s\S]*?\$\$|\$(?:[^$\\]|\\.)+?\$")
    last_end = 0
    for m in pattern.finditer(text):
        if m.start() > last_end:
            segments.append((False, text[last_end : m.start()]))
        segments.append((True, m.group(0)))
        last_end = m.end()
    if last_end < len(text):
        segments.append((False, text[last_end:]))
    return segments


def postprocess_math(text: str) -> str:
    """Post-process LLM output to wrap bare math expressions in LaTeX $ delimiters.

    Catches common patterns that Mistral outputs as plain text instead of LaTeX:
    1. Bare Greek Unicode letters (α, β, θ) -> $\\alpha$, $\\beta$, $\\theta$
    2. Named Greek with subscripts: alpha_t -> $\\alpha_t$
    3. Subscripted variables: x_t, x_{t-1}
    4. Simple superscripts: x^2
    5. Bare Unicode math symbols (∑, ∏, √, etc.)

    Already-wrapped text ($...$) is preserved via segment splitting.
    Each transformation is applied as a separate pass with fresh splitting
    so earlier transforms don't interfere with later ones.
    """
    if not text:
        return text

    # --- Pass 1: Greek Unicode letters with optional subscripts/superscripts ---
    def _pass_greek_unicode(plain: str) -> str:
        def replace_greek(m):
            char = m.group(1)
            latex_cmd = _GREEK_UNICODE_TO_LATEX.get(char, char)
            sub_sup = m.group(2) or ""
            start = m.start()
            before = plain[max(0, start - 1) : start] if start > 0 else " "
            if before.isalpha():
                return m.group(0)
            if sub_sup:
                kind = sub_sup[0]  # _ or ^
                inner = sub_sup[1:].strip("{}")
                return f"${latex_cmd}{kind}{{{inner}}}$"
            return f"${latex_cmd}$"

        pat = re.compile(
            f"([{re.escape(_GREEK_CHARS)}])"
            r"([_^](?:\{[^}]+\}|[a-zA-Z0-9]+))?"
        )
        return pat.sub(replace_greek, plain)

    # --- Pass 2: Unicode math symbols ---
    _LATEX_SYM_MAP = {
        "∑": "\\sum",
        "∏": "\\prod",
        "∫": "\\int",
        "√": "\\sqrt{}",
        "∂": "\\partial",
        "∇": "\\nabla",
        "∞": "\\infty",
        "≈": "\\approx",
        "≤": "\\leq",
        "≥": "\\geq",
        "≠": "\\neq",
        "±": "\\pm",
        "×": "\\times",
        "·": "\\cdot",
        "∈": "\\in",
        "∉": "\\notin",
        "⊂": "\\subset",
        "⊃": "\\supset",
        "∩": "\\cap",
        "∪": "\\cup",
        "∼": "\\sim",
        "∝": "\\propto",
        "→": "\\rightarrow",
        "←": "\\leftarrow",
        "↔": "\\leftrightarrow",
        "⟨": "\\langle",
        "⟩": "\\rangle",
        "‖": "\\|",
    }

    def _pass_unicode_symbols(plain: str) -> str:
        for sym, latex in _LATEX_SYM_MAP.items():
            if sym in plain:
                plain = plain.replace(sym, f"${latex}$")
        return plain

    # --- Pass 3: Named Greek with subscripts (alpha_t, sigma_{t-1}) ---
    _GREEK_NAMES = {
        "alpha": "\\alpha",
        "beta": "\\beta",
        "gamma": "\\gamma",
        "delta": "\\delta",
        "epsilon": "\\epsilon",
        "theta": "\\theta",
        "lambda": "\\lambda",
        "mu": "\\mu",
        "sigma": "\\sigma",
        "tau": "\\tau",
        "phi": "\\phi",
        "psi": "\\psi",
        "omega": "\\omega",
        "rho": "\\rho",
        "eta": "\\eta",
        "pi": "\\pi",
        "nu": "\\nu",
        "kappa": "\\kappa",
        "zeta": "\\zeta",
        "chi": "\\chi",
        "xi": "\\xi",
    }
    # Pre-compile named Greek patterns
    _NAMED_GREEK_PATS = [
        (re.compile(rf"(?<![a-zA-Z`]){name}_(\{{[^}}]+\}}|[a-zA-Z0-9]+)(?![a-zA-Z`])"), latex)
        for name, latex in _GREEK_NAMES.items()
    ]

    def _pass_named_greek(plain: str) -> str:
        for pat, latex_cmd in _NAMED_GREEK_PATS:
            plain = pat.sub(lambda m, lc=latex_cmd: f"${lc}_{{{m.group(1).strip('{}')}}}$", plain)
        return plain

    # --- Pass 4: Single-letter variable subscripts (x_t, W_q) ---
    _VAR_SUB_RE = re.compile(r"(?<![a-zA-Z`_])([a-zA-Z])_(\{[^}]+\}|[a-zA-Z0-9])(?![a-zA-Z`_])")

    def _pass_var_subscripts(plain: str) -> str:
        return _VAR_SUB_RE.sub(lambda m: f"${m.group(1)}_{{{m.group(2).strip('{}')}}}$", plain)

    # --- Pass 5: Simple superscripts (x^2, e^x) ---
    _VAR_SUP_RE = re.compile(r"(?<![a-zA-Z`])([a-zA-Z])\^(\{[^}]+\}|[a-zA-Z0-9])(?![a-zA-Z`])")

    def _pass_var_superscripts(plain: str) -> str:
        return _VAR_SUP_RE.sub(lambda m: f"${m.group(1)}^{{{m.group(2).strip('{}')}}}$", plain)

    # --- Apply all passes sequentially, re-splitting between each ---
    passes = [
        _pass_greek_unicode,
        _pass_unicode_symbols,
        _pass_named_greek,
        _pass_var_subscripts,
        _pass_var_superscripts,
    ]

    current = text
    for transform in passes:
        segments = _split_math_segments(current)
        parts = []
        for is_math, content in segments:
            if is_math:
                parts.append(content)
            else:
                parts.append(transform(content))
        current = "".join(parts)

    return current


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Templates
# ═══════════════════════════════════════════════════════════════════════════
# fmt: off
# ruff: noqa: E501
SYSTEM_PROMPT = """You are a precise AI/ML research assistant. Answer questions using ONLY the provided source passages. Follow these rules strictly:

1. CITE EVERY FACTUAL CLAIM inline using [Source N] notation, where N is the number shown in the provided sources (e.g. [Source 1], [Source 2]). Valid citation numbers are only those that appear in the source list you were given.
2. ONLY cite the provided sources. NEVER cite papers, authors, or results from your training memory. If you know a relevant paper that is not in the provided sources, do not cite it — instead note that the provided sources do not cover that aspect.
3. NEVER add a bibliography, references, or sources list at the end of your answer. All citations must be inline within the answer text only.
4. CITE PRECISELY — place [Source N] only immediately after the specific sentence or phrase that source directly supports. NEVER write catch-all sentences like "this is discussed in [Source 1], [Source 2], [Source 3]". If a source does not contribute a specific fact to your answer, do not cite it at all.
5. If the sources don't contain enough information, say so explicitly — never fabricate.
6. Use direct language. Prefer the paper's own terminology.
7. For numerical claims (BLEU scores, parameter counts, etc.), cite the exact source.
8. If sources conflict, note the disagreement and cite both.
9. Never cite a source for a claim it doesn't support.
9. IMPORTANT: When sources are relevant to the question's topic, USE THEM — synthesize what they cover and note any gaps. However, if the question asks for a SPECIFIC piece of information (a number, a fact, a detail) and that specific information does not appear anywhere in the sources, you MUST explicitly state that the sources do not contain/specify/mention that particular detail. For example: if asked "What hardware was used to train X?" and the sources discuss X but never mention hardware, say "The sources do not specify the hardware used." Do NOT answer with unrelated information from the same paper just because the topic matches.
10. If the question contains a false premise (e.g., "Paper X showed result Y" but the sources don't support that claim), point out that the premise appears incorrect based on the available sources.
11. FORMAT — CRITICAL RULES:
   a) Use **markdown** for structure (headings, lists, bold).
   b) ALL math MUST be written in LaTeX wrapped in dollar signs. Use $...$ for inline math and $$...$$ for display equations.
   c) NEVER copy raw equation text from the sources. ALWAYS rewrite equations in proper LaTeX.
   d) Greek letters MUST use LaTeX commands: \\alpha, \\beta, \\gamma, \\theta, \\epsilon, \\sigma, \\mu, \\pi, etc.
   e) Subscripts use _{...}, superscripts use ^{...}. Example: $x_{t-1}$, $\\alpha_t$, $e^{-x}$.
   f) Fractions use \\frac{num}{den}. Square roots use \\sqrt{...}. Products use \\prod, sums use \\sum.
   g) Common distributions: $\\mathcal{N}(\\mu, \\sigma^2)$. Loss functions: $\\mathcal{L}_{\\text{simple}}$.
   h) Examples of CORRECT formatting:
      - Inline: The noise schedule defines $\\alpha_t$ and $\\bar{\\alpha}_t = \\prod_{s=1}^{t} \\alpha_s$.
      - Display: $$q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{\\alpha_t} x_{t-1}, (1-\\alpha_t)\\mathbf{I})$$
      - Attention: $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
   i) If a source contains garbled text like "Î±" or "q(xo:r)", interpret it and write clean LaTeX."""
# fmt: on

GENERATION_PROMPT = """Based on the following source passages, answer the user's question.

{sources_block}

Question: {question}

CITATION RULES — read before writing:
- Use [Source N] immediately after the specific claim it supports. N must match the number in the sources listed above.
- NEVER write author names or paper titles as citations. Use [Source N] instead.
  WRONG:  "Vaswani et al. (2017) introduced scaled dot-product attention."
  CORRECT: "Scaled dot-product attention computes scores via softmax over dot products [Source 1]."
- Each [Source N] must follow a specific claim from that source. Do NOT write catch-all sentences.
  WRONG:  "This is discussed in [Source 1], [Source 2], and [Source 3]."
  CORRECT: Cite each source only where it contributed a specific fact.
- If a source does not contribute a specific fact to your answer, do not cite it at all.
- Do NOT add a bibliography or references section at the end.
- If the sources are insufficient for a specific claim, say so explicitly.

Formatting rules:
- Use **markdown** for structure. Use LaTeX for ALL math: $inline$ and $$display$$
- NEVER copy raw math text. Rewrite ALL equations in clean LaTeX:
  $$q(x_t | x_{{t-1}}) = \\mathcal{{N}}(x_t; \\sqrt{{\\alpha_t}} x_{{t-1}}, (1-\\alpha_t)\\mathbf{{I}})$$
- Greek letters: \\alpha, \\beta, \\theta, \\epsilon, \\sigma, \\mu, etc.

Answer:"""

DEFLECTION_RESPONSE = """I don't have sufficient information in my knowledge base to answer this accurately. {reason}

For the most current information, I'd recommend checking the original papers on arXiv or recent survey papers on this topic."""  # noqa: E501

# Phrases that indicate the LLM itself refused to answer (used by eval and CLI)
DEFLECTION_PHRASES = [
    "i don't have information",
    "i don't have sufficient information",
    "not covered in",
    "not appear in the",
    "no information about",
    "not mentioned in",
    "don't have enough information",
    "sources do not contain",
    "sources don't contain",
    "premise appears incorrect",
    "premise may be incorrect",
    "cannot find information",
    "not in my knowledge base",
    "not specified in",
    "not detailed in",
    "not disclosed",
    "does not specify",
    "does not mention",
    "does not provide",
    "do not contain",
    "do not specify",
    "do not provide",
    "no specific information",
    "insufficient information",
    "not explicitly stated",
    "not explicitly mentioned",
]


# ═══════════════════════════════════════════════════════════════════════════
# LLM Backends
# ═══════════════════════════════════════════════════════════════════════════
class LLMBackend:
    """Base class for LLM backends."""

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    """Generate via Ollama API (local LLM server)."""

    def __init__(self, model: str = "gemma2:27b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        logger.info(f"Ollama backend: {model} at {base_url}")

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,  # Low temp for factual accuracy
                "top_p": 0.9,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def generate_stream(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024):
        """
        Stream tokens from Ollama one at a time.

        Yields:
            str: Each token/chunk as it arrives from Ollama.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise


class OpenAICompatibleBackend(LLMBackend):
    """Generate via any OpenAI-compatible API (vLLM, LM Studio, Together, etc.)."""

    def __init__(
        self,
        model: str = "mistral-7b-instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        logger.info(f"OpenAI-compatible backend: {model} at {base_url}")

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            raise


class TemplateFallbackBackend(LLMBackend):
    """
    No-LLM fallback: builds answers by stitching together retrieved passages.

    Useful for testing the pipeline end-to-end without a GPU or API.
    Produces reasonable (if mechanical) answers with proper citations.
    """

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        # Extract source passages from the prompt
        sources = re.findall(
            r"\[Source (\d+)\].*?\n(.*?)(?=\[Source \d+\]|\nQuestion:)",
            prompt,
            re.DOTALL,
        )

        if not sources:
            return "I could not find relevant information to answer this question."

        # Build answer from passages
        answer_parts = []
        for num, text in sources:
            text = text.strip()
            if len(text) > 20:
                # Take first 2-3 sentences
                sentences = re.split(r"(?<=[.!?])\s+", text)
                excerpt = " ".join(sentences[:3]).strip()
                if excerpt:
                    answer_parts.append(f"{excerpt} [Source {num}].")

        if not answer_parts:
            return "The retrieved sources do not contain sufficient detail to answer this question precisely."

        return "\n\n".join(answer_parts)


def build_llm_backend(backend: str, model: str = "", base_url: str = ""):
    """Create the appropriate LLM backend.

    Centralised factory used by both the CLI and the API server.
    """
    if backend == "ollama":
        return OllamaBackend(
            model=model or "gemma2:27b",
            base_url=base_url or "http://localhost:11434",
        )
    elif backend == "openai":
        return OpenAICompatibleBackend(
            model=model or "mistral-7b-instruct",
            base_url=base_url or "http://localhost:8000/v1",
        )
    else:
        return TemplateFallbackBackend()


# ═══════════════════════════════════════════════════════════════════════════
# Relevance Gate (deflection logic)
# ═══════════════════════════════════════════════════════════════════════════
class RelevanceGate:
    """
    Decides whether retrieved context is sufficient to answer the question.
    If not, triggers deflection instead of hallucinating.

    Works with cross-encoder reranker scores (typically -10 to +10 range)
    as well as cosine similarity scores (0-1).

    Uses multiple signals:
    - Top retrieval score (below threshold -> deflect)
    - Score gap between #1 and #2 (large gap = more confident)
    - Query-document keyword overlap (cheap NLI proxy)
    - Score spread analysis
    """

    def __init__(
        self,
        min_top_score: float = 0.3,
        min_relevant_chunks: int = 1,
        score_concentration_threshold: float = 0.15,
        keyword_overlap_threshold: float = 0.25,
    ):
        self.min_top_score = min_top_score
        self.min_relevant_chunks = min_relevant_chunks
        self.score_concentration_threshold = score_concentration_threshold
        self.keyword_overlap_threshold = keyword_overlap_threshold
        self._calibrated = False
        self._effective_threshold = min_top_score

    def _auto_calibrate(self, top_score: float):
        """
        Auto-detect the scoring scale and set an effective threshold.

        Cross-encoder scores: roughly -10 to +10, good matches typically 0.5+
        Cosine similarity: 0 to 1, good matches > 0.5
        BM25/RRF: 0 to ~0.02, varied
        """
        if self._calibrated:
            return

        if top_score > 2.0:
            # Cross-encoder scale: scores can range widely, 0.5+ indicates relevance
            self._effective_threshold = max(self.min_top_score, 0.5)
            logger.debug(f"Auto-calibrated to cross-encoder scale: threshold={self._effective_threshold}")
        elif top_score > 0.05:
            self._effective_threshold = max(self.min_top_score, 0.3)
        else:
            self._effective_threshold = self.min_top_score

        self._calibrated = True

    @staticmethod
    def _naive_stem(word: str) -> str:
        """Naive suffix-stripping stemmer for keyword overlap matching."""
        if len(word) <= 4:
            return word
        for suffix in (
            "ation",
            "tion",
            "sion",
            "ment",
            "ness",
            "able",
            "ible",
            "ting",
            "ing",
            "ated",
            "ized",
            "ised",
            "ous",
            "ive",
            "ful",
            "less",
            "ity",
            "ies",
            "ally",
            "ly",
            "ers",
            "er",
            "ed",
            "es",
            "al",
            "s",
        ):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    def _compute_keyword_overlap(self, question: str, text: str) -> float:
        """Compute keyword overlap between question and passage."""
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "which",
            "who",
            "when",
            "where",
            "why",
            "does",
            "did",
            "do",
            "in",
            "of",
            "to",
            "for",
            "on",
            "with",
            "by",
            "from",
            "at",
            "and",
            "or",
            "not",
            "that",
            "this",
            "it",
            "its",
            "be",
            "been",
            "has",
            "have",
            "had",
            "their",
            "they",
            "them",
            "than",
            "used",
            "using",
            "about",
            "between",
            "specific",
            "according",
            "original",
            "showed",
            "paper",
            "model",
            "results",
            "much",
            "many",
            "can",
            "would",
            "could",
            "should",
            "will",
            "more",
            "most",
            "such",
            "also",
            "each",
            "other",
            "these",
            "those",
            "some",
            "all",
            "best",
            "way",
            "ways",
            "good",
            "better",
            "different",
            "method",
            "approach",
            "technique",
            "work",
            "works",
            "thing",
            "things",
        }

        q_raw = set(re.findall(r"[a-z0-9]+", question.lower())) - stopwords
        q_tokens = {t for t in q_raw if len(t) > 1}
        t_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))

        if not q_tokens:
            return 1.0

        q_stems = {self._naive_stem(t) for t in q_tokens}
        t_stems = {self._naive_stem(t) for t in t_tokens}

        matches = len(q_tokens & t_tokens) + len((q_stems - q_tokens) & t_stems)
        return min(1.0, matches / len(q_tokens))

    def _extract_entities(self, question: str) -> list[str]:
        """Extract named entities and technical terms from a question."""
        named_entities = re.findall(r"\b([A-Z][A-Za-z]*(?:[-][A-Za-z0-9]+)*)\b", question)
        sentence_start_words = {
            "What",
            "How",
            "Why",
            "When",
            "Where",
            "Which",
            "The",
            "According",
            "Did",
            "Does",
            "Is",
            "Are",
            "Can",
            "Could",
            "Would",
            "Should",
            "Both",
            "Compare",
            "Describe",
            "Explain",
            "Walk",
            "Tell",
            "List",
            "Summarize",
            "Outline",
            "Detail",
        }
        common_suffixes = {
            "trained",
            "based",
            "like",
            "style",
            "type",
            "level",
            "free",
            "aware",
            "specific",
            "driven",
            "tuned",
        }
        entities = []
        for e in named_entities:
            if e in sentence_start_words or len(e) <= 2:
                continue
            entities.append(e)
            if "-" in e:
                parts = e.split("-")
                base = parts[0]
                suffix = parts[-1].lower() if len(parts) > 1 else ""
                if len(base) > 2 and suffix in common_suffixes:
                    entities.append(base)

        tech_terms = re.findall(r"([A-Z]{2,}[-]?\d[\w.-]*)", question)
        entities.extend(tech_terms)

        camel_case = re.findall(r"([A-Z][a-z]+(?:[A-Z][a-z]+)+)", question)
        entities.extend(camel_case)

        quoted = re.findall(r'"([^"]+)"', question)
        entities.extend(quoted)

        seen = set()
        unique = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique.append(e)
        return unique

    def _normalize_entity(self, entity: str) -> set[str]:
        """Generate normalized variants of an entity name for flexible matching."""
        variants = {entity.lower()}
        e_lower = entity.lower()

        # Remove common suffixes like -trained, -based, -like
        for suffix in ["-trained", "-based", "-like", "-tuned", "-aware", "-specific", "-driven"]:
            if e_lower.endswith(suffix):
                base = e_lower[: -len(suffix)]
                variants.add(base)

        # Add camelCase splits (e.g., "InstructGPT" -> "instruct", "gpt")
        if any(c.isupper() for c in entity[1:]):
            parts = re.findall(r"[A-Z][a-z]*|[a-z]+", entity)
            variants.update(p.lower() for p in parts if len(p) > 2)

        # Add hyphen splits
        if "-" in entity:
            parts = entity.split("-")
            variants.update(p.lower() for p in parts if len(p) > 2)

        return variants

    def _check_entity_presence(self, question: str, retrieval_results: list[dict]) -> tuple[bool, str]:
        """Check if the main entity/concept is the subject of the retrieved passages."""
        entities = self._extract_entities(question)
        if not entities:
            return False, ""

        # Detect multi-document synthesis questions
        synthesis_indicators = [
            "compare",
            "contrast",
            "difference",
            "differ",
            "evolution",
            "trace",
            "design a",
            "combine",
            "synthesis",
            "across",
            "build a system",
            "pipeline",
            "approaches",
            "each",
        ]
        q_lower = question.lower()
        is_synthesis = any(indicator in q_lower for indicator in synthesis_indicators)

        # For synthesis questions, be more lenient
        MIN_PROMINENT_COUNT = 1 if is_synthesis and len(entities) > 1 else 3

        passage_text = " ".join(r.get("text", "") for r in retrieval_results[:5])  # Check more passages
        p_lower = passage_text.lower()

        titles = set()
        for r in retrieval_results[:5]:
            meta = r.get("metadata", {})
            title = meta.get("title", "")
            if title:
                titles.add(title.lower())
            source_display = meta.get("source_display", "")
            if source_display:
                titles.add(source_display.lower())

        titles_combined = " ".join(titles)

        CONCEPT_ACRONYMS = {
            "rag",
            "llm",
            "llms",
            "nlp",
            "ml",
            "ai",
            "cv",
            "rl",
            "gan",
            "gans",
            "vae",
            "vaes",
            "rnn",
            "rnns",
            "cnn",
            "cnns",
            "dnn",
            "dnns",
            "ssl",
            "rlhf",
            "dpo",
            "sft",
            "ppo",
            "moe",
        }

        if entities and all(e.lower() in CONCEPT_ACRONYMS for e in entities):
            return False, ""

        # Check which entities are covered by titles or prominent passage mentions (with variants)
        covered = set()
        for entity in entities:
            variants = self._normalize_entity(entity)

            # Check if any variant is in titles or passages
            for variant in variants:
                if variant in titles_combined or p_lower.count(variant) >= MIN_PROMINENT_COUNT:
                    covered.add(entity.lower())
                    break

        # For synthesis questions, require at least 50% entity coverage
        if is_synthesis:
            coverage_ratio = len(covered) / len(entities) if entities else 0
            if coverage_ratio >= 0.5:  # At least half the concepts present
                return False, ""
        else:
            # For single-focus questions, require high coverage
            any_covered = len(covered) > 0
            if any_covered:
                # At least one entity is well-represented. Check if any OTHER
                # entities are COMPLETELY absent — a sign of a false premise
                absent = []
                for e in entities:
                    if e.lower() in covered:
                        continue
                    if e.lower() in CONCEPT_ACRONYMS:
                        continue
                    if len(e) <= 2:
                        continue
                    # Check all variants
                    variants = self._normalize_entity(e)
                    if not any(v in p_lower or v in titles_combined for v in variants):
                        absent.append(e)

                if absent and len(absent) < len(entities):  # Some present, some absent
                    absent_str = ", ".join(absent)
                    return True, (
                        f"The key term(s) '{absent_str}' from the question do not appear in the retrieved passages."
                    )
                return False, ""

        # No entity is well-covered — check if any appear at all (with variants)
        any_present = False
        for e in entities:
            variants = self._normalize_entity(e)
            if any(v in p_lower for v in variants):
                any_present = True
                break

        if not any_present:
            missing = ", ".join(entities)
            return True, (f"The key term(s) '{missing}' from the question do not appear in any retrieved passages.")

        counts = {}
        for e in entities:
            variants = self._normalize_entity(e)
            total_count = sum(p_lower.count(v) for v in variants)
            if total_count > 0:
                counts[e] = total_count

        if not counts:  # No entities found even with variants
            return True, "The retrieved passages do not cover the concepts mentioned in the question."

        detail = ", ".join(f"'{e}' ({c}x)" for e, c in counts.items())
        top_titles = [r.get("metadata", {}).get("title", "?")[:50] for r in retrieval_results[:2]]
        return True, (
            f"The term(s) {detail} appear only in passing in the retrieved passages. "
            f"Top source papers are about: {'; '.join(top_titles)}. "
            f"The knowledge base likely doesn't have detailed coverage of this topic."
        )

    def _is_multi_document_query(self, question: str) -> bool:
        """Detect queries that require synthesizing information from multiple documents."""
        q_lower = question.lower()

        # Comparison indicators
        comparison_words = ["compare", "contrast", "differ", "difference", "versus", "vs", "both", "each", "between"]

        # Synthesis indicators
        synthesis_words = [
            "evolution",
            "trace",
            "history",
            "combine",
            "together",
            "design a",
            "build a",
            "pipeline",
            "system",
        ]

        # Abstract analysis indicators (need very lenient thresholds)
        abstract_words = ["impact", "think about", "philosophy", "approach to", "foundational", "fundamental"]

        # Multiple entity indicators
        has_comma_separated = len(re.findall(r"\w+,\s*\w+", question)) > 0
        has_and_separated = " and " in q_lower and not q_lower.startswith("what is")

        return (
            any(word in q_lower for word in comparison_words)
            or any(word in q_lower for word in synthesis_words)
            or any(word in q_lower for word in abstract_words)
            or has_comma_separated
            or has_and_separated
        )

    def _is_abstract_analysis_query(self, question: str) -> bool:
        """Detect high-level abstract questions about impact, design, philosophy."""
        q_lower = question.lower()
        abstract_indicators = [
            "impact",
            "think about",
            "philosophy",
            "foundational",
            "design a",
            "build a",
            "create a",
            "system that",
            "how we think",
            "approaches to",
            "fundamental",
        ]
        return any(indicator in q_lower for indicator in abstract_indicators)

    def should_deflect(self, retrieval_results: list[dict], question: str = "") -> tuple[bool, str]:
        """Determine if the system should deflect (refuse to answer)."""
        if not retrieval_results:
            return True, "No relevant passages were found in the knowledge base."

        scores = [r.get("score", 0.0) for r in retrieval_results]
        top_score = scores[0]

        self._auto_calibrate(top_score)

        # Detect multi-document synthesis queries
        is_synthesis = self._is_multi_document_query(question) if question else False
        is_abstract = self._is_abstract_analysis_query(question) if question else False

        # Adjust threshold for synthesis queries - they naturally have lower scores
        # because no single passage perfectly matches a cross-document question
        effective_threshold = self._effective_threshold
        if is_abstract:
            # Abstract "impact" and "design" questions need the most lenient threshold
            effective_threshold = max(0.3, self._effective_threshold * 0.4)  # 60% lower
            logger.debug(f"Abstract analysis query detected, lowered threshold to {effective_threshold:.2f}")
        elif is_synthesis:
            effective_threshold = max(0.5, self._effective_threshold * 0.6)  # 40% lower threshold
            logger.debug(f"Multi-document synthesis query detected, lowered threshold to {effective_threshold:.2f}")

        if top_score < effective_threshold:
            return True, (
                f"The most relevant passage scored only {top_score:.2f}, "
                f"below the confidence threshold of {effective_threshold:.2f}."
            )

        # For synthesis queries, check if we have diverse sources (but be lenient)
        if is_synthesis and not is_abstract:
            # Count unique source papers in top results
            unique_sources = set()
            for r in retrieval_results[:10]:
                meta = r.get("metadata", {})
                source = meta.get("source_display", "")
                if source:
                    unique_sources.add(source)

            # For multi-document questions with explicit comparisons, require 2+ sources
            # For other synthesis (e.g., "How does X work?" that mentions multiple concepts),
            # don't enforce this - they can be answered from one comprehensive source
            requires_multiple = any(
                word in question.lower()
                for word in ["compare", "contrast", "differ", "both", "versus", "vs", "evolution from", "trace"]
            )

            if requires_multiple and len(unique_sources) < 2:
                return True, (
                    f"This question appears to require information from multiple papers, "
                    f"but only {len(unique_sources)} distinct source(s) found in top results."
                )

        relevant_count = sum(1 for s in scores if s >= effective_threshold)
        if relevant_count < self.min_relevant_chunks:
            return True, (
                f"Only {relevant_count} passage(s) met the relevance threshold ({effective_threshold:.1f}). "
                f"At least {self.min_relevant_chunks} are needed for a confident answer."
            )

        return False, ""

    def _check_focal_terms(self, question: str, retrieval_results: list[dict]) -> tuple[bool, str]:
        """Check if the question's focal detail terms appear in the passages.

        Catches questions that ask about a specific detail (e.g., 'hardware')
        about a topic in the corpus where that detail isn't actually covered.
        Also catches adversarial questions claiming a model does something it doesn't.
        """
        q_lower = question.lower()
        passage_text = " ".join(r.get("text", "") for r in retrieval_results[:5]).lower()

        # Pattern 1: "what specific/particular/exact [detail]"
        match = re.search(r"what\s+(?:specific|particular|exact)\s+(\w+)", q_lower)
        if match:
            focal = match.group(1).strip()
            # Use word boundary regex to avoid substring matches
            focal_pattern = re.compile(r"\b" + re.escape(focal) + r"\b", re.IGNORECASE)
            matches = focal_pattern.findall(passage_text)
            # Require at least 3 occurrences for substantial coverage (not just passing mentions)
            if len(focal) > 3 and len(matches) < 3:
                if len(matches) >= 1:
                    return True, (
                        f"The question asks about '{focal}' but this specific "
                        f"detail appears only {len(matches)} time(s) in the retrieved passages, "
                        f"suggesting limited coverage of this detail."
                    )
                return True, (
                    f"The question asks about '{focal}' but this specific "
                    f"detail does not appear in the retrieved passages."
                )

        # Pattern 2: "use/used to [verb] [target-noun]" — verify the target noun
        match = re.search(r"\buse(?:s|d)?\s+to\s+\w+\s+([\w][\w-]+)", q_lower)
        if match:
            focal = match.group(1).strip()
            focal_pattern = re.compile(r"\b" + re.escape(focal) + r"\b", re.IGNORECASE)
            matches = focal_pattern.findall(passage_text)
            if len(focal) > 3 and len(matches) < 2:
                if matches:
                    return True, (
                        f"The question asks about '{focal}' but this term "
                        f"appears only once (likely a passing mention) in the retrieved passages."
                    )
                return True, (
                    f"The question asks about '{focal}' but this term does not appear in the retrieved passages."
                )

        # Pattern 3: "be used to [verb] [target-noun]"
        match = re.search(r"\bused\s+to\s+\w+\s+([\w][\w-]+)", q_lower)
        if match:
            focal = match.group(1).strip()
            focal_pattern = re.compile(r"\b" + re.escape(focal) + r"\b", re.IGNORECASE)
            matches = focal_pattern.findall(passage_text)
            if len(focal) > 3 and len(matches) < 2:
                if matches:
                    return True, (
                        f"The question asks about '{focal}' but this term "
                        f"appears only once (likely a passing mention) in the retrieved passages."
                    )
                return True, (
                    f"The question asks about '{focal}' but this term does not appear in the retrieved passages."
                )

        return False, ""

    def _check_false_premise(
        self,
        question: str,
        passages_text: str,
        retrieval_results: list[dict] | None = None,
    ) -> tuple[bool, str]:
        """Detect adversarial queries that reference real papers but make false claims."""
        q_lower = question.lower()
        p_lower = passages_text.lower()

        claim_patterns = [
            r"according to (?:the )?(\w[\w\s]*?)(?:paper|study|work|authors?),?\s+(?:what|how|the|their)\s+(?:were\s+)?(?:the )?results? (?:of|on|for)\s+(?:applying\s+\w+\s+to\s+)?(\w[\w\s]*?)[\?\.]",  # noqa: E501
            r"(?:the|that)\s+(?:original\s+)?(\w[\w\s]*?)paper\s+(?:showed?|demonstrated?|proved?|claimed?)\s+(?:that\s+)?(?:their\s+)?model\s+(\w[\w\s]*?)[\.\?]",
            r"what (?:accuracy|score|result|performance) did (?:the )?(\w[\w\s]*?)(?:achieve|get|reach) on (\w[\w\s]*?)[\?\.]",  # noqa: E501
        ]

        for pattern in claim_patterns:
            match = re.search(pattern, q_lower)
            if match:
                groups = match.groups()
                for claimed_term in groups:
                    claimed_term = claimed_term.strip()
                    if len(claimed_term) > 3 and claimed_term not in p_lower:
                        return True, (
                            f"The question claims something about '{claimed_term}', "
                            f"but this term doesn't appear in the retrieved passages. "
                            f"The premise of the question may be incorrect."
                        )

        # Paper-specific claim check: for "according to the X paper" questions,
        # verify the claimed detail exists in passages FROM that paper (not just
        # from other papers that might discuss the topic).
        paper_ref = re.search(r"according to (?:the )?(\w[\w\s-]*?)(?:paper|study|work)", q_lower)
        if paper_ref:
            ref_name = paper_ref.group(1).strip()
            # Find passages specifically from the referenced paper
            ref_passages_text = []
            for r in retrieval_results[:10]:
                meta = r.get("metadata", {})
                title = (meta.get("title", "") or "").lower()
                source = (meta.get("source_display", "") or "").lower()
                if ref_name in title or ref_name in source:
                    ref_passages_text.append(r.get("text", ""))

            if not ref_passages_text:
                # Referenced paper not found in top results - suspicious
                return True, (
                    f"The question references 'the {ref_name} paper', but no passages "
                    f"from this specific paper appear in the top retrieval results. "
                    f"Unable to verify the claim."
                )

            if ref_passages_text:
                ref_text_lower = " ".join(ref_passages_text).lower()
                # Extract the claimed application/domain from the question
                claim_match = re.search(
                    r"(?:applying\s+\w+\s+to|applied\s+to|results?\s+(?:of|on)\s+applying\s+\w+\s+to)\s+"
                    r"([\w][\w\s-]+?)(?:\s+models?)?[\?\.]",
                    q_lower,
                )
                if claim_match:
                    claimed = claim_match.group(1).strip()
                    # Check both the exact phrase and key terms using word boundaries
                    claimed_pattern = re.compile(r"\b" + re.escape(claimed) + r"\b", re.IGNORECASE)
                    matches = claimed_pattern.findall(ref_text_lower)

                    # For claimed applications/results, check TWO signals:
                    # 1. Frequency: appears 5+ times (papers mention related work 2-4 times)
                    # 2. Context: the claimed domain AND the method appear TOGETHER near result phrases
                    if len(claimed) > 3 and len(matches) < 5:
                        if len(matches) >= 1:
                            return True, (
                                f"The question asks about '{claimed}' in the context of "
                                f"the {ref_name} paper, but this term appears only {len(matches)} time(s) "
                                f"(likely a passing mention in related work, not a studied application). "
                                f"The premise may be incorrect."
                            )
                        return True, (
                            f"The question asks about '{claimed}' in the context of "
                            f"the {ref_name} paper, but this term doesn't appear in "
                            f"that paper's passages. The premise may be incorrect."
                        )

                    # Even if it appears 5+ times, check if BOTH the method name and claimed
                    # domain appear together in experimental contexts (not just one or the other)
                    result_indicators = [
                        "accuracy",
                        "performance",
                        "results",
                        "experiments",
                        "trained",
                        "evaluated",
                        "tested",
                        "score",
                        "benchmark",
                        "dataset",
                        "loss",
                        "metric",
                        "improve",
                        "achieve",
                        "outperform",
                        "table",
                        "figure",
                        "shown in",
                    ]
                    # Look for windows where claimed term, method name, and result indicators all appear
                    method_terms = [ref_name.lower(), "lora", "low-rank", "adaptation"]
                    has_joint_experimental_context = False
                    for match_pos in [m.start() for m in re.finditer(claimed_pattern, ref_text_lower)]:
                        # Use a tighter window (100 chars each side) for stricter matching
                        window_start = max(0, match_pos - 100)
                        window_end = min(len(ref_text_lower), match_pos + len(claimed) + 100)
                        context_window = ref_text_lower[window_start:window_end]
                        # Check if BOTH method and result indicators appear in this tight window
                        has_method = any(term in context_window for term in method_terms)
                        has_results = any(indicator in context_window for indicator in result_indicators)
                        if has_method and has_results:
                            has_joint_experimental_context = True
                            break

                    if not has_joint_experimental_context and len(matches) >= 1:
                        return True, (
                            f"The question asks about applying {ref_name} to '{claimed}', "
                            f"but while '{claimed}' is mentioned {len(matches)} time(s) in the paper, "
                            f"there's no evidence of the paper actually studying {ref_name} on '{claimed}' "
                            f"(no joint mentions with experimental results). "
                            f"The premise may be incorrect."
                        )

        return False, ""


# ═══════════════════════════════════════════════════════════════════════════
# Main Generator
# ═══════════════════════════════════════════════════════════════════════════
class RAGGenerator:
    """
    Full RAG generation pipeline: retrieve -> gate -> generate -> cite.

    Usage:
        generator = RAGGenerator(retriever=hybrid_retriever)
        answer = generator.answer("How does attention work?")
    """

    def __init__(
        self,
        retriever,
        llm_backend: LLMBackend | None = None,
        relevance_gate: RelevanceGate | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        top_k: int = 5,
        citation_booster=None,
    ):
        self.retriever = retriever
        self.llm = llm_backend or TemplateFallbackBackend()
        self.gate = relevance_gate or RelevanceGate()
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.citation_booster = citation_booster

        backend_name = type(self.llm).__name__
        logger.info(f"RAGGenerator ready (backend: {backend_name}, top_k={top_k})")

    def _build_sources_block(self, results: list[dict]) -> str:
        """Format retrieved chunks as numbered source blocks for the prompt."""
        blocks = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            source_label = meta.get("source_display", "Unknown source")
            section = meta.get("section", "")

            # Clean encoding artifacts and broken LaTeX from source data
            text = fix_encoding(r.get("text", ""))
            text = clean_latex_artifacts(text)

            block = f"[Source {i}] {source_label}{f' — {section}' if section else ''}\n{text}"
            blocks.append(block)

        return "\n\n".join(blocks)

    def _check_answer_alignment(self, question: str, answer: str) -> tuple[bool, str]:
        """Detect tangential answers where the LLM dumps related info instead of addressing the specific detail."""
        q_lower = question.lower()
        a_lower = answer.lower()

        patterns = [
            r"what\s+(?:specific|particular|exact)\s+(\w+)",
            r"how\s+much\s+(\w+)",
            r"how\s+many\s+(\w+)",
        ]

        skip_words = {
            "is",
            "are",
            "was",
            "were",
            "does",
            "did",
            "do",
            "the",
            "a",
            "an",
            "of",
            "data",
            "information",
        }

        for pattern in patterns:
            match = re.search(pattern, q_lower)
            if match:
                focus = match.group(1)
                if focus in skip_words:
                    continue

                focus_stem = self.gate._naive_stem(focus) if hasattr(self.gate, "_naive_stem") else focus
                if focus not in a_lower and focus_stem not in a_lower:
                    return True, focus

        return False, ""

    def _filter_relevant_sources(self, results: list[dict]) -> list[dict]:
        """Filter retrieval results to only include genuinely relevant sources.

        Uses a relative threshold based on the score gap between the top result
        and the rest:

        - Dominant winner: top score is strong AND the gap ratio to the second
          result is large (≥ 1.8×). Use a tight 70% threshold so only sources
          that are also strong matches survive. This prevents a single highly
          relevant chunk from dragging in several loosely related ones.
        - Competitive field: scores are closer together. Use a loose 35%
          threshold so multiple relevant sources are kept, enabling the model
          to draw on a broader evidence base for comparative or survey queries.
        """
        if not results:
            return results

        top_score = results[0].get("score", 0.0)
        second_score = results[1].get("score", 0.0) if len(results) > 1 else 0.0
        gap_ratio = (top_score / second_score) if second_score > 0 else float("inf")

        if top_score > 2.0:  # cross-encoder range
            min_ratio = 0.7 if top_score > 3.0 and gap_ratio >= 1.8 else 0.35
        elif top_score > 0.05:  # cosine similarity range
            min_ratio = 0.8 if top_score > 0.7 and gap_ratio >= 1.8 else 0.5
        else:
            return results[:1]

        filtered = []
        for r in results:
            score = r.get("score", 0.0)
            if score < top_score * min_ratio:
                continue
            filtered.append(r)
            if len(filtered) >= 5:
                break

        if not filtered and results:
            filtered = results[:1]

        return filtered

    def _format_citations(self, results: list[dict]) -> list[str]:
        """Build formatted citation list."""
        citations = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            citation = f"[Source {i}] {meta.get('source_display', 'Unknown')} — {meta.get('section', 'Unknown section')}"
            citations.append(citation)
        return citations

    def answer(self, question: str, top_k: int | None = None) -> dict:
        """Generate a grounded answer with citations."""
        k = top_k or self.top_k

        # Increase retrieval for multi-document synthesis queries
        is_synthesis = RelevanceGate._is_multi_document_query(RelevanceGate(), question)
        if is_synthesis and k < 15:
            k = min(k + 5, 15)  # Retrieve 5 more passages, max 15
            logger.debug(f"Multi-document query detected, increased top_k to {k}")

        # Step 0.5: Identify foundational papers to inject
        inject_chunks = []
        if self.citation_booster:
            relevant_ids = self.citation_booster.identify_relevant_papers(question)
            if relevant_ids:
                for arxiv_id in relevant_ids:
                    inject_chunks.extend(self.retriever.fetch_paper_chunks(arxiv_id, max_chunks=5))
                if inject_chunks:
                    logger.info(f"Foundational injection: {len(inject_chunks)} chunks from {len(relevant_ids)} paper(s)")

        # Step 1: Retrieve
        # For citation boosting, we need MORE candidates to ensure foundational papers are included
        boost_candidates = k * 5 if self.citation_booster else k  # 5x candidates for boosting
        boost_candidates = min(boost_candidates, 50)  # Cap at 50

        retrieval = self.retriever.query(
            question,
            top_k=boost_candidates,
            inject_chunks=inject_chunks or None,
        )

        # Step 1.5: Apply citation boosting (if enabled)
        if self.citation_booster:
            # Check if we have foundational papers before boosting
            if logger.isEnabledFor(logging.DEBUG):
                foundational_found = []
                for r in retrieval[:20]:
                    arxiv_id = r.get("metadata", {}).get("arxiv_id", "").replace("arxiv_", "").replace("_", ".")
                    if arxiv_id in self.citation_booster.foundational_papers:
                        title = self.citation_booster.foundational_papers[arxiv_id]["title"]
                        rank = next(i for i, x in enumerate(retrieval, 1) if x == r)
                        foundational_found.append(f"{title} at rank {rank}")
                if foundational_found:
                    logger.debug(f"Foundational papers before boosting: {', '.join(foundational_found)}")
                else:
                    logger.warning(f"No foundational papers found in top-20 results for query: {question[:50]}...")

            retrieval = self.citation_booster.boost_results(
                retrieval,
                query=question,
                top_k=None,  # Don't truncate; let diversify_results handle selection
            )

            # Apply diversification to ensure foundational papers are included
            retrieval = self.citation_booster.diversify_results(
                retrieval,
                top_k=k,
                max_per_paper=1,
                require_foundational=True,
            )

            logger.debug(f"Citation boosting applied: {len(retrieval)} results after boost+diversity")

        # Step 2: Relevance gate
        should_deflect, reason = self.gate.should_deflect(retrieval, question=question)

        if not should_deflect and retrieval:
            # Adjust keyword overlap threshold for synthesis queries
            is_synthesis = self.gate._is_multi_document_query(question)
            overlap_threshold = (
                self.gate.keyword_overlap_threshold * 0.6 if is_synthesis else self.gate.keyword_overlap_threshold
            )

            top_texts = " ".join(r["text"] for r in retrieval[:3])
            overlap = self.gate._compute_keyword_overlap(question, top_texts)

            if overlap < overlap_threshold:
                should_deflect = True
                reason = (
                    f"The retrieved passages have low keyword overlap ({overlap:.0%}) "
                    f"with the question, suggesting this topic isn't covered in the knowledge base."
                )

            if not should_deflect:
                should_deflect, entity_reason = self.gate._check_entity_presence(question, retrieval)
                if should_deflect:
                    reason = entity_reason

            if not should_deflect:
                should_deflect, adv_reason = self.gate._check_false_premise(
                    question,
                    top_texts,
                    retrieval_results=retrieval,
                )
                if should_deflect:
                    reason = adv_reason

            if not should_deflect:
                should_deflect, focal_reason = self.gate._check_focal_terms(question, retrieval)
                if should_deflect:
                    reason = focal_reason

        if should_deflect:
            logger.info(f"Deflecting query: {reason}")
            return {
                "answer": DEFLECTION_RESPONSE.format(reason=reason),
                "sources": [],
                "results": retrieval,
                "deflected": True,
                "deflection_reason": reason,
                "scores": [r.get("score", 0) for r in retrieval],
            }

        # Step 3: Filter to relevant sources only
        relevant = self._filter_relevant_sources(retrieval)
        n_filtered = len(retrieval) - len(relevant)
        if n_filtered:
            logger.debug(
                f"Source filtering: {len(retrieval)} -> {len(relevant)} (dropped {n_filtered} low-relevance sources)"
            )

        # Step 3.5: Ensure foundational paper survives source filtering
        if self.citation_booster and relevant:
            has_foundational = any(
                r.get("metadata", {}).get("arxiv_id", "").replace("arxiv_", "").replace("_", ".")
                in self.citation_booster.foundational_papers
                for r in relevant
            )
            if not has_foundational:
                for r in retrieval:
                    aid = r.get("metadata", {}).get("arxiv_id", "").replace("arxiv_", "").replace("_", ".")
                    if aid in self.citation_booster.foundational_papers:
                        relevant[-1] = r
                        logger.debug(f"Foundational paper preserved after filtering: {aid}")
                        break

        # Step 4: Build prompt with filtered sources
        sources_block = self._build_sources_block(relevant)
        prompt = GENERATION_PROMPT.format(
            sources_block=sources_block,
            question=question,
        )

        # Step 5: Generate answer
        try:
            answer_text = self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            fallback = TemplateFallbackBackend()
            answer_text = fallback.generate(prompt=prompt)

        # Step 5.5: Post-process math formatting
        answer_text = postprocess_math(answer_text)

        # Step 5.6: Post-generation answer alignment check
        is_tangential, focus_term = self._check_answer_alignment(question, answer_text)
        if is_tangential:
            reason = (
                f"The question asks about '{focus_term}' but this specific detail "
                f"does not appear in the retrieved sources or the generated answer."
            )
            logger.info(f"Tangential answer detected: {reason}")
            return {
                "answer": (
                    f"The sources do not contain specific information about "
                    f"'{focus_term}' for this topic. While related papers were found, "
                    f"they do not address this particular detail."
                ),
                "sources": [],
                "results": retrieval,
                "filtered_results": [],
                "deflected": True,
                "deflection_reason": reason,
                "scores": [r.get("score", 0) for r in retrieval],
            }

        # Step 6: Format citations
        citations = self._format_citations(relevant)

        # Append sources to answer
        full_answer = answer_text.strip()
        if citations:
            full_answer += "\n\nSources:\n" + "\n".join(citations)

        return {
            "answer": full_answer,
            "sources": citations,
            "results": retrieval,
            "filtered_results": relevant,
            "deflected": False,
            "deflection_reason": "",
            "scores": [r.get("score", 0) for r in retrieval],
        }

    def answer_stream(self, question: str, top_k: int | None = None):
        """
        Streaming version of answer() — yields events for SSE.

        Yields dicts with 'event' key:
        - {"event": "sources", "sources": [...], "filtered_sources": [...]}
        - {"event": "deflected", "answer": "...", "reason": "..."}
        - {"event": "token", "token": "..."}
        - {"event": "done", "answer": "...", "sources": [...]}
        """
        k = top_k or self.top_k

        # Identify foundational papers to inject
        inject_chunks = []
        if self.citation_booster:
            relevant_ids = self.citation_booster.identify_relevant_papers(question)
            if relevant_ids:
                for arxiv_id in relevant_ids:
                    inject_chunks.extend(self.retriever.fetch_paper_chunks(arxiv_id, max_chunks=5))
                if inject_chunks:
                    logger.info(f"Foundational injection: {len(inject_chunks)} chunks from {len(relevant_ids)} paper(s)")

        # For citation boosting, we need MORE candidates to ensure foundational papers are included
        boost_candidates = k * 5 if self.citation_booster else k  # 5x candidates for boosting
        boost_candidates = min(boost_candidates, 50)  # Cap at 50

        retrieval = self.retriever.query(
            question,
            top_k=boost_candidates,
            inject_chunks=inject_chunks or None,
        )

        # Apply citation boosting (if enabled)
        if self.citation_booster:
            # Check if we have foundational papers before boosting
            if logger.isEnabledFor(logging.DEBUG):
                foundational_found = []
                for r in retrieval[:20]:
                    arxiv_id = r.get("metadata", {}).get("arxiv_id", "").replace("arxiv_", "").replace("_", ".")
                    if arxiv_id in self.citation_booster.foundational_papers:
                        title = self.citation_booster.foundational_papers[arxiv_id]["title"]
                        rank = next(i for i, x in enumerate(retrieval, 1) if x == r)
                        foundational_found.append(f"{title} at rank {rank}")
                if foundational_found:
                    logger.debug(f"Foundational papers before boosting: {', '.join(foundational_found)}")
                else:
                    logger.warning(f"No foundational papers found in top-20 results for query: {question[:50]}...")

            retrieval = self.citation_booster.boost_results(
                retrieval,
                query=question,
                top_k=None,  # Don't truncate; let diversify_results handle selection
            )

            # Apply diversification to ensure foundational papers are included
            retrieval = self.citation_booster.diversify_results(
                retrieval,
                top_k=k,
                max_per_paper=1,
                require_foundational=True,
            )

            logger.debug(f"Citation boosting applied: {len(retrieval)} results after boost+diversity")

        relevant = self._filter_relevant_sources(retrieval)

        # Ensure foundational paper survives source filtering
        if self.citation_booster and relevant:
            has_foundational = any(
                r.get("metadata", {}).get("arxiv_id", "").replace("arxiv_", "").replace("_", ".")
                in self.citation_booster.foundational_papers
                for r in relevant
            )
            if not has_foundational:
                for r in retrieval:
                    aid = r.get("metadata", {}).get("arxiv_id", "").replace("arxiv_", "").replace("_", ".")
                    if aid in self.citation_booster.foundational_papers:
                        relevant[-1] = r
                        logger.debug(f"Foundational paper preserved after filtering: {aid}")
                        break

        yield {
            "event": "sources",
            "results": retrieval,
            "filtered_results": relevant,
            "scores": [r.get("score", 0) for r in retrieval],
        }

        should_deflect, reason = self.gate.should_deflect(retrieval, question=question)

        if not should_deflect and retrieval:
            # Adjust keyword overlap threshold for synthesis queries
            is_synthesis = self.gate._is_multi_document_query(question)
            overlap_threshold = (
                self.gate.keyword_overlap_threshold * 0.6 if is_synthesis else self.gate.keyword_overlap_threshold
            )

            top_texts = " ".join(r["text"] for r in retrieval[:3])
            overlap = self.gate._compute_keyword_overlap(question, top_texts)
            if overlap < overlap_threshold:
                should_deflect = True
                reason = f"Low keyword overlap ({overlap:.0%})"

            if not should_deflect:
                should_deflect, entity_reason = self.gate._check_entity_presence(question, retrieval)
                if should_deflect:
                    reason = entity_reason

            if not should_deflect:
                should_deflect, adv_reason = self.gate._check_false_premise(
                    question,
                    top_texts,
                    retrieval_results=retrieval,
                )
                if should_deflect:
                    reason = adv_reason

            if not should_deflect:
                should_deflect, focal_reason = self.gate._check_focal_terms(question, retrieval)
                if should_deflect:
                    reason = focal_reason

        if should_deflect:
            logger.info(f"Deflecting query: {reason}")
            yield {
                "event": "deflected",
                "answer": DEFLECTION_RESPONSE.format(reason=reason),
                "reason": reason,
            }
            return

        sources_block = self._build_sources_block(relevant)
        prompt = GENERATION_PROMPT.format(
            sources_block=sources_block,
            question=question,
        )

        full_text = ""
        has_streaming = hasattr(self.llm, "generate_stream")

        if has_streaming:
            try:
                for token in self.llm.generate_stream(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                ):
                    full_text += token
                    yield {"event": "token", "token": token}
            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
                has_streaming = False

        if not has_streaming:
            try:
                full_text = self.llm.generate(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                fallback = TemplateFallbackBackend()
                full_text = fallback.generate(prompt=prompt)
            yield {"event": "token", "token": full_text}

        full_text = postprocess_math(full_text)

        is_tangential, focus_term = self._check_answer_alignment(question, full_text)
        if is_tangential:
            reason = (
                f"The question asks about '{focus_term}' but this specific detail "
                f"does not appear in the retrieved sources or the generated answer."
            )
            logger.info(f"Tangential answer detected: {reason}")
            yield {
                "event": "deflected",
                "answer": (
                    f"The sources do not contain specific information about "
                    f"'{focus_term}' for this topic. While related papers were found, "
                    f"they do not address this particular detail."
                ),
                "reason": reason,
            }
            return

        citations = self._format_citations(relevant)
        yield {
            "event": "done",
            "answer": full_text.strip(),
            "sources": citations,
        }

    def print_answer(self, question: str, top_k: int | None = None):
        """Pretty-print a generated answer."""
        result = self.answer(question, top_k=top_k)

        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print(f"{'=' * 70}")

        if result["deflected"]:
            print(f"\n  DEFLECTED: {result['deflection_reason']}")
            print(f"\n{result['answer']}")
        else:
            print(f"\n{result['answer']}")
            print(f"\n  Retrieval scores: {[f'{s:.3f}' for s in result['scores']]}")

        print()
        return result
