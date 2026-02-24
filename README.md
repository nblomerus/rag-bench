# RAG-Bench

[![CI](https://github.com/nblomerus/rag-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/nblomerus/rag-bench/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.0.19-blue)

A self-hosted RAG pipeline for querying 500+ AI/ML research papers. Hybrid retrieval (BM25 + dense embeddings + cross-encoder reranking), LLM generation with inline source citations, in-browser PDF viewing, and a built-in evaluation framework — all running on a single machine.

## Features

- **Hybrid retrieval** — BM25 sparse + BGE dense embeddings fused with Reciprocal Rank Fusion, then reranked by a cross-encoder (`BAAI/bge-reranker-v2-m3`)
- **Grounded answers** — LLM generates responses with `[Source N]` citations mapped to specific paper chunks; citation boost for foundational papers
- **Evaluation suite** — Built-in benchmarks (RAG-Bench + RAGTruth) with retrieval, citation, faithfulness, and relevance metrics
- **Production dashboard** — Live latency percentiles, pipeline breakdown, hardware monitoring (CPU/RAM/GPU), with configurable time ranges
- **Relevance gating** — Low-confidence queries are deflected instead of hallucinated
- **Self-contained** — Docker Compose brings up the full stack; no external services required

## Quick Start

```bash
git clone https://github.com/nblomerus/rag-bench.git && cd rag-bench
cp .env.example .env
docker compose up -d
```

Open http://localhost:3000 (frontend) or http://localhost:8000/docs (API docs).

See [QUICKSTART.md](docs/QUICKSTART.md) for the full 5-minute guide.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Frontend (React 18 + Vite + Tailwind)              │
│  Ask tab · Benchmarks tab · Production dashboard    │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP / SSE
┌──────────────────▼──────────────────────────────────┐
│  Nginx (production only)                            │
│  SSL termination · static caching                   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  FastAPI Backend                                    │
│  Hybrid retrieval · LLM generation · citation boost │
│  Evaluation runner · Prometheus metrics             │
└──────────┬───────────────────┬──────────────────────┘
           │                   │
     ┌─────▼──────┐     ┌─────▼──────┐
     │  ChromaDB  │     │   Ollama   │
     │ BGE embed  │     │  (or cloud │
     │ + BM25     │     │   LLM API) │
     └────────────┘     └────────────┘
```

```
rag_bench/
├── api/             FastAPI server & Pydantic schemas
├── core/            RAG pipeline — chunker, embedder, retriever, generator
├── eval/            Benchmarks, metrics, judge LLM, report generation
├── observability/   Prometheus metrics, structured logging, request tracker
├── cli/             Pipeline & query CLI tools
├── utils/           Text processing utilities
└── config.py        All configuration & hyperparameters
frontend/            React SPA (Vite + Tailwind)
scripts/             ArXiv scraper, deployment, backup/restore
tests/               Unit tests (90%+ coverage enforced)
```

## How It Works

A query flows through five stages — from HTTP request to streamed, cited response. Every numeric threshold below is the actual production value from `config.py`.

### Ingestion: Building the Corpus

Before any queries, papers are scraped, chunked, embedded, and indexed.

**Data sources.** The primary corpus comes from a combination of around ~19 000 articles scraped from ArXiv and the HuggingFace `jamescalam/ai-arxiv2` dataset (train split).

```bash
python scripts/scrape_arxiv.py                  # ~170 landmark papers
python scripts/scrape_arxiv.py --mode extended  # ~15000+ across 18 topics
python -m rag_bench.cli.pipeline --hybrid       # Chunk, embed, index
```

**Chunking.** Each paper is split into 1024-character chunks with 128-character overlap. Before splitting, equations (`$$...$$`, `\begin{equation}`, `\begin{align}`) are replaced with placeholders so they're never broken across chunk boundaries — same for multi-row tables detected via `| ... |` patterns. Noise sections (references, acknowledgments, ethics statements, funding) are filtered out. Each chunk gets a contextual prefix — `"{Title} — {Section}\n\n"` — prepended to improve embedding quality.

**Acronym expansion.** A per-paper pass extracts all ALL-CAPS terms with parenthetical definitions (e.g., "Maximum Inner Product Search (MIPS)"). The first occurrence of each acronym per chunk is expanded inline so the embeddings capture the full meaning.

**Embedding.** Chunks are embedded with `BAAI/bge-base-en-v1.5` (768-dim) in batches of 254, L2-normalized, and stored in ChromaDB with HNSW indexing using cosine distance. Each chunk carries metadata: `chunk_id`, `arxiv_id`, `section`, `source_display` (authors, year, title), and paper categories.

### Stage 1: Hybrid Retrieval

Two retrievers run in parallel and their results are fused.

**BM25 (sparse).** Okapi BM25 (`k1=1.5`, `b=0.75`) over all chunks. Queries are tokenized to lowercase alphanumeric tokens with 100+ stopwords removed (standard English + ML-specific: "model", "method", "approach", "results", "proposed"). Top 200 candidates are selected via numpy `argpartition` in O(n).

**Dense retrieval.** The query is prefixed with `"Represent this sentence for searching relevant passages: "` (BGE requirement), embedded with the same `bge-base-en-v1.5` model, and searched against the ChromaDB HNSW index. Returns the top 200 nearest neighbors by cosine similarity.

**Reciprocal Rank Fusion.** Both ranked lists are merged by chunk ID using RRF with `k=60`:

```
RRF_score(chunk) = 0.3 / (60 + rank_bm25 + 1) + 0.7 / (60 + rank_dense + 1)
```

BM25 gets 30% weight (lower precision at scale), dense gets 70%. The fused list is sorted by combined RRF score.

### Stage 2: Citation Boost

An optional stage that injects chunks from foundational papers when the query warrants it.

**Intent classification.** The query is classified as *foundational* ("what is", "explain", "how does", "original paper"), *recent* ("SOTA", "latest", "2024"), or *balanced*. This is pattern-based, no LLM call.

**Paper injection.** 27 manually curated landmark papers (Attention Is All You Need, BERT, GPT-3, LoRA, DDPM, etc.) each have a boost factor from 1.3x to 2.0x. For foundational queries, the 3 most relevant chunks per matching paper are injected into the candidate pool (up to 25% of the rerank pool size). An age-based multiplier further adjusts scores — for foundational intent, papers from 2017 or earlier get 1.8x while recent papers get 1.0x; for recent intent, the curve inverts. Boosted papers are guaranteed to meet the 75th percentile score floor.

### Stage 3: Reranking

The top 100 fused candidates are rescored by a cross-encoder (`BAAI/bge-reranker-v2-m3`). Each (query, passage) pair is scored jointly — cross-encoder output ranges from roughly -10 to +10. The final top-k results (default 10) are returned to the generator. If the cross-encoder is unavailable, a keyword-overlap + phrase-bonus heuristic fills in.

### Stage 4: Relevance Gating

Before generation, a relevance gate decides whether to answer or deflect.

**Thresholds** (auto-calibrated based on whether scores come from the cross-encoder or cosine):

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Min top score | 0.5 (cross-encoder) / 0.3 (cosine) | At least one strong result |
| Score concentration | 0.15 | Results aren't uniformly mediocre |
| Keyword overlap | 0.25 | Query terms actually appear in passages |
| Min relevant chunks | 1 | At least one passage passes all checks |

**Additional checks.** Queries asking for specific claims ("X achieved Y% accuracy") trigger false-premise detection — the system verifies that focal terms from the query appear 3+ times in the retrieved passages and that any referenced numbers actually exist. If checks fail, the query is deflected with a reason: *"I don't have sufficient information in my knowledge base to answer this accurately. {reason}"*

### Stage 5: Generation

**Prompt construction.** The LLM receives a system prompt with 11 rules enforcing citation discipline (`[Source N]` only for supported claims, no hallucination, LaTeX math in `$...$`), followed by the retrieved passages formatted as:

```
[1] Vaswani et al. (2017) "Attention Is All You Need" — §3.2 — "The scaled dot-product..."
[2] Devlin et al. (2019) "BERT" — §2 — "We introduce a new language..."
...
```

The LLM generates token by token, streamed to the client via SSE. Three event types are emitted: `sources` (retrieval results, sent first), `token` (each generated token), and `done` (final answer + quality metrics).

**Math post-processing.** Five regex passes clean up the LLM's math output: Greek Unicode → LaTeX (`α` → `\alpha`), math symbols (`∑` → `\sum`), subscripts (`x_t` → `$x_{t}$`), superscripts (`x^2` → `$x^{2}$`), with existing `$...$` blocks preserved.

**Quality metrics** (computed per response, no LLM needed):

| Metric | How |
|--------|-----|
| Retrieval confidence | Score gap ratio between top-1 and top-2 (high/medium/low) |
| Citation coverage | % of provided sources actually cited in the answer |
| Citation density | `[Source N]` markers per 100 words |
| Unsupported claims | Heuristic count of claims without adjacent citations |
| Faithfulness score | Content-word overlap between answer and source passages (1–5) |
| Source diversity | Unique papers and sections represented |

### Parameters

All tunable values live in `config.py`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CHUNK_SIZE_CHARS` | 1024 | Characters per chunk |
| `CHUNK_OVERLAP_CHARS` | 128 | Overlap between chunks |
| `FIRST_STAGE_K` | 200 | Candidates per retriever |
| `RERANK_CANDIDATES` | 100 | Candidates sent to cross-encoder |
| `BM25_WEIGHT` | 0.3 | RRF sparse weight |
| `DENSE_WEIGHT` | 0.7 | RRF dense weight |
| `DEFAULT_TOP_K` | 10 | Final results to generator |
| `RELEVANCE_THRESHOLD` | 0.3 | Min score for relevance gate |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Dense encoder (768-dim) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `EMBEDDING_BATCH_SIZE` | 254 | GPU batch size for indexing |

## Data Pipeline

```bash
# 1. Scrape papers from ArXiv
python scripts/scrape_arxiv.py                  # ~170 landmark papers
python scripts/scrape_arxiv.py --mode extended  # ~1500+ across 18 topics

# 2. Chunk, embed, and index
python -m rag_bench.cli.pipeline                # Scraped papers only
python -m rag_bench.cli.pipeline --hybrid       # Merge scraped + HuggingFace

# 3. Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a transformer?", "top_k": 5}'
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/query` | Query with grounded answer + citations |
| `POST` | `/api/query/stream` | Same, with SSE streaming |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | Corpus statistics |
| `GET` | `/api/papers` | List indexed papers |
| `GET` | `/api/papers/{id}` | Paper chunks |
| `GET` | `/api/papers/{id}/pdf` | Fetch & cache ArXiv PDF |
| `POST` | `/api/eval` | Run evaluation suite |
| `GET` | `/api/metrics/summary` | Live Prometheus-style metrics |

Interactive docs at `/docs` (Swagger) or `/redoc`.

## Configuration

Key settings in `.env` (see [.env.example](.env.example)):

```bash
RAG_LLM_BACKEND=ollama          # ollama | openai | anthropic
RAG_LLM_MODEL=gemma2:27b
RAG_LLM_BASE_URL=http://ollama:11434
DOMAIN=ragbench.co.za           # Production only
```

## Setup

**Requirements:** Docker & Docker Compose v2, 16 GB+ RAM, 100 GB+ disk. NVIDIA GPU optional.

```bash
# Local development
cp .env.example .env
make docker-up

# Production
make deploy              # Guided deployment with SSL
```

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for production setup.

## Development

```bash
make test            # pytest (90%+ coverage required)
make ruff            # Format & lint
make check           # Pre-commit hooks
```

Run `make help` for the full list of targets.

## Docs

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](docs/QUICKSTART.md) | Get running in 5 minutes |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment & operations |
| [RUNBOOK.md](docs/RUNBOOK.md) | Daily ops, troubleshooting, emergencies |
| [DEV_ENVIRONMENT.md](docs/DEV_ENVIRONMENT.md) | Local development setup |
| [DEPLOYMENT_CHECKLIST.md](docs/DEPLOYMENT_CHECKLIST.md) | Interactive deployment checklist |

## License

MIT. See [LICENSE](LICENSE).
