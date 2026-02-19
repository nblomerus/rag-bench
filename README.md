# RAG-Bench

[![CI](https://github.com/nblomerus/rag-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/nblomerus/rag-bench/actions/workflows/ci.yml)

A self-hosted Retrieval-Augmented Generation system for querying AI/ML research papers. Scrapes papers via the ArXiv API, chunks with equation/table/acronym awareness, indexes with BGE embeddings in ChromaDB, and serves a web UI with source citations and PDF viewing.

## Quick Start

```bash
# Clone and start (Docker)
git clone https://github.com/nblomerus/rag-bench.git && cd rag-bench
cp .env.example .env
docker compose up -d

# Open http://localhost (frontend) or http://localhost:8000/docs (API)
```

See [QUICKSTART.md](docs/QUICKSTART.md) for the full 5-minute guide.

## Documentation

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](docs/QUICKSTART.md) | Get running in 5 minutes |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment & operations |
| [RUNBOOK.md](docs/RUNBOOK.md) | Daily ops, troubleshooting, emergencies |
| [DEV_ENVIRONMENT.md](docs/DEV_ENVIRONMENT.md) | Local development setup & workflow |
| [DEPLOYMENT_CHECKLIST.md](docs/DEPLOYMENT_CHECKLIST.md) | Interactive deployment checklist |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Frontend (React 18 + Vite + Tailwind)          │
│  - Question interface with streaming responses  │
│  - Source citations with PDF viewing            │
└──────────────┬──────────────────────────────────┘
               │ HTTP / HTTPS
┌──────────────▼──────────────────────────────────┐
│  Nginx Reverse Proxy (production only)          │
│  - SSL/TLS termination (Let's Encrypt)          │
│  - Rate limiting & caching                      │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  FastAPI Backend                                │
│  - Hybrid retrieval (BM25 + dense + reranking)  │
│  - LLM answer generation with citation boost    │
│  - Relevance gating & query deflection          │
└─────────┬─────────────────┬─────────────────────┘
          │                 │
    ┌─────▼─────┐    ┌─────▼──────┐
    │ ChromaDB  │    │ Ollama LLM │
    │ (BGE      │    │ (or OpenAI │
    │  vectors) │    │  /Anthropic)│
    └───────────┘    └────────────┘
```

**Project layout:**

```
rag_bench/
├── api/          FastAPI server & Pydantic schemas
├── core/         RAG pipeline (chunker, embedder, retriever, generator)
├── cli/          Pipeline & query CLI tools
├── eval/         Evaluation queries & results
├── utils/        Text processing utilities
└── config.py     All configuration & hyperparameters
frontend/         React SPA (Vite build)
scripts/          ArXiv scraper, deployment, backup/restore
tests/            Unit tests (90%+ coverage enforced)
```

## Setup

### Requirements

- Docker & Docker Compose v2
- 16+ GB RAM (for LLM models)
- 100+ GB disk space (for models and data)
- Optional: NVIDIA GPU with CUDA for faster embeddings

### Local Development

```bash
cp .env.example .env
make docker-up       # Start all services
make docker-logs     # View logs
```

- **Frontend**: http://localhost
- **API**: http://localhost:8000
- **API docs**: http://localhost:8000/docs
- **Ollama**: http://localhost:11434

### Production

```bash
ssh user@your-server
git clone https://github.com/nblomerus/rag-bench.git /opt/ragbench
cd /opt/ragbench
make deploy          # Guided deployment with SSL setup
```

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for full instructions.

## Data Pipeline

### 1. Scrape papers from ArXiv

```bash
python scripts/scrape_arxiv.py                  # ~170 landmark papers
python scripts/scrape_arxiv.py --mode extended  # ~1500+ across 18 topics
python scripts/scrape_arxiv.py --mode abstracts # ~5000+ abstracts (fast)
```

### 2. Index papers (chunk, embed, store)

```bash
python -m rag_bench.cli.pipeline               # Scraped papers only
python -m rag_bench.cli.pipeline --hybrid      # Merge scraped + HuggingFace
python -m rag_bench.cli.pipeline --step chunk  # Run a single step
```

Steps: ingest → chunk (1024 char, 128 overlap) → embed (BGE) → index (ChromaDB) → smoke test

### 3. Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a transformer?", "top_k": 5}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/query` | Query papers, get grounded answer with citations |
| POST | `/api/query/stream` | Same with SSE streaming |
| GET | `/api/stats` | Corpus statistics |
| GET | `/api/health` | Health check |
| GET | `/api/papers` | List all indexed papers |
| GET | `/api/papers/{id}` | Get chunks for a paper |
| GET | `/api/papers/{id}/pdf` | Fetch & cache ArXiv PDF |
| POST | `/api/eval` | Run evaluation suite |

Interactive docs at `/docs` (Swagger) or `/redoc` when running.

## Configuration

Key settings in `.env` (see [.env.example](.env.example) for all options):

```bash
RAG_LLM_BACKEND=ollama                    # ollama | openai | anthropic
RAG_LLM_MODEL=gemma2:27b
RAG_LLM_BASE_URL=http://ollama:11434
DOMAIN=ragbench.co.za                     # Production only
```

## Retrieval Pipeline

Three-stage hybrid retrieval:

1. **First stage** — BM25 (sparse, 30% weight) + BGE dense embeddings (70% weight), fused with Reciprocal Rank Fusion → top 200 candidates
2. **Reranking** — Cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) scores top 100 candidates
3. **Generation** — LLM generates answer with citation boost for foundational papers; queries below relevance threshold (0.3) are deflected

## Development

```bash
make test           # Run pytest (90%+ coverage required)
make ruff           # Format & lint
make check          # Pre-commit hooks
make clean          # Remove Python artifacts
```

## Make Targets

```bash
# Docker
make docker-up / docker-down / docker-logs / docker-ps

# Production
make deploy              # Full deployment with SSL
make prod-start / prod-stop / prod-restart / prod-rebuild
make prod-ingest         # Ingest papers on production
make prod-logs / prod-status

# Operations
make backup / restore / rollback
make switch-dev / switch-prod / env-status
```

Run `make help` for the complete list.

## License

See [LICENSE](LICENSE).
