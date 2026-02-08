# RAG-Bench

[![CI](https://github.com/nblomerus/rag-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/nblomerus/rag-bench/actions/workflows/ci.yml)

A Retrieval-Augmented Generation system for querying AI/ML research papers. Scrapes papers via the ArXiv API, chunks them with equation/table/acronym awareness, indexes with BGE embeddings in ChromaDB, and serves a web UI for asking questions with inline source citations and PDF highlighting.

## Architecture

```
frontend/index.html    React 18 + Tailwind single-page app
api/server.py          FastAPI REST API with SSE streaming
src/
  chunker.py           Section-aware chunking (~2048 chars, 200 overlap)
  embedder.py          BGE-base-en-v1.5 embeddings -> ChromaDB
  retriever.py         Dense vector retrieval
  hybrid_retriever.py  BM25 + dense + cross-encoder reranking
  generator.py         LLM answer generation (Ollama / OpenAI-compatible)
config/settings.py     All hyperparameters and paths
scripts/scrape_arxiv.py  ArXiv API scraper (PDF download + text extraction)
eval/                  Evaluation queries and results
```

## Setup

```bash
# Create environment
make pyenv
make deps

# Scrape papers from ArXiv
python scripts/scrape_arxiv.py                # ~50 landmark papers
python scripts/scrape_arxiv.py --mode extended # ~500 papers

# Run the indexing pipeline (chunk -> embed -> test)
python main.py
```

## Running

```bash
# Start the API server
python -m api.server

## Development

make ruff    # Format and lint
make test    # Run tests
make check   # Run pre-commit hooks
make clean   # Remove Python artifacts
```
