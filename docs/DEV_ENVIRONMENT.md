# Development Environment Guide

**Complete guide for setting up and working with RAG-Bench locally**

---

## 🚀 Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/nblomerus/rag-bench.git
cd rag-bench

# 2. Create environment file
cp .env.example .env
# Edit .env if needed (defaults work for most setups)

# 3. Start development environment
make deploy-dev

# 4. Access application
# Frontend: http://localhost:3000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

That's it! 🎉

---

## 📋 Prerequisites

- **Docker** (20.10+) & **Docker Compose v2**
- **Git**
- **16GB RAM** (for LLM models)
- **50GB disk space** (for models and data)

**Optional for local Python development**:
- Python 3.11+
- pyenv (recommended)
- make

---

## 🛠️ Setup Options

### Option 1: Docker Only (Recommended for Quick Start)

**Best for**: Testing, frontend development, quick contributions

```bash
# Clone and start
git clone https://github.com/nblomerus/rag-bench.git
cd rag-bench
make deploy-dev

# That's it! Everything runs in containers
```

**Advantages**:
- ✅ No Python setup needed
- ✅ Consistent environment
- ✅ Quick to start
- ✅ Matches production

**Development workflow**:
```bash
# View logs
make docker-logs

# Restart after backend changes
docker compose restart api

# Access container shell
docker compose exec api bash

# Run commands in container
docker compose exec api python -m rag_bench.cli.query "test query"
```

### Option 2: Local Python + Docker (Advanced)

**Best for**: Backend development, debugging, testing

```bash
# 1. Setup Python environment
make pyenv  # Creates pyenv virtualenv
make install  # Installs dependencies

# 2. Start supporting services (Ollama, ChromaDB)
docker compose up -d ollama

# 3. Run API locally
make run-server

# 4. In another terminal, start frontend in Docker
docker compose up -d frontend
```

**Advantages**:
- ✅ Fast iteration (no container rebuild)
- ✅ Easy debugging (use your IDE)
- ✅ Run tests locally
- ✅ Better code completion

**Development workflow**:
```bash
# Make changes to Python code
# API auto-reloads (FastAPI hot-reload)

# Run tests
make test

# Format code
make ruff

# Run pre-commit checks
make check
```

---

## 🔧 Configuration

### Environment Files

| File | Purpose | Location | Committed? |
|------|---------|----------|------------|
| `.env.example` | Template with all options | Repo | ✅ Yes |
| `.env` | Your local dev config | Local only | ❌ No (gitignored) |
| `.env.dev` | Alternative dev config | Local only | ❌ No (gitignored) |
| `.env.prod` | Production config | Server only | ❌ No (gitignored) |
| `.env.production.template` | Production template | Repo | ✅ Yes |

### Key Environment Variables

**LLM Configuration** (most important):
```bash
# Use Ollama (default, runs in Docker)
RAG_LLM_BACKEND=ollama
RAG_LLM_MODEL=mistral:7b-instruct-q4_K_M
RAG_LLM_BASE_URL=http://ollama:11434

# Or use OpenAI
RAG_LLM_BACKEND=openai
RAG_LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-xxx
```

**Common Adjustments**:
```bash
# Increase logging
LOG_LEVEL=DEBUG

# Use different Ollama model
RAG_LLM_MODEL=llama2:7b

# Change Ollama port (if conflict)
OLLAMA_PORT=11435

# Disable citation boost (for testing)
ENABLE_CITATION_BOOST=false
```

---

## 📁 Project Structure

```
rag-bench/
├── rag_bench/              # Main Python package
│   ├── api/                # FastAPI server
│   │   ├── server.py       # API entry point
│   │   └── schemas.py      # Request/response models
│   ├── core/               # Core RAG logic
│   │   ├── chunker.py      # Document chunking
│   │   ├── embedder.py     # Embedding generation
│   │   ├── retriever.py    # Vector search
│   │   └── generator.py    # LLM generation
│   ├── cli/                # Command-line tools
│   └── eval/               # Evaluation scripts
│
├── frontend/               # React application
│   ├── src/
│   │   ├── App.jsx         # Main component
│   │   └── components/
│   └── Dockerfile
│
├── tests/                  # Test suite
│   └── unit_tests/
│
├── scripts/                # Deployment & ops scripts
│   ├── deploy-dev.sh
│   ├── backup.sh
│   ├── restore.sh
│   └── rollback.sh
│
├── data/                   # Data directory (gitignored)
│   └── pdfs/               # PDF papers
│
├── chroma_db/              # Vector database (gitignored)
│
├── docker-compose.yml      # Dev environment
├── docker-compose.prod.yml # Production environment
├── Dockerfile              # Backend image
├── Makefile                # Common commands
└── requirements.txt        # Python dependencies
```

---

## 🔄 Development Workflow

### 1. Starting Your Day

```bash
# Update code
git pull origin main

# Start/restart services
make docker-up

# Check status
make docker-ps

# View logs
make docker-logs
```

### 2. Making Changes

**Backend Changes** (Python):
```bash
# Edit files in rag_bench/
# Container auto-reloads (if using Docker)

# If changes don't reload:
docker compose restart api

# Run tests
make test

# Format code
make ruff
```

**Frontend Changes** (React):
```bash
# Edit files in frontend/src/
# Vite auto-reloads in browser

# If changes don't appear:
docker compose restart frontend
```

**Configuration Changes**:
```bash
# Edit .env file
# Restart affected services
docker compose restart api
```

### 3. Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit_tests/test_retriever.py

# Run with coverage
pytest --cov --cov-report=html

# View coverage report
open htmlcov/index.html
```

### 4. Committing Changes

```bash
# Format code
make ruff

# Run pre-commit checks
make check

# Commit
git add .
git commit -m "feat: add new feature"
git push origin feature-branch
```

### 5. Ending Your Day

```bash
# Stop containers (keeps data)
make docker-down

# Or stop and cleanup everything
make docker-clean
```

---

## 🐛 Debugging

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f frontend
docker compose logs -f ollama

# Last 100 lines
docker compose logs --tail=100 api
```

### Access Container Shell

```bash
# API container
docker compose exec api bash

# Inside container, you can:
python -m rag_bench.cli.query "test query"
ls -la chroma_db/
cat /app/logs/rag_bench.log
```

### Check Service Health

```bash
# API health endpoint
curl http://localhost:8000/api/health

# Ollama
curl http://localhost:11434/api/tags

# Frontend
curl -I http://localhost:3000
```

### Common Issues

**Issue: Container won't start**
```bash
# Check logs
docker compose logs api

# Check if port is in use
sudo lsof -i :8000

# Rebuild container
docker compose build api
docker compose up -d api
```

**Issue: Out of memory**
```bash
# Check Docker memory
docker stats

# Restart Ollama (largest consumer)
docker compose restart ollama

# Or increase Docker memory:
# Docker Desktop → Settings → Resources → Memory → 16GB+
```

**Issue: ChromaDB corruption**
```bash
# Backup first!
cp -r chroma_db chroma_db.backup

# Remove and rebuild
rm -rf chroma_db
docker compose restart api

# Re-ingest data
docker compose exec api python -m rag_bench.cli.pipeline --ingest
```

---

## 🎯 Best Practices

### Git Workflow

```bash
# Always work on a branch
git checkout -b feature/new-feature

# Keep branch up to date
git checkout main
git pull origin main
git checkout feature/new-feature
git rebase main

# Push and create PR
git push origin feature/new-feature
# Then create PR on GitHub
```

### Code Quality

```bash
# Before committing, always:
make ruff      # Format and lint
make test      # Run tests
make check     # Pre-commit checks
```

### Docker Management

```bash
# Clean up regularly
docker system prune

# Remove unused volumes
docker volume prune

# Full cleanup (careful - removes all data)
make docker-clean
```

### Environment Management

✅ **DO**:
- Keep `.env` gitignored
- Use `.env.example` as template
- Document any new env variables
- Use sensible defaults

❌ **DON'T**:
- Commit `.env` files
- Hardcode secrets in code
- Change `.env.example` without documenting

---

## 🚢 Testing Changes in Production-Like Environment

Before deploying to production, test in production mode locally:

```bash
# 1. Create .env.prod locally
cp .env.production.template .env.prod
# Edit .env.prod:
# - DOMAIN=localhost
# - LOG_LEVEL=DEBUG

# 2. Build production images
docker compose -f docker-compose.prod.yml build

# 3. Start production stack (without SSL)
docker compose -f docker-compose.prod.yml up -d

# 4. Test
curl http://localhost/api/health

# 5. Cleanup
docker compose -f docker-compose.prod.yml down
```

---

## 📚 Common Tasks

### Ingest New Papers

```bash
# Add PDFs to data/pdfs/
cp ~/Downloads/*.pdf data/pdfs/

# Ingest
docker compose exec api python -m rag_bench.cli.pipeline --ingest

# Verify
docker compose exec api python -m rag_bench.cli.query "test query"
```

### Change LLM Model

```bash
# Edit .env
RAG_LLM_MODEL=llama2:7b

# Restart API
docker compose restart api

# Download model (if using Ollama)
docker compose exec ollama ollama pull llama2:7b
```

### Reset Database

```bash
# Backup first!
cp -r chroma_db chroma_db.backup

# Stop services
docker compose down

# Remove database
rm -rf chroma_db

# Restart
docker compose up -d

# Re-ingest
docker compose exec api python -m rag_bench.cli.pipeline --ingest
```

### Update Dependencies

```bash
# Edit requirements.in
echo "new-package==1.0.0" >> requirements.in

# Compile
make upgrade

# Install
make install

# Or in Docker, rebuild
docker compose build api
docker compose up -d api
```

---

## 🔗 Resources

- **Documentation**: 
  - [README.md](README.md) - Project overview
  - [QUICKSTART.md](QUICKSTART.md) - Quick start guide
  - [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
  
- **Tools**:
  - FastAPI docs: http://localhost:8000/docs
  - Ollama API: http://localhost:11434
  
- **External**:
  - [FastAPI Documentation](https://fastapi.tiangolo.com/)
  - [Docker Compose Reference](https://docs.docker.com/compose/)
  - [Ollama Documentation](https://ollama.ai/docs)
  - [ChromaDB Documentation](https://docs.trychroma.com/)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## ❓ Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check docs/ directory
- **Community**: Join discussions on GitHub

---

**Happy Coding! 🚀**
