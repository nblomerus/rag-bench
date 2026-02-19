# RAG-Bench Quick Start Guide

## 🚀 Quick Start (5 minutes)

### Local Development

**Option 1: Quick Start (Automated)**

```bash
# One command setup - creates directories, environment, and starts everything
make deploy-dev
```

**Option 2: Manual Start**

```bash
# 1. Prerequisites installed?
docker --version && docker compose version

# 2. Start services
make docker-up

# 3. Open your browser
open http://localhost
```

Your RAG-Bench instance is now running with:
- **Frontend**: http://localhost
- **API**: http://localhost:8000
- **Ollama**: http://localhost:11434

> **Note**: Uses Docker Compose v2 (`docker compose`, not `docker-compose`)

### Production Deployment (ragbench.co.za)

```bash
# 1. Set up your server (Ubuntu 20.04+, 16GB+ RAM)

# 2. Clone and deploy
git clone <repo> /opt/ragbench && cd /opt/ragbench
make deploy

# 3. That's it! Your instance is at https://ragbench.co.za
```

## 📚 Full Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment & operations guide
- **[Architecture](#architecture)** - System design overview
- **Docker Commands**: `make help | grep docker`

## 🐳 Docker Quick Commands

```bash
# Quick Setup
make deploy-dev             # Automated dev environment setup

# Development
make docker-up              # Start services  
make docker-logs            # View logs
make docker-down            # Stop services
make docker-ps              # List running services
make docker-restart         # Restart all services

# Production  
make docker-up-prod         # Start with Nginx + SSL
make docker-logs-prod       # View production logs
make deploy                 # Full deployment
```

📌 **Note**: All commands use Docker Compose v2 (`docker compose`) under the hood.

## 📋 System Requirements

| Environment | CPU | RAM | Disk | Notes |
|------------|-----|-----|------|-------|
| **Local Dev** | 4+ cores | 16+ GB | 50+ GB | With Ollama locally |
| **Production** | 8+ cores | 32+ GB | 200+ GB | Handles 1000+ concurrent queries |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│  Domain: ragbench.co.za (HTTPS)                 │
├─────────────────────────────────────────────────┤
│  Nginx Reverse Proxy (SSL/TLS, caching)         │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐  ┌──────────────────────┐    │
│  │  Frontend   │  │  Backend API         │    │
│  │  React SPA  │→ │  FastAPI + RAG       │    │
│  │  (Nginx)    │← │  Pipeline            │    │
│  └─────────────┘  └─────────┬────────────┘    │
│                             │                 │
│                   ┌─────────▼──────┐          │
│                   │ ChromaDB       │          │
│                   │ Vector Index   │          │
│                   └────────────────┘          │
│                                               │
│                   ┌──────────────┐            │
│                   │ Ollama LLM   │            │
│                   │ (gemma2)     │            │
│                   └──────────────┘            │
└─────────────────────────────────────────────────┘
```

## 🔑 Key Features

✅ **Self-Hosted** - Full control, no external APIs (except optional LLM)
✅ **Production Ready** - SSL/TLS, auto-renewal, rate limiting, caching
✅ **Easy Deployment** - Single command: `make deploy`
✅ **Scalable** - Docker-based, can be deployed on any cloud
✅ **Secure** - Security headers, CORS, API rate limiting
✅ **Monitored** - Health checks, logging, performance metrics

## ⚙️ Configuration

Edit `.env` for configuration (created automatically by `make deploy-dev`):

```bash
RAG_LLM_BACKEND=ollama                    # LLM provider
RAG_LLM_MODEL=gemma2:27b                 # Model to use
DOMAIN=ragbench.co.za                     # Your domain
LETSENCRYPT_EMAIL=admin@ragbench.co.za    # SSL renewal
OLLAMA_PORT=11435                         # (Optional) Change if port conflicts
```

**Important**: Docker Compose looks for `.env` in the same directory as `docker-compose.yml`. Use `.env` not `.env.dev`.

## 📊 Monitoring

```bash
# View logs
make docker-logs-prod

# Check service status  
make docker-ps-prod

# View resource usage
docker stats
```

## 🐛 Troubleshooting

```bash
# Services won't start?
make docker-logs-prod

# SSL certificate issues?
docker compose -f docker-compose.prod.yml logs certbot

# Connection problems?
docker compose -f docker-compose.prod.yml exec api curl http://localhost:8000/api/health
```

## 📖 Full Documentation

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for:
- Detailed setup instructions
- Manual deployment steps
- SSL certificate configuration
- Backup & restore procedures
- Performance optimization
- Security best practices
- Troubleshooting guide

## 🆘 Need Help?

- **Docs**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Issues**: GitHub Issues on the project repo
- **Logs**: `make docker-logs` for troubleshooting

---

**Ready to deploy?** Start with `make deploy` and follow the prompts! 🚀
