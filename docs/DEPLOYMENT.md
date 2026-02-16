# RAG-Bench Containerization & Deployment Guide

Complete guide for containerizing and self-hosting RAG-Bench with your domain `ragbench.co.za`.

## Overview

RAG-Bench is containerized into the following services:

- **Backend API** (`api`): FastAPI server with RAG pipeline
- **Frontend UI** (`frontend`): React SPA with Nginx
- **LLM Service** (`ollama`): Local LLM inference (Ollama)
- **Reverse Proxy** (`nginx`): SSL/TLS, caching, rate limiting (production only)
- **SSL Certificate Manager** (`certbot`): Automatic Let's Encrypt renewal (production only)

## Quick Start (Development)

### Prerequisites
- Docker and Docker Compose
- 16+ GB RAM (for LLM models)
- 100+ GB disk space (for models and data)

### 1. Clone and Setup

```bash
cd /opt/ragbench  # or your preferred location
git clone https://github.com/nblomerus/rag-bench.git .
cp .env.example .env.dev
```

### 2. Build and Start

```bash
# Start all services (development mode)
docker compose up -d

# Verify all services are running
docker compose ps

# View logs
docker compose logs -f api
docker compose logs -f frontend
docker compose logs -f ollama
```

### 3. Test the Application

```bash
# Check API health
curl http://localhost:8000/api/health

# Access frontend
open http://localhost

# View logs
docker compose logs -f
```

### 4. Usage

The development deployment exposes:
- **Frontend**: http://localhost
- **API**: http://localhost:8000
- **Ollama**: http://localhost:11434

## Production Deployment (Domain: ragbench.co.za)

### Prerequisites

1. **Server Requirements**
   - Ubuntu 20.04+ or similar Linux
   - 16+ GB RAM
   - 200+ GB disk space
   - Static public IP address
   - Port 80 and 443 open

2. **Domain Setup**
   - Domain: `ragbench.co.za`
   - DNS A record pointing to your server IP
   - For SSL auto-renewal: No other service on ports 80/443

3. **Installed Software**
   ```bash
   # Install Docker and Docker Compose
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   newgrp docker
   
   # Install Docker Compose (if not already included)
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker compose
   sudo chmod +x /usr/local/bin/docker compose
   ```

### Step 1: Deploy Using Script (Recommended)

```bash
# Clone the repository
git clone https://github.com/nblomerus/rag-bench.git /opt/ragbench
cd /opt/ragbench

# Make deploy script executable
chmod +x scripts/deploy.sh scripts/init-ssl.sh

# Run deployment
./scripts/deploy.sh

# This will:
# 1. Check prerequisites
# 2. Create deployment directories
# 3. Ask you to configure .env.prod
# 4. Build Docker images
# 5. Start all services
```

### Step 2: Initialize SSL Certificates

```bash
# Initialize Let's Encrypt certificates
cd /opt/ragbench
chmod +x scripts/init-ssl.sh
./scripts/init-ssl.sh
```

### Step 3: Verify Deployment

```bash
# Check all services are running
docker compose -f docker compose.prod.yml ps

# Test HTTPS access
curl https://ragbench.co.za/health
curl -k https://localhost/health  # If testing locally

# View logs
docker compose -f docker compose.prod.yml logs -f nginx
docker compose -f docker compose.prod.yml logs -f api
docker compose -f docker compose.prod.yml logs -f certbot
```

## Manual Deployment (Non-Scripted)

If you prefer manual control, follow these steps:

### 1. Initial Setup

```bash
# Create deployment directory
mkdir -p /opt/ragbench && cd /opt/ragbench

# Clone repository
git clone https://github.com/nblomerus/rag-bench.git .

# Create volume directories
mkdir -p chroma_db data logs certbot/conf certbot/www nginx/conf.d

# Copy and edit environment file
cp .env.example .env.prod
nano .env.prod  # Edit with your settings
```

### 2. Configure Environment (.env.prod)

```bash
# .env.prod
RAG_LLM_BACKEND=ollama
RAG_LLM_MODEL=mistral:7b-instruct-q4_K_M
RAG_LLM_BASE_URL=http://ollama:11434
DOMAIN=ragbench.co.za
LETSENCRYPT_EMAIL=admin@ragbench.co.za
LOG_LEVEL=INFO
```

### 3. Initialize SSL Certificates

```bash
# Create dummy certificate for initial nginx startup
docker run --rm \
    -v "$(pwd)/certbot/conf:/etc/letsencrypt" \
    -v "$(pwd)/certbot/www:/var/www/certbot" \
    certbot/certbot:latest \
    certonly \
    --webroot \
    -w /var/www/certbot \
    -d ragbench.co.za \
    -d www.ragbench.co.za \
    --email admin@ragbench.co.za \
    --agree-tos \
    --no-eff-email \
    --non-interactive
```

### 4. Build and Start Services

```bash
# Load environment variables
export $(cat .env.prod | grep -v '^#' | xargs)

# Build images
docker compose -f docker compose.prod.yml build

# Start all services
docker compose -f docker compose.prod.yml up -d

# Wait for services to initialize
sleep 30

# Verify services
docker compose -f docker compose.prod.yml ps
```

### 5. Verify HTTPS

```bash
# Test API
curl https://ragbench.co.za/api/health

# Or use your domain with curl
curl -v https://ragbench.co.za/
```

## Local Production Testing

Test production configuration on your dev machine before deploying:

```bash
# Switch to production mode locally
make switch-prod

# Access at http://localhost (port 80, via nginx)
curl http://localhost/api/health

# Switch back to development
make switch-dev

# Check current mode
make env-status
```

**Key differences from dev mode:**

| Feature | Development | Production (Local) |
|---------|------------|-------------------|
| Frontend URL | http://localhost:3000 | http://localhost |
| API URL | http://localhost:8000 | http://localhost/api |
| Hot Reload | Yes | No |
| Nginx | Not used | Used |

**Testing checklist before deploying:**
- Production build completes without errors
- All containers start successfully
- Health endpoint responds at `http://localhost/api/health`
- Frontend loads at `http://localhost`
- Can query and get responses

## Configuration Files

### .env.prod - Main Configuration

```bash
# Copy .env.example and edit these critical settings:

# LLM Configuration
RAG_LLM_BACKEND=ollama              # or openai, anthropic
RAG_LLM_MODEL=mistral:7b-instruct-q4_K_M
RAG_LLM_BASE_URL=http://ollama:11434

# Domain & SSL
DOMAIN=ragbench.co.za
LETSENCRYPT_EMAIL=admin@ragbench.co.za

# Logging
LOG_LEVEL=INFO
```

### docker compose.prod.yml

Primary orchestration file with:
- API service (FastAPI backend)
- Frontend service (React SPA)
- Nginx reverse proxy
- Certbot for SSL renewal
- Ollama LLM service

### nginx/nginx-prod.conf

Production Nginx configuration with:
- SSL/TLS setup with Let's Encrypt
- Security headers
- Rate limiting
- Caching
- GZIP compression
- SSE support for streaming

## Service Management

### View Logs

```bash
# All services
docker compose -f docker compose.prod.yml logs -f

# Specific service
docker compose -f docker compose.prod.yml logs -f api
docker compose -f docker compose.prod.yml logs -f nginx
docker compose -f docker compose.prod.yml logs -f certbot
```

### Restart Services

```bash
# Restart all
docker compose -f docker compose.prod.yml restart

# Restart specific service
docker compose -f docker compose.prod.yml restart api
docker compose -f docker compose.prod.yml restart frontend
```

### Stop Services

```bash
# Stop all services
docker compose -f docker compose.prod.yml stop

# Stop specific service
docker compose -f docker compose.prod.yml stop api
```

### Start Services

```bash
# Start all services
docker compose -f docker compose.prod.yml start

# Start specific service
docker compose -f docker compose.prod.yml start api
```

## Monitoring & Health Checks

### Health Endpoints

```bash
# API health
curl https://ragbench.co.za/api/health

# Frontend health
curl https://ragbench.co.za/health

# Check all service status
docker compose -f docker compose.prod.yml ps
```

### Log Analysis

```bash
# Recent errors
docker compose -f docker compose.prod.yml logs --tail=100 | grep ERROR

# Nginx access logs
docker exec ragbench-nginx tail -f /var/log/nginx/ragbench_access.log

# API logs
docker compose -f docker compose.prod.yml logs -f api --tail=50
```

## Updating

### Update Application Code

```bash
cd /opt/ragbench

# Stop services
docker compose -f docker compose.prod.yml down

# Update repository
git pull origin main

# Rebuild images
docker compose -f docker compose.prod.yml build

# Restart services
docker compose -f docker compose.prod.yml up -d
```

### Update Models/Data

```bash
# Update indexed data (from updated data files)
docker compose -f docker compose.prod.yml exec api python -m rag_bench.cli.pipeline

# Restart API to reload
docker compose -f docker compose.prod.yml restart api
```

## Backup & Restore

### Backup Data

```bash
cd /opt/ragbench

# Backup ChromaDB
tar -czf backups/chroma_db_$(date +%Y%m%d_%H%M%S).tar.gz chroma_db/

# Backup indexed data
tar -czf backups/data_$(date +%Y%m%d_%H%M%S).tar.gz data/

# Backup environment
cp .env.prod backups/.env.prod_$(date +%Y%m%d_%H%M%S)
```

### Restore Data

```bash
# Stop services
docker compose -f docker compose.prod.yml down

# Restore from backup
tar -xzf backups/chroma_db_20230115_120000.tar.gz

# Restart
docker compose -f docker compose.prod.yml up -d
```

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker compose -f docker compose.prod.yml logs

# Verify Docker daemon
docker ps

# Check disk space
df -h

# Check memory
free -h
```

### SSL Certificate Issues

```bash
# Check certificate status
docker compose -f docker compose.prod.yml logs certbot

# Manually renew certificate
docker compose -f docker compose.prod.yml exec certbot certbot renew --force-renewal

# Check certificate validity
docker exec ragbench-nginx openssl s_client -connect localhost:443 -servername ragbench.co.za
```

### API Connection Issues

```bash
# Check API health
docker compose -f docker compose.prod.yml exec api curl http://localhost:8000/api/health

# Verify network connectivity
docker compose -f docker compose.prod.yml exec frontend curl http://api:8000/api/health

# Check logs
docker compose -f docker compose.prod.yml logs api
```

### High Memory Usage

```bash
# Check resource usage
docker stats

# Check which service is using memory
docker compose -f docker compose.prod.yml stats

# Restart memory-heavy services (Ollama)
docker compose -f docker compose.prod.yml restart ollama
```

## Performance Optimization

### Nginx Caching

Edit `nginx/nginx-prod.conf` to enable caching:

```nginx
# Add to upstream section
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=ragbench_cache:10m;

# In location /api/ section
proxy_cache ragbench_cache;
proxy_cache_valid 200 1h;
add_header X-Cache-Status $upstream_cache_status;
```

### Database Optimization

```bash
# Monitor ChromaDB size
du -sh /opt/ragbench/chroma_db

# Optimize collection
docker compose -f docker compose.prod.yml exec api python -c \
  "from rag_bench.core.retriever import HybridRetriever; \
   r = HybridRetriever(); \
   r.client.delete_collection(name='ai_ml_papers'); \
   print('Collection optimized')"
```

## Security Best Practices

✅ **Implemented in Production Setup:**

- SSL/TLS with Let's Encrypt (auto-renewal)
- Security headers (HSTS, CSP, X-Frame-Options)
- Rate limiting on API endpoints
- Nginx reverse proxy with gzip
- No exposed internal services
- Automatic certificate renewal

🔒 **Additional Recommendations:**

```bash
# 1. Regular backups
0 2 * * * cd /opt/ragbench && tar -czf backups/chroma_db_$(date +\%Y\%m\%d).tar.gz chroma_db/

# 2. Monitor disk space
0 * * * * df -h | grep -v "^Filesystem" | awk '{if ($5 > 80) print $0}' | mail -s "Disk Alert" admin@ragbench.co.za

# 3. Update system regularly
apt-get update && apt-get upgrade -y

# 4. Use a firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

## Next Steps

1. **Configure LLM Model**: Choose and configure your preferred LLM
2. **Index Data**: Run the pipeline to index your papers
3. **Monitor Performance**: Check logs and metrics
4. **Set up Backups**: Implement automatic backup strategy
5. **Enable Monitoring**: Add Prometheus/Grafana for metrics

## Support & Documentation

- API Documentation: http://ragbench.co.za/docs (when running)
- GitHub Issues: https://github.com/nblomerus/rag-bench/issues
- Docker Documentation: https://docs.docker.com

## Environment Variables Reference

All environment variables available in `.env.example`:

```bash
# LLM Configuration
RAG_LLM_BACKEND          # ollama | openai | anthropic
RAG_LLM_MODEL            # Model identifier
RAG_LLM_BASE_URL         # API endpoint

# API Configuration
RAG_API_PORT             # Default: 8000
CORS_ORIGINS             # Comma-separated origins

# Database
CHROMA_DB                # ChromaDB path

# Domain & Security
DOMAIN                   # ragbench.co.za
LETSENCRYPT_EMAIL        # SSL certificate email

# Logging
LOG_LEVEL                # DEBUG | INFO | WARNING | ERROR

# Features
ENABLE_EVAL              # true | false
ENABLE_PDF_SERVING       # true | false
```

---

**Last Updated**: February 2026
**For questions or issues**: See GitHub issues or documentation
