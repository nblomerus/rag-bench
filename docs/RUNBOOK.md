# RAG-Bench Deployment Runbook

Quick reference guide for common deployment operations.

## Daily Operations

### Check Service Health

```bash
# Quick health check
curl https://ragbench.co.za/api/health

# Detailed status
cd /opt/ragbench
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs --tail=50
```

### View Logs

```bash
# All services
docker compose -f docker-compose.prod.yml logs -f

# Specific service
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f nginx

# Last 100 lines
docker compose -f docker-compose.prod.yml logs --tail=100
```

### Restart Service

```bash
cd /opt/ragbench

# Restart specific service
docker compose -f docker-compose.prod.yml restart api

# Restart all services
docker compose -f docker-compose.prod.yml restart

# Full rebuild and restart
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

## Deployment Operations

### Deploy New Version

```bash
cd /opt/ragbench

# Pull latest code
git fetch --all
git pull origin main

# Rebuild and deploy
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

# Verify
curl https://ragbench.co.za/api/health
```

### Rollback to Previous Version

```bash
cd /opt/ragbench

# List available versions
git tag -l "prod-v*"

# Rollback
./scripts/rollback.sh prod-v1.0.0

# Or manual rollback
git checkout tags/prod-v1.0.0
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

## Backup & Restore

### Create Manual Backup

```bash
cd /opt/ragbench
./scripts/backup.sh
```

### List Available Backups

```bash
ls -lh /opt/ragbench/backups/*.tar.gz
```

### Restore from Backup

```bash
cd /opt/ragbench

# List available backups
./scripts/restore.sh

# Restore specific backup
./scripts/restore.sh 20260215-143000
```

## Port Configuration

### Default Ports

| Service | Port | Purpose |
|---------|------|---------|
| Frontend | 80 | Web UI |
| API | 8000 | REST API |
| Ollama | 11434 | LLM inference (configurable) |
| Nginx (Prod) | 443 | HTTPS |

### Resolving Port Conflicts

If you get `address already in use`:

```bash
# Check what's using a port
lsof -i :11434
docker ps | grep ollama

# Option 1: Change Ollama port in .env
OLLAMA_PORT=11435

# Option 2: Stop conflicting service
docker stop <container_id>

# Option 3: Use external Ollama
RAG_LLM_BASE_URL=http://host.docker.internal:11434
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker compose -f docker-compose.prod.yml logs api

# Check container status
docker compose -f docker-compose.prod.yml ps

# Check resources
df -h  # Disk space
free -h  # Memory
docker stats  # Container resources
```

### SSL Certificate Issues

```bash
# Check certificate
docker compose -f docker-compose.prod.yml exec nginx ls -la /etc/letsencrypt/live/

# Renew manually
docker compose -f docker-compose.prod.yml run --rm certbot renew

# Restart nginx
docker compose -f docker-compose.prod.yml restart nginx
```

### Database Issues

```bash
# Check ChromaDB size
du -sh /opt/ragbench/chroma_db

# Backup before any fixes
./scripts/backup.sh

# Rebuild ChromaDB (WARNING: Data loss)
docker compose -f docker-compose.prod.yml down
rm -rf chroma_db/
docker compose -f docker-compose.prod.yml up -d
# Then re-ingest data
```

### Out of Memory

```bash
# Check memory usage
free -h
docker stats

# Restart Ollama (largest memory consumer)
docker compose -f docker-compose.prod.yml restart ollama

# Or restart all
docker compose -f docker-compose.prod.yml restart
```

### Slow Queries

```bash
# Check Ollama models
docker compose -f docker-compose.prod.yml exec ollama ollama list

# Check system load
uptime
top

# Check disk I/O
iostat -x 1
```

## Monitoring

### Set Up Health Monitor (One-time)

```bash
# Copy systemd service
sudo cp /opt/ragbench/systemd/ragbench-health-monitor.service /etc/systemd/system/

# Edit if needed
sudo nano /etc/systemd/system/ragbench-health-monitor.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable ragbench-health-monitor
sudo systemctl start ragbench-health-monitor

# Check status
sudo systemctl status ragbench-health-monitor
```

### Set Up Automated Backups (One-time)

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * cd /opt/ragbench && ./scripts/backup.sh >> /opt/ragbench/logs/backup.log 2>&1
```

## Maintenance

### Update Docker Images

```bash
cd /opt/ragbench

# Pull base images
docker compose -f docker-compose.prod.yml pull ollama

# Rebuild
docker compose -f docker-compose.prod.yml build --no-cache

# Deploy
docker compose -f docker-compose.prod.yml up -d
```

### Clean Up Docker

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup (careful!)
docker system prune -a
```

### Update SSL Certificate (Automatic)

Certbot automatically renews. To force renewal:

```bash
docker compose -f docker-compose.prod.yml run --rm certbot renew --force-renewal
docker compose -f docker-compose.prod.yml restart nginx
```

## Emergency Procedures

### Service Completely Down

1. Check if server is reachable: `ping ragbench.co.za`
2. SSH into server
3. Check Docker: `docker ps`
4. Check services: `cd /opt/ragbench && docker compose -f docker-compose.prod.yml ps`
5. Check logs: `docker compose -f docker-compose.prod.yml logs`
6. Restart: `docker compose -f docker-compose.prod.yml restart`
7. If still down, restore from backup: `./scripts/restore.sh`

### Data Corruption

1. Stop services: `docker compose -f docker-compose.prod.yml down`
2. Backup current state: `./scripts/backup.sh`
3. Restore from last known good backup: `./scripts/restore.sh <timestamp>`
4. Verify: `curl https://ragbench.co.za/api/health`

### Security Breach

1. Stop all services immediately: `docker compose -f docker-compose.prod.yml down`
2. Block attacker IP at firewall: `sudo ufw deny from <IP>`
3. Review logs: Check `/opt/ragbench/logs/` and `docker compose logs`
4. Rotate secrets: Update `.env.prod` with new API keys
5. Restore from pre-breach backup if needed
6. Update and redeploy: `git pull && docker compose -f docker-compose.prod.yml build && docker compose -f docker-compose.prod.yml up -d`

## Performance Tuning

### Increase Ollama Memory

Edit `docker-compose.prod.yml`:
```yaml
ollama:
  environment:
    - OLLAMA_NUM_GPU=1  # If GPU available
  deploy:
    resources:
      limits:
        memory: 8G
```

### Scale API Workers

Edit `.env.prod`:
```bash
WORKERS=4  # Increase from 1
```

Restart:
```bash
docker compose -f docker-compose.prod.yml restart api
```

## Contact Information

- **Primary Admin**: admin@ragbench.co.za
- **GitHub Repo**: https://github.com/nblomerus/rag-bench
- **Documentation**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **This Runbook**: [RUNBOOK.md](RUNBOOK.md)
