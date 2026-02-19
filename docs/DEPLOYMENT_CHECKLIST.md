# 🚀 RAG-Bench Deployment Checklist

Use this checklist to guide your first production deployment to `ragbench.co.za`.

---

## ☑️ Pre-Deployment (Day 1 Morning)

### Server Setup
- [ ] Server provisioned
  - [ ] Ubuntu 20.04+ installed
  - [ ] 16GB+ RAM
  - [ ] 200GB+ disk space
  - [ ] Static public IP assigned: `______________`
  
### DNS Configuration
- [ ] Domain registered: `ragbench.co.za`
- [ ] DNS A record created: `ragbench.co.za` → Server IP
- [ ] DNS propagated (verify with `dig ragbench.co.za`)

### Server Access
- [ ] SSH access configured
- [ ] SSH key added to server
- [ ] Can connect: `ssh user@ragbench.co.za`
- [ ] User added to docker group: `sudo usermod -aG docker $USER`

### Firewall
- [ ] Port 22 (SSH) open
- [ ] Port 80 (HTTP) open  
- [ ] Port 443 (HTTPS) open
- [ ] All other ports closed

### Software Installation
- [ ] Docker installed (`docker --version`)
- [ ] Docker Compose v2 installed (`docker compose version`)
- [ ] Git installed (`git --version`)

---

## ☑️ Initial Deployment (Day 1 Afternoon)

### Repository Setup
- [ ] Created deployment directory: `/opt/ragbench`
- [ ] Repository cloned: `git clone https://github.com/nblomerus/rag-bench.git /opt/ragbench`
- [ ] Branch confirmed: `git branch` shows `main`

### Environment Configuration
- [ ] Created `.env.prod` from template:
  ```bash
  cd /opt/ragbench
  cp .env.production.template .env.prod
  nano .env.prod
  ```
- [ ] Set `DOMAIN=ragbench.co.za`
- [ ] Set `LETSENCRYPT_EMAIL=admin@ragbench.co.za`
- [ ] Set `CORS_ORIGINS=https://ragbench.co.za`
- [ ] Set `LOG_LEVEL=INFO`
- [ ] Set `RAG_LLM_MODEL` (default: `gemma2:27b`)
- [ ] Added any API keys (OpenAI, etc.) if needed
- [ ] Verified `.env.prod` is gitignored

### Initial Build
- [ ] Built Docker images:
  ```bash
  docker compose -f docker-compose.prod.yml build
  ```
- [ ] Build completed without errors
- [ ] Images listed: `docker images | grep ragbench`

### SSL Certificate Setup
- [ ] Created certificate directories:
  ```bash
  mkdir -p certbot/conf certbot/www
  ```
- [ ] Obtained Let's Encrypt certificate:
  ```bash
  ./scripts/init-ssl.sh
  # OR
  make deploy-ssl
  ```
- [ ] Certificate files exist: `ls certbot/conf/live/ragbench.co.za/`
- [ ] Contains: `fullchain.pem`, `privkey.pem`

### Service Startup
- [ ] Started all services:
  ```bash
  docker compose -f docker-compose.prod.yml up -d
  ```
- [ ] All containers running:
  ```bash
  docker compose -f docker-compose.prod.yml ps
  ```
  Expected: `api`, `frontend`, `ollama`, `nginx`, `certbot` all `Up`

### Initial Verification
- [ ] Waited 2 minutes for services to initialize
- [ ] API health check passed:
  ```bash
  curl https://ragbench.co.za/api/health
  # Expected: {"status":"healthy"}
  ```
- [ ] Frontend accessible in browser: `https://ragbench.co.za`
- [ ] API docs accessible: `https://ragbench.co.za/docs`
- [ ] SSL certificate valid (green padlock in browser)

### Data Ingestion
- [ ] Uploaded initial PDFs to `/opt/ragbench/data/pdfs/`
- [ ] Ingested papers:
  ```bash
  docker compose -f docker-compose.prod.yml exec api \
    python -m rag_bench.cli.pipeline --ingest
  ```
- [ ] Ingestion completed successfully
- [ ] ChromaDB populated: `du -sh chroma_db/`

### Functional Test
- [ ] Opened frontend: `https://ragbench.co.za`
- [ ] Submitted test query: "What is a transformer?"
- [ ] Received response with citations
- [ ] Response time reasonable (< 30s)
- [ ] Citations link to papers correctly

---

## ☑️ CI/CD Setup (Day 1 Evening / Day 2)

### GitHub Secrets Configuration
Navigate to: GitHub repo → Settings → Secrets and variables → Actions → New repository secret

- [ ] `DEPLOY_HOST`: `ragbench.co.za` or server IP
- [ ] `DEPLOY_USER`: SSH username (e.g., `ragbench`)
- [ ] `DEPLOY_SSH_KEY`: Private SSH key (from `cat ~/.ssh/id_rsa`)
- [ ] `DEPLOY_PATH`: `/opt/ragbench`
- [ ] `ENV_PROD_FILE`: Contents of `.env.prod` (from `cat .env.prod`)

### Deploy Workflow Testing
- [ ] Workflow file exists: `.github/workflows/deploy.yml`
- [ ] Made a test commit to `main` branch
- [ ] GitHub Actions triggered automatically
- [ ] Workflow completed successfully
- [ ] Production deployment tag created

### Manual Deployment Test
- [ ] Can manually trigger workflow from GitHub Actions UI
- [ ] Manual deployment succeeds
- [ ] Health checks pass in workflow

---

## ☑️ Monitoring & Automation (Day 2-3)

### Health Monitoring
- [ ] Health monitor script exists: `scripts/health-monitor.sh`
- [ ] Script is executable: `chmod +x scripts/health-monitor.sh`

**Option A: Systemd Service (Recommended)**
- [ ] Copied systemd service:
  ```bash
  sudo cp systemd/ragbench-health-monitor.service /etc/systemd/system/
  ```
- [ ] Edited service if needed:
  ```bash
  sudo nano /etc/systemd/system/ragbench-health-monitor.service
  ```
- [ ] Enabled service:
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl enable ragbench-health-monitor
  sudo systemctl start ragbench-health-monitor
  ```
- [ ] Service running: `sudo systemctl status ragbench-health-monitor`

**Option B: Screen/Tmux**
- [ ] Started in screen: `screen -S health-monitor ./scripts/health-monitor.sh`

### Automated Backups
- [ ] Backup script exists: `scripts/backup.sh`
- [ ] Script is executable: `chmod +x scripts/backup.sh`
- [ ] Manual backup tested: `./scripts/backup.sh`
- [ ] Backup created: `ls -lh backups/`
- [ ] Added to crontab:
  ```bash
  crontab -e
  # Add: 0 2 * * * cd /opt/ragbench && ./scripts/backup.sh >> /opt/ragbench/logs/backup.log 2>&1
  ```
- [ ] Cron entry saved and verified: `crontab -l`

### Restore Testing
- [ ] Restore script exists: `scripts/restore.sh`
- [ ] Script is executable: `chmod +x scripts/restore.sh`
- [ ] Listed available backups: `./scripts/restore.sh`
- [ ] Tested restore (in staging or local): `./scripts/restore.sh <timestamp>`

### Rollback Testing
- [ ] Rollback script exists: `scripts/rollback.sh`
- [ ] Script is executable: `chmod +x scripts/rollback.sh`
- [ ] Current version tagged: `git tag prod-v1.0.0 && git push origin prod-v1.0.0`
- [ ] Tested rollback (in staging or local): `./scripts/rollback.sh prod-v1.0.0`

---

## ☑️ Documentation & Team Handoff

### Documentation Complete
- [ ] Read [DEPLOYMENT.md](DEPLOYMENT.md)
- [ ] Read [RUNBOOK.md](RUNBOOK.md)
- [ ] Read [QUICKSTART.md](QUICKSTART.md)

### Credentials Secured
- [ ] `.env.prod` backed up to secure password manager
- [ ] SSH keys secured
- [ ] GitHub secrets documented
- [ ] API keys (if any) documented

### Team Training
- [ ] Team members have SSH access
- [ ] Team trained on deployment process
- [ ] Team trained on rollback procedure
- [ ] Team knows where to find logs
- [ ] Emergency contact list created

### Monitoring Configured
- [ ] Set up external uptime monitoring (optional):
  - [ ] UptimeRobot, Pingdom, or similar
  - [ ] Check URL: `https://ragbench.co.za/api/health`
  - [ ] Alert email configured
- [ ] Slack/Discord webhook for alerts (optional)

---

## ☑️ Post-Deployment (Week 1)

### Verification & Monitoring
- [ ] **Day 1**: Check service health every 2 hours
- [ ] **Day 2-3**: Check service health twice daily
- [ ] **Day 4-7**: Check service health daily
- [ ] Monitor disk space: `df -h`
- [ ] Monitor memory usage: `free -h`
- [ ] Review logs: `docker compose -f docker-compose.prod.yml logs`

### Performance Baseline
- [ ] Measured average query response time: `______ seconds`
- [ ] Measured API throughput: `______ requests/minute`
- [ ] Documented ChromaDB size: `______ GB`
- [ ] Documented memory usage: `______ GB`

### User Feedback
- [ ] Collected user feedback on performance
- [ ] Identified any issues or bugs
- [ ] Created GitHub issues for improvements
- [ ] Prioritized next features

### Security Review
- [ ] Verified HTTPS everywhere (no mixed content)
- [ ] Checked SSL certificate auto-renewal works
- [ ] Reviewed API access logs for anomalies
- [ ] Confirmed firewall rules correct
- [ ] Updated dependencies if needed

---

## ☑️ Ongoing Operations

### Daily Tasks
- [ ] Check health monitor status
- [ ] Review application logs
- [ ] Verify backups completed

### Weekly Tasks
- [ ] Review system resource usage
- [ ] Check for Docker image updates
- [ ] Review access logs for errors
- [ ] Test backup restoration (monthly)

### Monthly Tasks
- [ ] Security updates: `sudo apt update && sudo apt upgrade`
- [ ] Review and rotate logs
- [ ] Review CloudFlare/CDN analytics
- [ ] Capacity planning review

---

## 📞 Emergency Contacts

| Role | Name | Contact |
|------|------|---------|
| **Primary Admin** | _____________ | _____________ |
| **Backup Admin** | _____________ | _____________ |
| **On-Call** | _____________ | _____________ |

## 🔗 Quick Links

- **Production**: https://ragbench.co.za
- **API Docs**: https://ragbench.co.za/docs
- **GitHub Repo**: https://github.com/nblomerus/rag-bench
- **GitHub Actions**: https://github.com/nblomerus/rag-bench/actions

---

## ✅ Deployment Complete!

**Deployment Date**: `_______________`  
**Deployed By**: `_______________`  
**Version**: `_______________`  
**Next Review Date**: `_______________`

**Congratulations! 🎉 Your RAG-Bench instance is now live in production!**

---

### Need Help?

- **Troubleshooting**: See [RUNBOOK.md](RUNBOOK.md)
- **Detailed Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)
