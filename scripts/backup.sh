#!/bin/bash
# RAG-Bench Backup Script
# Schedule with cron: 0 2 * * * /opt/ragbench/scripts/backup.sh

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/ragbench/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
DATE=$(date +%Y%m%d-%H%M%S)
PROJECT_DIR="${PROJECT_DIR:-/opt/ragbench}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

log "Starting backup at $(date)"
log "Backup directory: $BACKUP_DIR"

cd "$PROJECT_DIR"

# 1. Backup ChromaDB
log "Backing up ChromaDB..."
if [ -d "chroma_db" ]; then
    tar -czf "$BACKUP_DIR/chroma_db-$DATE.tar.gz" chroma_db/
    CHROMA_SIZE=$(du -sh "$BACKUP_DIR/chroma_db-$DATE.tar.gz" | cut -f1)
    log "✅ ChromaDB backup complete ($CHROMA_SIZE)"
else
    warn "chroma_db directory not found, skipping"
fi

# 2. Backup data directory
log "Backing up data directory..."
if [ -d "data" ]; then
    tar -czf "$BACKUP_DIR/data-$DATE.tar.gz" data/
    DATA_SIZE=$(du -sh "$BACKUP_DIR/data-$DATE.tar.gz" | cut -f1)
    log "✅ Data backup complete ($DATA_SIZE)"
else
    warn "data directory not found, skipping"
fi

# 3. Backup environment configuration (if exists)
log "Backing up configuration..."
if [ -f ".env.prod" ]; then
    cp .env.prod "$BACKUP_DIR/.env.prod-$DATE"
    log "✅ Configuration backup complete"
else
    warn ".env.prod not found, skipping"
fi

# 4. Export Docker Compose configuration
log "Exporting Docker configuration..."
if [ -f "docker-compose.prod.yml" ]; then
    docker compose -f docker-compose.prod.yml config \
        > "$BACKUP_DIR/docker-config-$DATE.yml" 2>/dev/null || warn "Docker config export failed"
    log "✅ Docker configuration exported"
fi

# 5. Backup git commit hash for reference
log "Recording git version..."
if [ -d ".git" ]; then
    echo "$(git rev-parse HEAD)" > "$BACKUP_DIR/git-commit-$DATE.txt"
    echo "$(git describe --tags --always)" >> "$BACKUP_DIR/git-commit-$DATE.txt"
    log "✅ Git version recorded"
fi

# 6. Create backup manifest
log "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest-$DATE.txt" << EOF
RAG-Bench Backup Manifest
========================
Date: $(date)
Version: $(cat version 2>/dev/null || echo "unknown")
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "unknown")

Backup Contents:
$(ls -lh "$BACKUP_DIR"/*-$DATE.* 2>/dev/null || echo "No files found")

System Info:
Hostname: $(hostname)
Disk Usage: $(df -h "$PROJECT_DIR" | tail -n 1)
EOF

log "✅ Manifest created"

# 7. Cleanup old backups
log "Cleaning up backups older than $RETENTION_DAYS days..."
BEFORE_COUNT=$(find "$BACKUP_DIR" -name "*.tar.gz" | wc -l)

find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name ".env.prod-*" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "docker-config-*" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "git-commit-*" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "manifest-*" -mtime +$RETENTION_DAYS -delete

AFTER_COUNT=$(find "$BACKUP_DIR" -name "*.tar.gz" | wc -l)
DELETED=$((BEFORE_COUNT - AFTER_COUNT))

if [ $DELETED -gt 0 ]; then
    log "✅ Removed $DELETED old backup(s)"
fi

# 8. Summary
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
BACKUP_COUNT=$(find "$BACKUP_DIR" -name "*.tar.gz" | wc -l)

log "════════════════════════════════════════"
log "Backup Summary:"
log "  Location: $BACKUP_DIR"
log "  Total backups: $BACKUP_COUNT"
log "  Total size: $TOTAL_SIZE"
log "  Latest backup: $DATE"
log "════════════════════════════════════════"
log "✅ Backup completed successfully!"

# Optional: Send notification
if [ -n "$BACKUP_WEBHOOK" ]; then
    curl -X POST "$BACKUP_WEBHOOK" \
        -H 'Content-Type: application/json' \
        -d "{\"text\":\"✅ RAG-Bench backup completed: $BACKUP_COUNT backups, $TOTAL_SIZE total\"}" \
        2>/dev/null || true
fi
