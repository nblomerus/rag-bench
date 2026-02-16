#!/bin/bash
# RAG-Bench Restore Script
# Usage: ./restore.sh <backup-timestamp>
# Example: ./restore.sh 20260215-143000

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/ragbench/backups}"
PROJECT_DIR="${PROJECT_DIR:-/opt/ragbench}"
TIMESTAMP="$1"

# Check arguments
if [ -z "$TIMESTAMP" ]; then
    error "No timestamp provided!"
    echo ""
    echo "Usage: $0 <backup-timestamp>"
    echo ""
    echo "Available backups:"
    ls -lh "$BACKUP_DIR"/*.tar.gz 2>/dev/null | awk '{print $9}' | sed 's/.*-/  /' | sed 's/.tar.gz//' | sort -u
    exit 1
fi

# Verify backup files exist
CHROMA_BACKUP="$BACKUP_DIR/chroma_db-$TIMESTAMP.tar.gz"
DATA_BACKUP="$BACKUP_DIR/data-$TIMESTAMP.tar.gz"
ENV_BACKUP="$BACKUP_DIR/.env.prod-$TIMESTAMP"

if [ ! -f "$CHROMA_BACKUP" ]; then
    error "ChromaDB backup not found: $CHROMA_BACKUP"
    exit 1
fi

log "═══════════════════════════════════════════════"
log "RAG-Bench Restore Utility"
log "═══════════════════════════════════════════════"
log "Restore timestamp: $TIMESTAMP"
log "Project directory: $PROJECT_DIR"
log "Backup directory: $BACKUP_DIR"
log "═══════════════════════════════════════════════"

# Show backup manifest if available
MANIFEST="$BACKUP_DIR/manifest-$TIMESTAMP.txt"
if [ -f "$MANIFEST" ]; then
    info "Backup manifest:"
    cat "$MANIFEST"
    echo ""
fi

# Confirmation prompt
warn "This will REPLACE your current data with the backup from $TIMESTAMP"
read -p "Continue? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log "Restore cancelled"
    exit 0
fi

cd "$PROJECT_DIR"

# 1. Stop services
log "Stopping services..."
docker compose -f docker-compose.prod.yml down
log "✅ Services stopped"

# 2. Backup current state (just in case)
SAFETY_BACKUP="$BACKUP_DIR/pre-restore-$(date +%Y%m%d-%H%M%S)"
log "Creating safety backup: $SAFETY_BACKUP"

if [ -d "chroma_db" ]; then
    tar -czf "$SAFETY_BACKUP-chroma_db.tar.gz" chroma_db/
    log "✅ Current ChromaDB backed up"
fi

if [ -d "data" ]; then
    tar -czf "$SAFETY_BACKUP-data.tar.gz" data/
    log "✅ Current data backed up"
fi

if [ -f ".env.prod" ]; then
    cp .env.prod "$SAFETY_BACKUP-.env.prod"
    log "✅ Current .env.prod backed up"
fi

# 3. Restore ChromaDB
log "Restoring ChromaDB..."
if [ -d "chroma_db" ]; then
    rm -rf chroma_db.restore-backup
    mv chroma_db chroma_db.restore-backup
fi

tar -xzf "$CHROMA_BACKUP"
log "✅ ChromaDB restored"

# 4. Restore data directory (if backup exists)
if [ -f "$DATA_BACKUP" ]; then
    log "Restoring data directory..."
    if [ -d "data" ]; then
        rm -rf data.restore-backup
        mv data data.restore-backup
    fi
    tar -xzf "$DATA_BACKUP"
    log "✅ Data directory restored"
else
    warn "Data backup not found, skipping"
fi

# 5. Restore environment file (if backup exists)
if [ -f "$ENV_BACKUP" ]; then
    log "Restoring environment configuration..."
    cp "$ENV_BACKUP" .env.prod
    log "✅ Environment configuration restored"
else
    warn "Environment backup not found, keeping current .env.prod"
fi

# 6. Verify restore
log "Verifying restore..."

if [ ! -d "chroma_db" ]; then
    error "ChromaDB directory missing after restore!"
    exit 1
fi

CHROMA_SIZE=$(du -sh chroma_db | cut -f1)
log "✅ ChromaDB size: $CHROMA_SIZE"

# 7. Restart services
log "Restarting services..."
docker compose -f docker-compose.prod.yml up -d

log "⏳ Waiting for services to start..."
sleep 15

# 8. Health check
log "Performing health check..."
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker compose -f docker-compose.prod.yml exec -T api python -c "import httpx; httpx.get('http://localhost:8000/api/health')" 2>/dev/null; then
        log "✅ Health check passed!"
        break
    fi
    warn "Waiting for application... ($((RETRY_COUNT + 1))/$MAX_RETRIES)"
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    error "Health check failed after $MAX_RETRIES attempts"
    error "Check logs with: docker compose -f docker-compose.prod.yml logs"
    exit 1
fi

# 9. Summary
log "═══════════════════════════════════════════════"
log "✅ Restore completed successfully!"
log "═══════════════════════════════════════════════"
log "Restored from: $TIMESTAMP"
log "Safety backup: $SAFETY_BACKUP-*"
log "Services status:"
docker compose -f docker-compose.prod.yml ps
log "═══════════════════════════════════════════════"

info "To remove safety backups (after verification):"
echo "  rm $SAFETY_BACKUP-*"

info "To rollback this restore:"
echo "  ./restore.sh <previous-timestamp>"
