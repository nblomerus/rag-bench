#!/bin/bash
# RAG-Bench Rollback Script
# Usage: ./rollback.sh [tag]
# Example: ./rollback.sh prod-v1.0.0

set -e

# Colors
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
PROJECT_DIR="${PROJECT_DIR:-/opt/ragbench}"
TAG="$1"

cd "$PROJECT_DIR"

# If no tag provided, show available tags
if [ -z "$TAG" ]; then
    log "RAG-Bench Rollback Utility"
    echo ""
    echo "Available deployment tags:"
    git fetch --all --tags 2>/dev/null || true
    git tag -l "prod-v*" | sort -V | tail -n 10 | while read tag; do
        TAG_DATE=$(git log -1 --format=%ai "$tag" 2>/dev/null || echo "unknown")
        TAG_MSG=$(git tag -l --format='%(contents:subject)' "$tag" 2>/dev/null || echo "")
        echo "  $tag ($TAG_DATE) - $TAG_MSG"
    done
    echo ""
    echo "Usage: $0 <tag>"
    echo "Example: $0 prod-v1.0.0"
    exit 0
fi

# Verify tag exists
if ! git rev-parse "$TAG" >/dev/null 2>&1; then
    error "Tag '$TAG' not found!"
    echo ""
    echo "Available tags:"
    git tag -l "prod-v*" | sort -V | tail -n 10
    exit 1
fi

# Get current state
CURRENT_COMMIT=$(git rev-parse HEAD)
CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "no tag")

log "═══════════════════════════════════════════════"
log "RAG-Bench Rollback"
log "═══════════════════════════════════════════════"
log "Current state:"
log "  Commit: $CURRENT_COMMIT"
log "  Tag: $CURRENT_TAG"
log "  Version: $(cat version 2>/dev/null || echo 'unknown')"
echo ""
log "Rolling back to:"
log "  Tag: $TAG"
TARGET_VERSION=$(git show "$TAG:version" 2>/dev/null || echo "unknown")
log "  Version: $TARGET_VERSION"
log "═══════════════════════════════════════════════"

# Confirmation
warn "This will rollback your deployment to $TAG"
warn "Current containers will be stopped and rebuilt"
echo ""
read -p "Continue? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log "Rollback cancelled"
    exit 0
fi

# 1. Create emergency backup
log "Creating emergency backup..."
EMERGENCY_BACKUP="/opt/ragbench/backups/pre-rollback-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$EMERGENCY_BACKUP")"

if [ -d "chroma_db" ]; then
    tar -czf "$EMERGENCY_BACKUP-chroma_db.tar.gz" chroma_db/
    log "✅ Emergency backup created: $EMERGENCY_BACKUP-chroma_db.tar.gz"
fi

# 2. Stop services
log "Stopping services..."
docker compose -f docker-compose.prod.yml down
log "✅ Services stopped"

# 3. Checkout target tag
log "Checking out $TAG..."
git fetch --all --tags
git checkout "$TAG"
log "✅ Checked out $TAG"

# 4. Rebuild containers
log "Rebuilding Docker images..."
docker compose -f docker-compose.prod.yml build --no-cache
log "✅ Images rebuilt"

# 5. Start services
log "Starting services..."
docker compose -f docker-compose.prod.yml up -d
log "✅ Services started"

# 6. Wait for startup
log "⏳ Waiting for services to start..."
sleep 15

# 7. Health check
log "Performing health check..."
MAX_RETRIES=10
RETRY_COUNT=0
HEALTH_URL="${HEALTH_URL:-https://ragbench.co.za/api/health}"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f -s "$HEALTH_URL" > /dev/null 2>&1; then
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
    echo ""
    warn "To restore previous state:"
    echo "  git checkout $CURRENT_COMMIT"
    echo "  docker compose -f docker-compose.prod.yml down"
    echo "  docker compose -f docker-compose.prod.yml build"
    echo "  docker compose -f docker-compose.prod.yml up -d"
    exit 1
fi

# 8. Verify frontend
if curl -f -s -I "$HEALTH_URL" 2>&1 | grep -q "200"; then
    log "✅ Frontend is accessible"
else
    warn "Frontend check inconclusive"
fi

# 9. Show current status
log "═══════════════════════════════════════════════"
log "✅ Rollback completed successfully!"
log "═══════════════════════════════════════════════"
log "Rolled back to:"
log "  Tag: $TAG"
log "  Version: $(cat version)"
log "  Commit: $(git rev-parse HEAD)"
echo ""
log "Services status:"
docker compose -f docker-compose.prod.yml ps
log "═══════════════════════════════════════════════"

info "Emergency backup saved:"
echo "  $EMERGENCY_BACKUP-chroma_db.tar.gz"
echo ""
info "To rollback this rollback (if needed):"
echo "  ./rollback.sh $CURRENT_TAG"
echo "  # or"
echo "  git checkout $CURRENT_COMMIT"
