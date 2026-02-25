#!/bin/bash
# RAG-Bench Production Deployment Script
# Usage: ./deploy.sh

set -e
trap 'docker rm -f ragbench-nginx-bootstrap 2>/dev/null || true' EXIT

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  RAG-Bench Production Deployment                           ║"
echo "║  Domain: ragbench.co.za                                    ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="ragbench.co.za"
DOMAIN_EMAIL="admin@ragbench.co.za"
DEPLOYMENT_USER="ragbench"
DEPLOYMENT_PATH="/opt/ragbench"

# ════════════════════════════════════════════════════════════════
# 1. Check prerequisites
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[1/6]${NC} Checking prerequisites..."

prerequisites=("docker" "docker compose" "git" "curl")
missing=()

for cmd in "${prerequisites[@]}"; do
    if ! command -v $cmd &> /dev/null; then
        missing+=("$cmd")
    fi
done

if [ ${#missing[@]} -ne 0 ]; then
    echo -e "${RED}✗ Missing: ${missing[@]}${NC}"
    echo "Please install missing dependencies before proceeding."
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites installed${NC}"

# ════════════════════════════════════════════════════════════════
# 2. Create deployment directory and set permissions
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[2/6]${NC} Setting up deployment directory..."

if [ ! -d "$DEPLOYMENT_PATH" ]; then
    echo "Creating $DEPLOYMENT_PATH..."
    sudo mkdir -p "$DEPLOYMENT_PATH"
    sudo chown $(whoami):$(whoami) "$DEPLOYMENT_PATH"
fi

cd "$DEPLOYMENT_PATH"
echo -e "${GREEN}✓ Deployment directory ready${NC}"

# ════════════════════════════════════════════════════════════════
# 3. Clone or update repository
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[3/6]${NC} Setting up repository..."

if [ ! -d ".git" ]; then
    echo "Cloning repository..."
    git clone https://github.com/nblomerus/rag-bench.git .
else
    echo "Updating repository..."
    git fetch origin master
    git reset --hard origin/master
fi

echo -e "${GREEN}✓ Repository synced${NC}"

# ════════════════════════════════════════════════════════════════
# 4. Setup environment files
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[4/6]${NC} Setting up environment configuration..."

if [ ! -f ".env.prod" ]; then
    echo "Creating .env.prod..."
    cp .env.example .env.prod
    echo -e "${GREEN}✓ Created .env.prod from .env.example${NC}"
    echo -e "${YELLOW}  Edit .env.prod to override defaults if needed${NC}"
fi

echo -e "${GREEN}✓ Environment configured${NC}"

# ════════════════════════════════════════════════════════════════
# 5. Create directory structure for volumes
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[5/6]${NC} Setting up persistent volumes..."

mkdir -p chroma_db
mkdir -p data
mkdir -p logs
mkdir -p certbot/conf
mkdir -p certbot/www
mkdir -p nginx/conf.d

chmod 755 chroma_db data logs certbot

echo -e "${GREEN}✓ Volumes created${NC}"

# ════════════════════════════════════════════════════════════════
# 6. Start services
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[6/7]${NC} Tearing down any existing containers..."

docker compose -f docker-compose.prod.yml down --remove-orphans 2>/dev/null || true
# Remove any orphaned containers from a different compose context
for name in ragbench-api-prod ragbench-frontend-prod ragbench-nginx ragbench-ollama-prod ragbench-certbot; do
    if docker inspect "$name" &>/dev/null; then
        echo "Removing stale container: $name"
        docker stop "$name" 2>/dev/null && docker rm "$name" 2>/dev/null || true
    fi
done

echo -e "${GREEN}✓ Clean slate${NC}"

# ════════════════════════════════════════════════════════════════
# 7. Build and start
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[7/7]${NC} Building and starting services..."

# Load environment
export $(cat .env.prod | grep -v '^#' | xargs)

# Build images
echo "Building Docker images..."
docker compose -f docker-compose.prod.yml build

# Use renewal config as the cert-exists signal (world-readable, unlike live/ which is root 700)
CERT_RENEWAL="$DEPLOYMENT_PATH/certbot/conf/renewal/$DOMAIN.conf"

if [ ! -f "$CERT_RENEWAL" ]; then
    echo -e "\n${YELLOW}SSL certificates not found. Running initial cert bootstrap...${NC}"

    # Start all services except nginx (it requires certs to start)
    docker compose -f docker-compose.prod.yml up -d api frontend ollama certbot

    echo "Waiting for services to initialize..."
    sleep 15

    echo "Requesting Let's Encrypt certificate for $DOMAIN (standalone mode)..."

    # set -e ensures we exit if certbot fails
    docker run --rm \
        -p 80:80 \
        -v "$DEPLOYMENT_PATH/certbot/conf:/etc/letsencrypt" \
        certbot/certbot certonly \
        --standalone \
        --email "$DOMAIN_EMAIL" \
        --agree-tos \
        --no-eff-email \
        -d "$DOMAIN" \
        -d "www.$DOMAIN"

    echo -e "${GREEN}✓ Certificate issued${NC}"
fi

# Start (or restart) the full stack
echo "Starting full stack..."
docker compose -f docker-compose.prod.yml up -d

echo "Waiting for services to initialize..."
sleep 15

# Check status
echo "Checking service status..."
docker compose -f docker-compose.prod.yml ps

# ════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo -e "║  ${GREEN}✓ Deployment Complete${NC}                                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Your RAG-Bench instance is starting up..."
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Monitor logs: docker compose -f docker-compose.prod.yml logs -f"
echo "2. Check health: curl https://${DOMAIN}/health"
echo "3. Access UI: https://${DOMAIN}"
echo ""
echo "Configuration locations:"
echo "- Environment: $DEPLOYMENT_PATH/.env.prod"
echo "- Nginx config: $DEPLOYMENT_PATH/nginx/"
echo "- Data: $DEPLOYMENT_PATH/chroma_db/, $DEPLOYMENT_PATH/data/"
echo ""
echo "For SSL renewals, the certbot service handles them automatically."
echo "Check certbot logs: docker compose -f docker-compose.prod.yml logs certbot"
echo ""
