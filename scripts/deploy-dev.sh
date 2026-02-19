#!/bin/bash
# RAG-Bench Development Environment Setup Script
# Usage: ./deploy-dev.sh

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  RAG-Bench Development Environment Setup                   ║"
echo "║  Local deployment with hot-reload and debugging            ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ════════════════════════════════════════════════════════════════
# 1. Check prerequisites
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[1/5]${NC} Checking prerequisites..."

prerequisites=("docker" "git")
missing=()

for cmd in "${prerequisites[@]}"; do
    if ! command -v $cmd &> /dev/null; then
        missing+=("$cmd")
    fi
done

# Check for Docker Compose v2
if ! docker compose version &> /dev/null; then
    missing+=("docker compose v2")
fi

if [ ${#missing[@]} -ne 0 ]; then
    echo -e "${RED}✗ Missing: ${missing[@]}${NC}"
    echo "Please install missing dependencies before proceeding."
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites installed${NC}"
docker compose version

# ════════════════════════════════════════════════════════════════
# 2. Setup environment files
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[2/5]${NC} Setting up environment configuration..."

if [ ! -f ".env" ]; then
    echo "Creating .env for development..."
    cp .env.example .env
    
    # Set development-specific defaults
    sed -i 's/RAG_LLM_BACKEND=ollama/RAG_LLM_BACKEND=ollama/' .env
    sed -i 's/RAG_LLM_BASE_URL=http:\/\/ollama:11434/RAG_LLM_BASE_URL=http:\/\/ollama:11434/' .env
    sed -i 's/DOMAIN=ragbench.co.za/DOMAIN=localhost/' .env
    sed -i 's/LOG_LEVEL=INFO/LOG_LEVEL=DEBUG/' .env
    
    echo -e "${GREEN}✓ Created .env${NC}"
else
    echo -e "${BLUE}ℹ .env already exists${NC}"
fi

# Also create .env.dev for reference (optional)
if [ ! -f ".env.dev" ]; then
    cp .env .env.dev
fi

# ════════════════════════════════════════════════════════════════
# 3. Create directory structure for volumes
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[3/5]${NC} Setting up data directories..."

mkdir -p chroma_db
mkdir -p data/pdfs
mkdir -p logs

chmod 755 chroma_db data logs

# Create .gitkeep files to preserve directory structure
touch chroma_db/.gitkeep
touch data/.gitkeep
touch logs/.gitkeep

echo -e "${GREEN}✓ Directories created${NC}"

# ════════════════════════════════════════════════════════════════
# 4. Pull/Download Ollama model (optional)
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[4/5]${NC} Checking LLM setup..."

# Load environment
if [ -f ".env.dev" ]; then
    export $(cat .env.dev | grep -v '^#' | xargs)
fi

echo "LLM Backend: ${RAG_LLM_BACKEND:-ollama}"
echo "LLM Model: ${RAG_LLM_MODEL:-gemma2:27b}"

if [ "${RAG_LLM_BACKEND}" == "ollama" ]; then
    echo ""
    echo -e "${BLUE}ℹ Ollama will download the model on first use${NC}"
    echo "  Model: ${RAG_LLM_MODEL:-gemma2:27b}"
    echo "  This may take 5-10 minutes depending on your connection"
fi

echo -e "${GREEN}✓ LLM configuration ready${NC}"

# ════════════════════════════════════════════════════════════════
# 5. Build and start services
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}[5/5]${NC} Building and starting services..."

# Stop any existing containers
echo "Stopping existing containers..."
docker compose down 2>/dev/null || true

# Build images
echo "Building Docker images..."
docker compose build

# Start services
echo "Starting services..."
docker compose up -d

# Wait for services to start
echo "Waiting for services to initialize..."
sleep 5

# Check status
echo "Checking service status..."
docker compose ps

# ════════════════════════════════════════════════════════════════
# 6. Health checks
# ════════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}Running health checks...${NC}"

# Wait for API to be ready
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker compose exec -T api python -c "import httpx; httpx.get('http://localhost:8000/api/health', timeout=2)" 2>/dev/null; then
        echo -e "${GREEN}✓ API is healthy${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo -e "${BLUE}  Waiting for API... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
        sleep 2
    else
        echo -e "${YELLOW}⚠ API health check timeout (this is OK, it may still be starting)${NC}"
    fi
done

# ════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo -e "║  ${GREEN}✓ Development Environment Ready${NC}                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Services running:${NC}"
echo "  • Frontend: http://localhost"
echo "  • API:      http://localhost:8000"
echo "  • API Docs: http://localhost:8000/docs"
echo "  • Ollama:   http://localhost:11434"
echo ""
echo -e "${GREEN}Useful commands:${NC}"
echo "  • View logs:        make docker-logs"
echo "  • Stop services:    make docker-down"
echo "  • Restart:          make docker-restart"
echo "  • View status:      make docker-ps"
echo "  • Shell into API:   make docker-exec-api"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Open http://localhost in your browser"
echo "  2. Check API health: curl http://localhost:8000/api/health"
echo "  3. View logs: make docker-logs"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  • Environment: .env.dev"
echo "  • Compose:     docker-compose.yml + docker-compose.override.yml"
echo "  • Data:        ./data/, ./chroma_db/"
echo ""
