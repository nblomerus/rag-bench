#!/bin/bash
# Toggle between development and production environments locally

set -e

CURRENT_ENV_FILE=".current_env"

show_status() {
    if [ -f "$CURRENT_ENV_FILE" ]; then
        CURRENT=$(cat "$CURRENT_ENV_FILE")
        echo "Current environment: $CURRENT"
    else
        echo "Current environment: development (default)"
    fi
}

to_dev() {
    echo "Switching to development environment..."
    
    # Stop production if running
    if docker compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        echo "Stopping production containers..."
        docker compose -f docker-compose.prod.yml down
    fi
    
    # Start development
    echo "Starting development environment..."
    docker compose up -d
    
    echo "dev" > "$CURRENT_ENV_FILE"
    echo ""
    echo "✅ Development environment active"
    echo "   Frontend: http://localhost:3000"
    echo "   API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
}

to_prod() {
    echo "Switching to production environment (local test)..."
    
    # Check for .env.prod.local
    if [ ! -f ".env.prod.local" ]; then
        echo "Creating .env.prod.local from template..."
        cp .env.production.template .env.prod.local
        echo ""
        echo "⚠️  Please edit .env.prod.local:"
        echo "   - Set DOMAIN=localhost"
        echo "   - Set LOG_LEVEL=DEBUG (for testing)"
        read -p "Press enter when ready..."
    fi
    
    # Stop development if running
    if docker compose ps | grep -q "Up"; then
        echo "Stopping development containers..."
        docker compose down
    fi
    
    # Start production
    echo "Starting production environment..."
    docker compose -f docker-compose.prod.yml --env-file .env.prod.local up -d
    
    echo "prod" > "$CURRENT_ENV_FILE"
    echo ""
    echo "✅ Production environment active (local test)"
    echo "   Frontend: http://localhost"
    echo "   API: http://localhost/api"
    echo "   Note: SSL/Certbot disabled for local testing"
}

# Main logic
case "${1:-status}" in
    dev|development)
        to_dev
        ;;
    prod|production)
        to_prod
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {dev|prod|status}"
        echo ""
        echo "Commands:"
        echo "  dev     - Switch to development environment"
        echo "  prod    - Switch to production environment (local test)"
        echo "  status  - Show current environment"
        exit 1
        ;;
esac
