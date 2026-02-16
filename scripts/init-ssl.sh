#!/bin/bash
# Initialize Let's Encrypt SSL certificates for RAG-Bench
# Run this BEFORE starting the production docker-compose

set -e

DOMAIN="ragbench.co.za"
EMAIL="admin@ragbench.co.za"
CERTBOT_DIR="./certbot"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  RAG-Bench SSL Certificate Initialization                  ║"
echo "║  Domain: $DOMAIN                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Create required directories
mkdir -p "$CERTBOT_DIR/conf"
mkdir -p "$CERTBOT_DIR/www"

echo "Creating dummy SSL certificate for initial startup..."

# Create a dummy certificate to bootstrap Nginx
docker run --rm \
    -v "$CERTBOT_DIR/conf:/etc/letsencrypt" \
    -v "$CERTBOT_DIR/www:/var/www/certbot" \
    certbot/certbot:latest \
    certonly \
    --webroot \
    -w /var/www/certbot \
    -d "$DOMAIN" \
    -d "www.$DOMAIN" \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    --non-interactive || true

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  SSL initialization complete!                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "You can now start the production deployment with:"
echo "  docker-compose -f docker-compose.prod.yml up -d"
echo ""
