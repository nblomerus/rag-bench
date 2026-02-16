#!/bin/bash
# RAG-Bench Health Monitoring Script
# Run as a systemd service or cron job for continuous monitoring

set -e

# Configuration
HEALTH_URL="${HEALTH_URL:-https://ragbench.co.za/api/health}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"  # 5 minutes
LOG_FILE="${LOG_FILE:-/opt/ragbench/logs/health-monitor.log}"
MAX_FAILURES="${MAX_FAILURES:-3}"

# Alert configuration (optional)
ALERT_EMAIL="${ALERT_EMAIL:-admin@ragbench.co.za}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# State
FAILURE_COUNT=0
LAST_STATUS="unknown"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

send_alert() {
    local message="$1"
    
    log "ALERT: $message"
    
    # Email alert (requires mailutils or sendmail)
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "RAG-Bench Health Alert" "$ALERT_EMAIL"
    fi
    
    # Slack webhook
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"🚨 RAG-Bench Alert: $message\"}" \
            2>/dev/null || true
    fi
}

check_health() {
    local http_code
    local response
    
    # Check HTTP response
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" --max-time 10)
    
    if [ "$http_code" = "200" ]; then
        # Verify response content
        response=$(curl -s "$HEALTH_URL" --max-time 10)
        
        if echo "$response" | grep -q '"status":"healthy"'; then
            if [ "$LAST_STATUS" != "healthy" ]; then
                log "✅ Service is HEALTHY (recovered from failure)"
                if [ "$LAST_STATUS" = "unhealthy" ]; then
                    send_alert "Service has recovered and is now healthy"
                fi
            fi
            FAILURE_COUNT=0
            LAST_STATUS="healthy"
            return 0
        fi
    fi
    
    # Health check failed
    FAILURE_COUNT=$((FAILURE_COUNT + 1))
    log "❌ Health check FAILED (attempt $FAILURE_COUNT/$MAX_FAILURES) - HTTP $http_code"
    
    if [ "$FAILURE_COUNT" -ge "$MAX_FAILURES" ] && [ "$LAST_STATUS" != "unhealthy" ]; then
        LAST_STATUS="unhealthy"
        send_alert "Service is UNHEALTHY after $MAX_FAILURES consecutive failures (HTTP $http_code)"
        
        # Log container status
        if command -v docker &> /dev/null; then
            log "Container status:"
            docker compose -f /opt/ragbench/docker-compose.prod.yml ps >> "$LOG_FILE" 2>&1
        fi
    fi
    
    return 1
}

# Main monitoring loop
log "Starting health monitoring for $HEALTH_URL (interval: ${CHECK_INTERVAL}s)"

while true; do
    check_health || true
    sleep "$CHECK_INTERVAL"
done
