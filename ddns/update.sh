#!/bin/sh
# Cloudflare DDNS updater — checks public IP and updates A record if changed.
#
# Required env vars:
#   CF_API_TOKEN   — Cloudflare API token (Zone:DNS:Edit permission)
#   CF_ZONE_ID     — Zone ID (found on the domain's Overview page in Cloudflare)
#   CF_RECORD_NAME — DNS record to update (e.g. "ragbench.co.za" or "@")
#
# Optional:
#   CF_TTL         — TTL in seconds (default: 300)
#   CF_PROXIED     — "true" or "false" (default: "false")
#   UPDATE_INTERVAL— Seconds between checks (default: 300 = 5 min)

set -e

: "${CF_API_TOKEN:?CF_API_TOKEN is required}"
: "${CF_ZONE_ID:?CF_ZONE_ID is required}"
: "${CF_RECORD_NAME:?CF_RECORD_NAME is required}"
: "${CF_TTL:=300}"
: "${CF_PROXIED:=false}"
: "${UPDATE_INTERVAL:=300}"

CF_API="https://api.cloudflare.com/client/v4"
CACHED_IP=""

log() {
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"
}

get_public_ip() {
    # Try multiple sources for reliability
    ip=$(curl -sf --max-time 10 https://api.ipify.org) ||
    ip=$(curl -sf --max-time 10 https://ifconfig.me) ||
    ip=$(curl -sf --max-time 10 https://icanhazip.com) ||
    { log "ERROR: Failed to determine public IP"; return 1; }
    echo "$ip"
}

get_record_id() {
    response=$(curl -sf --max-time 10 \
        -H "Authorization: Bearer ${CF_API_TOKEN}" \
        -H "Content-Type: application/json" \
        "${CF_API}/zones/${CF_ZONE_ID}/dns_records?type=A&name=${CF_RECORD_NAME}")

    echo "$response" | jq -r '.result[0].id // empty'
}

get_record_ip() {
    response=$(curl -sf --max-time 10 \
        -H "Authorization: Bearer ${CF_API_TOKEN}" \
        -H "Content-Type: application/json" \
        "${CF_API}/zones/${CF_ZONE_ID}/dns_records?type=A&name=${CF_RECORD_NAME}")

    echo "$response" | jq -r '.result[0].content // empty'
}

update_record() {
    record_id=$1
    new_ip=$2

    response=$(curl -sf --max-time 10 -X PUT \
        -H "Authorization: Bearer ${CF_API_TOKEN}" \
        -H "Content-Type: application/json" \
        --data "{\"type\":\"A\",\"name\":\"${CF_RECORD_NAME}\",\"content\":\"${new_ip}\",\"ttl\":${CF_TTL},\"proxied\":${CF_PROXIED}}" \
        "${CF_API}/zones/${CF_ZONE_ID}/dns_records/${record_id}")

    success=$(echo "$response" | jq -r '.success')
    if [ "$success" = "true" ]; then
        return 0
    else
        log "ERROR: Cloudflare API response: $response"
        return 1
    fi
}

create_record() {
    new_ip=$1

    response=$(curl -sf --max-time 10 -X POST \
        -H "Authorization: Bearer ${CF_API_TOKEN}" \
        -H "Content-Type: application/json" \
        --data "{\"type\":\"A\",\"name\":\"${CF_RECORD_NAME}\",\"content\":\"${new_ip}\",\"ttl\":${CF_TTL},\"proxied\":${CF_PROXIED}}" \
        "${CF_API}/zones/${CF_ZONE_ID}/dns_records")

    success=$(echo "$response" | jq -r '.success')
    if [ "$success" = "true" ]; then
        return 0
    else
        log "ERROR: Cloudflare API response: $response"
        return 1
    fi
}

# ── Main loop ──

log "Starting Cloudflare DDNS updater for ${CF_RECORD_NAME}"
log "Check interval: ${UPDATE_INTERVAL}s, TTL: ${CF_TTL}, Proxied: ${CF_PROXIED}"

while true; do
    current_ip=$(get_public_ip) || { sleep "$UPDATE_INTERVAL"; continue; }

    if [ "$current_ip" = "$CACHED_IP" ]; then
        sleep "$UPDATE_INTERVAL"
        continue
    fi

    log "IP change detected: ${CACHED_IP:-<none>} -> ${current_ip}"

    record_id=$(get_record_id)

    if [ -n "$record_id" ]; then
        dns_ip=$(get_record_ip)
        if [ "$current_ip" = "$dns_ip" ]; then
            log "DNS already points to ${current_ip}, no update needed"
            CACHED_IP="$current_ip"
            sleep "$UPDATE_INTERVAL"
            continue
        fi

        if update_record "$record_id" "$current_ip"; then
            log "Updated A record to ${current_ip}"
            CACHED_IP="$current_ip"
        else
            log "Failed to update record, will retry"
        fi
    else
        log "No existing A record found, creating one"
        if create_record "$current_ip"; then
            log "Created A record pointing to ${current_ip}"
            CACHED_IP="$current_ip"
        else
            log "Failed to create record, will retry"
        fi
    fi

    sleep "$UPDATE_INTERVAL"
done
