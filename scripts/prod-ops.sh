#!/bin/bash
# Production Operations Helper Script
# Work with running production instance

set -e

COMPOSE_FILE="docker-compose.prod.yml"

show_help() {
    echo "RAG-Bench Production Operations"
    echo ""
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Lifecycle (downtime OK):"
    echo "  start                  Start production"
    echo "  stop                   Stop production (frees resources)"
    echo "  restart                Restart production"
    echo "  rebuild                Stop, rebuild, start production"
    echo ""
    echo "Data Operations (while running):"
    echo "  query <question>       Run a query against production"
    echo "  ingest                 Ingest new papers from data/pdfs/"
    echo "  add-pdf <file>         Copy PDF to production and ingest"
    echo ""
    echo "Development & Debugging:"
    echo "  shell                  Open shell in API container"
    echo "  python                 Open Python REPL in production"
    echo "  logs                   Show production logs"
    echo "  status                 Show production status & resources"
    echo "  stats                  Live resource usage (Ctrl+C to exit)"
    echo "  exec <command>         Execute command in API container"
    echo ""
    echo "Examples:"
    echo "  # Start/stop workflow"
    echo "  $0 start               # Start production"
    echo "  $0 stop                # Stop to work in dev"
    echo "  $0 rebuild             # Deploy code changes"
    echo ""
    echo "  # Work with running production"
    echo "  $0 query 'What is RAG?'"
    echo "  $0 ingest"
    echo "  $0 add-pdf paper.pdf"
    echo "  $0 status"
}

check_running() {
    if ! docker compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        echo "❌ Production is not running!"
        echo "Start with: docker compose -f $COMPOSE_FILE up -d"
        exit 1
    fi
}

cmd_query() {
    local question="$1"
    if [ -z "$question" ]; then
        echo "Usage: $0 query <question>"
        exit 1
    fi
    
    check_running
    echo "🔍 Querying production..."
    docker compose -f "$COMPOSE_FILE" exec -T api \
        python -m rag_bench.cli.query "$question"
}

cmd_ingest() {
    check_running
    echo "📥 Ingesting papers into production ChromaDB..."
    docker compose -f "$COMPOSE_FILE" exec api \
        python -m rag_bench.cli.pipeline --ingest
    echo "✅ Ingestion complete"
}

cmd_shell() {
    check_running
    echo "🐚 Opening shell in production API container..."
    echo "Type 'exit' to return"
    docker compose -f "$COMPOSE_FILE" exec api bash
}

cmd_python() {
    check_running
    echo "🐍 Opening Python REPL in production..."
    echo "Try: from rag_bench.core.retriever import Retriever"
    docker compose -f "$COMPOSE_FILE" exec api python
}

cmd_add_pdf() {
    local pdf_file="$1"
    if [ -z "$pdf_file" ] || [ ! -f "$pdf_file" ]; then
        echo "Usage: $0 add-pdf <pdf-file>"
        echo "File must exist!"
        exit 1
    fi
    
    check_running
    
    local container_id=$(docker compose -f "$COMPOSE_FILE" ps -q api)
    local filename=$(basename "$pdf_file")
    
    echo "📄 Copying $filename to production..."
    docker exec "$container_id" mkdir -p /app/data/pdfs
    docker cp "$pdf_file" "$container_id:/app/data/pdfs/$filename"
    
    echo "📥 Ingesting..."
    docker compose -f "$COMPOSE_FILE" exec -T api \
        python -m rag_bench.cli.pipeline --ingest
    
    echo "✅ $filename added and ingested"
}

cmd_logs() {
    check_running
    docker compose -f "$COMPOSE_FILE" logs -f --tail=100
}

cmd_status() {
    echo "📊 Production Status"
    echo "===================="
    echo ""
    echo "Containers:"
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    echo "Resource Usage:"
    docker stats --no-stream $(docker compose -f "$COMPOSE_FILE" ps -q)
    echo ""
    echo "ChromaDB Size:"
    du -sh chroma_db 2>/dev/null || echo "No ChromaDB data"
    echo ""
    echo "Health Check:"
    curl -s http://localhost/api/health || echo "API not responding"
}

cmd_exec() {
    local command="$@"
    if [ -z "$command" ]; then
        echo "Usage: $0 exec <command>"
        exit 1
    fi
    
    check_running
    docker compose -f "$COMPOSE_FILE" exec api $command
}

cmd_start() {
    echo "🚀 Starting production..."
    docker compose -f "$COMPOSE_FILE" up -d
    sleep 5
    echo ""
    echo "✅ Production started"
    echo "   Frontend: http://localhost"
    echo "   API: http://localhost/api/health"
}

cmd_stop() {
    echo "⏹️  Stopping production..."
    docker compose -f "$COMPOSE_FILE" down
    echo "✅ Production stopped (resources freed for dev)"
}

cmd_restart() {
    echo "🔄 Restarting production..."
    docker compose -f "$COMPOSE_FILE" restart
    sleep 5
    echo "✅ Production restarted"
}

cmd_rebuild() {
    echo "🔨 Rebuilding production (downtime: ~2 minutes)..."
    echo ""
    docker compose -f "$COMPOSE_FILE" down
    echo "Building images..."
    docker compose -f "$COMPOSE_FILE" build
    echo "Starting services..."
    docker compose -f "$COMPOSE_FILE" up -d
    sleep 10
    echo ""
    echo "✅ Production rebuilt and running"
    echo "   Frontend: http://localhost"
    echo "   Verifying health..."
    curl -s http://localhost/api/health && echo " ✓" || echo " ✗ (give it a moment)"
}

cmd_stats() {
    check_running
    echo "📊 Live Resource Usage (Ctrl+C to exit)"
    docker stats $(docker compose -f "$COMPOSE_FILE" ps -q)
}

# Main command router
case "${1:-help}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    rebuild)
        cmd_rebuild
        ;;
    stats)
        cmd_stats
        ;;
    query)
        shift
        cmd_query "$@"
        ;;
    ingest)
        cmd_ingest
        ;;
    shell|bash)
        cmd_shell
        ;;
    python|repl)
        cmd_python
        ;;
    add-pdf)
        shift
        cmd_add_pdf "$@"
        ;;
    logs)
        cmd_logs
        ;;
    status)
        cmd_status
        ;;
    exec)
        shift
        cmd_exec "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
