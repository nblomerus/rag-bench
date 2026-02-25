#!/bin/bash
# Production Operations Helper Script
# Work with running production instance

set -e

DEPLOYMENT_PATH="${RAGBENCH_DEPLOY_PATH:-/opt/ragbench}"
COMPOSE_FILE="docker-compose.prod.yml"

# Always run compose against the production deployment directory so the
# project name and volume paths resolve correctly regardless of CWD.
dc() {
    docker compose --project-directory "$DEPLOYMENT_PATH" \
        -f "$DEPLOYMENT_PATH/$COMPOSE_FILE" "$@"
}

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
    echo "Evaluation:"
    echo "  eval                   Run full evaluation suite (saved as manual)"
    echo "  eval-production        Run full evaluation suite (saved as production)"
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
    if ! dc ps | grep -q "Up"; then
        echo "❌ Production is not running!"
        echo "Start with: make deploy"
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
    dc exec -T api \
        python -m rag_bench.cli.query "$question"
}

cmd_ingest() {
    check_running
    echo "📥 Ingesting papers into production ChromaDB..."
    dc exec api \
        python -m rag_bench.cli.pipeline --ingest
    echo "✅ Ingestion complete"
}

cmd_shell() {
    check_running
    echo "🐚 Opening shell in production API container..."
    echo "Type 'exit' to return"
    dc exec api bash
}

cmd_python() {
    check_running
    echo "🐍 Opening Python REPL in production..."
    echo "Try: from rag_bench.core.retriever import Retriever"
    dc exec api python
}

cmd_add_pdf() {
    local pdf_file="$1"
    if [ -z "$pdf_file" ] || [ ! -f "$pdf_file" ]; then
        echo "Usage: $0 add-pdf <pdf-file>"
        echo "File must exist!"
        exit 1
    fi
    
    check_running
    
    local container_id=$(dc ps -q api)
    local filename=$(basename "$pdf_file")
    
    echo "📄 Copying $filename to production..."
    docker exec "$container_id" mkdir -p /app/data/pdfs
    docker cp "$pdf_file" "$container_id:/app/data/pdfs/$filename"
    
    echo "📥 Ingesting..."
    dc exec -T api \
        python -m rag_bench.cli.pipeline --ingest
    
    echo "✅ $filename added and ingested"
}

cmd_logs() {
    check_running
    dc logs -f --tail=100
}

cmd_status() {
    echo "📊 Production Status"
    echo "===================="
    echo ""
    echo "Containers:"
    dc ps
    echo ""
    echo "Resource Usage:"
    docker stats --no-stream $(dc ps -q)
    echo ""
    echo "ChromaDB Size:"
    du -sh "$DEPLOYMENT_PATH/chroma_db" 2>/dev/null || echo "No ChromaDB data"
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
    dc exec api $command
}

cmd_start() {
    echo "🚀 Starting production..."
    # Remove any orphaned containers from a different compose context
    for name in ragbench-api-prod ragbench-frontend-prod ragbench-nginx ragbench-ollama-prod ragbench-certbot; do
        if docker inspect "$name" &>/dev/null; then
            echo "Removing stale container: $name"
            docker stop "$name" 2>/dev/null && docker rm "$name" 2>/dev/null || true
        fi
    done
    dc up -d
    sleep 5
    echo ""
    echo "✅ Production started"
    echo "   Frontend: http://localhost"
    echo "   API: http://localhost/api/health"
}

cmd_stop() {
    echo "⏹️  Stopping production..."
    dc down
    # Stop any containers that were started outside this compose project context
    # (e.g. via --no-deps or a different working directory)
    for name in ragbench-api-prod ragbench-frontend-prod ragbench-nginx ragbench-ollama-prod ragbench-certbot; do
        if docker inspect "$name" &>/dev/null; then
            echo "Stopping straggler: $name"
            docker stop "$name" 2>/dev/null && docker rm "$name" 2>/dev/null || true
        fi
    done
    echo "✅ Production stopped (resources freed for dev)"
}

cmd_restart() {
    echo "🔄 Restarting production..."
    dc restart
    sleep 5
    echo "✅ Production restarted"
}

cmd_rebuild() {
    echo "🔨 Rebuilding production (downtime: ~2 minutes)..."
    echo ""
    dc down
    # Remove any orphaned containers from a different compose context
    for name in ragbench-api-prod ragbench-frontend-prod ragbench-nginx ragbench-ollama-prod ragbench-certbot; do
        if docker inspect "$name" &>/dev/null; then
            echo "Removing stale container: $name"
            docker stop "$name" 2>/dev/null && docker rm "$name" 2>/dev/null || true
        fi
    done
    echo "Building images..."
    dc build
    echo "Starting services..."
    dc up -d
    sleep 10
    echo ""
    echo "✅ Production rebuilt and running"
    echo "   Frontend: http://localhost"
    echo "   Verifying health..."
    curl -s http://localhost/api/health && echo " ✓" || echo " ✗ (give it a moment)"
}

cmd_eval() {
    check_running
    echo "📊 Running full evaluation suite (this may take several minutes)..."
    dc exec -T api python -c "
import httpx, json

client = httpx.Client(base_url='http://localhost:8000', timeout=None)

print('=== RAG-Bench ===')
r = client.post('/api/eval/benchmark', json={'benchmark': 'ragbench', 'sample_size': 0})
print(json.dumps(r.json(), indent=2))

print()
print('=== RAGTruth ===')
r = client.post('/api/eval/benchmark', json={'benchmark': 'ragtruth', 'sample_size': 0})
print(json.dumps(r.json(), indent=2))
"
}

cmd_eval_production() {
    check_running
    echo "📊 Running production evaluation (this may take several minutes)..."
    dc exec -T api python -c "
from rag_bench.eval.benchmark import get_benchmark
from rag_bench.eval.judge import JudgeLLM
from rag_bench.eval.report import save_report, generate_terminal_summary
from rag_bench.eval.runner import EvalRunner
from rag_bench.core.retriever import Retriever
from rag_bench.core.generator import Generator

retriever = Retriever()
generator = Generator()
judge = JudgeLLM(generator.llm)
benchmark = get_benchmark()

runner = EvalRunner(
    retriever=retriever,
    generator=generator,
    judge=judge,
    benchmark=benchmark,
)

report = runner.run_all()
save_report(report, 'eval_results', run_type='production')
print(generate_terminal_summary(report))
"
}

cmd_stats() {
    check_running
    echo "📊 Live Resource Usage (Ctrl+C to exit)"
    docker stats $(dc ps -q)
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
    eval)
        cmd_eval
        ;;
    eval-production)
        cmd_eval_production
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
