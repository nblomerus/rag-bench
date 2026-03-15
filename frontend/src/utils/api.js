export const API_BASE = '/api'

// Stream tokens from SSE endpoint, calling callbacks as events arrive
export async function queryRAGStream(question, { onSources, onToken, onDone, onError, onPipeline, onQueue, topK = 5 }) {
    const res = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: topK }),
    })
    if (!res.ok) throw new Error(`API error: ${res.status}`)

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line in buffer

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
                const evt = JSON.parse(line.slice(6))
                if (evt.event === 'queue' && onQueue) onQueue(evt)
                else if (evt.event === 'pipeline' && onPipeline) onPipeline(evt)
                else if (evt.event === 'sources' && onSources) onSources(evt.sources || [])
                else if (evt.event === 'token' && onToken) onToken(evt.token)
                else if (evt.event === 'done' && onDone) onDone(evt)
                else if (evt.event === 'error' && onError) onError(evt.message)
            } catch (e) { /* skip malformed */ }
        }
    }
}

// Non-streaming fallback (used by eval)
export async function queryRAG(question, topK = 5) {
    const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: topK }),
    })
    if (!res.ok) throw new Error(`API error: ${res.status}`)
    return res.json()
}

export async function fetchQueueStatus() {
    const res = await fetch(`${API_BASE}/queue/status`)
    if (!res.ok) return { active: 0, queued: 0, capacity: 2 }
    return res.json()
}

export async function fetchStats() {
    const res = await fetch(`${API_BASE}/stats`)
    if (!res.ok) throw new Error(`Stats error: ${res.status}`)
    return res.json()
}

export async function runEvalAPI(runAll = false) {
    const res = await fetch(`${API_BASE}/eval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_all: runAll }),
    })
    if (!res.ok) throw new Error(`Eval error: ${res.status}`)
    return res.json()
}

export async function fetchPaper(paperId) {
    const res = await fetch(`${API_BASE}/papers/${encodeURIComponent(paperId)}`)
    if (!res.ok) throw new Error(`Paper fetch error: ${res.status}`)
    return res.json()
}

export async function runBenchmarkAPI(benchmark, sampleSize = 50) {
    const res = await fetch(`${API_BASE}/eval/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ benchmark, sample_size: sampleSize }),
    })
    if (!res.ok) throw new Error(`Benchmark error: ${res.status}`)
    return res.json()
}

export async function fetchBenchmarkHistory() {
    const res = await fetch(`${API_BASE}/eval/benchmark/history`)
    if (!res.ok) throw new Error(`History error: ${res.status}`)
    return res.json()
}

export async function fetchBenchmarkLatest(benchmark) {
    const res = await fetch(`${API_BASE}/eval/benchmark/latest/${benchmark}`)
    if (!res.ok) throw new Error(`No results available`)
    return res.json()
}

export async function fetchBenchmarkExamples() {
    const res = await fetch(`${API_BASE}/eval/benchmark/examples`)
    if (!res.ok) throw new Error(`Examples error: ${res.status}`)
    return res.json()
}

export async function fetchRagtruthExamples() {
    const res = await fetch(`${API_BASE}/eval/ragtruth/examples`)
    if (!res.ok) throw new Error(`RAGTruth examples error: ${res.status}`)
    return res.json()
}

export async function detectHallucination(context, response) {
    const res = await fetch(`${API_BASE}/eval/ragtruth/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context, response }),
    })
    if (!res.ok) throw new Error(`Detection error: ${res.status}`)
    return res.json()
}

export async function fetchMetricsSummary() {
    const res = await fetch(`${API_BASE}/metrics/summary`)
    if (!res.ok) throw new Error(`Metrics error: ${res.status}`)
    return res.json()
}

export async function fetchBenchmarkTrends(runType = 'production') {
    const res = await fetch(`${API_BASE}/eval/benchmark/trends?run_type=${runType}`)
    if (!res.ok) throw new Error(`Trends error: ${res.status}`)
    return res.json()
}

export async function fetchEvalSchedule() {
    const res = await fetch(`${API_BASE}/eval/schedule`)
    if (!res.ok) throw new Error(`Schedule error: ${res.status}`)
    return res.json()
}

export async function fetchGraphContext(question) {
    const res = await fetch(`${API_BASE}/graph/context?question=${encodeURIComponent(question)}`)
    if (!res.ok) return { nodes: [], edges: [], matched_entities: [] }
    return res.json()
}

export async function updateEvalSchedule(enabled, intervalHours = 24) {
    const res = await fetch(`${API_BASE}/eval/schedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled, interval_hours: intervalHours }),
    })
    if (!res.ok) throw new Error(`Schedule update error: ${res.status}`)
    return res.json()
}
