import React, { useState, useEffect, useRef } from 'react'
import { BarChartIcon, Spinner, AlertIcon, SearchIcon, ChevronDown, ChevronUp, ZapIcon } from '../components/Icons'
import { fetchBenchmarkLatest, fetchBenchmarkExamples, queryRAG, detectHallucination, fetchBenchmarkTrends, fetchEvalSchedule, updateEvalSchedule } from '../utils/api'
import { formatAnswer } from '../utils/render'

// ── Metric definitions with explanations ──────────────────────────────────────
const METRIC_DEFS = {
    // RAG-Bench retrieval
    retrieval_mrr: {
        label: 'MRR',
        group: 'Retrieval',
        description: 'Mean Reciprocal Rank — how quickly the first relevant paper appears in results. 1.0 means the correct paper is always ranked first.',
        good: 0.7, mid: 0.4,
    },
    retrieval_ndcg_at_5: {
        label: 'NDCG@5',
        group: 'Retrieval',
        description: 'Normalized Discounted Cumulative Gain — measures how well the top 5 results are ranked. Rewards placing relevant papers higher.',
        good: 0.7, mid: 0.4,
    },
    retrieval_hit_rate: {
        label: 'Hit Rate',
        group: 'Retrieval',
        description: 'Percentage of queries where at least one relevant paper appeared in the top 5 results.',
        good: 0.8, mid: 0.5,
    },
    retrieval_recall_at_5: {
        label: 'Recall@5',
        group: 'Retrieval',
        description: 'Of all relevant papers for a query, what fraction were retrieved in the top 5.',
        good: 0.7, mid: 0.4,
    },
    retrieval_precision_at_5: {
        label: 'Precision@5',
        group: 'Retrieval',
        description: 'Of the 5 retrieved papers, what fraction were actually relevant. Low values are normal since most queries need only 1-2 papers.',
        good: 0.4, mid: 0.2,
    },
    // RAG-Bench generation
    avg_citation_precision: {
        label: 'Citation Precision',
        group: 'Generation',
        description: 'Of the sources the LLM cited in its answer, what fraction were actually the correct source papers.',
        good: 0.6, mid: 0.3,
    },
    avg_citation_recall: {
        label: 'Citation Recall',
        group: 'Generation',
        description: 'Of the correct source papers, what fraction did the LLM actually cite in its answer.',
        good: 0.6, mid: 0.3,
    },
    avg_completeness: {
        label: 'Completeness',
        group: 'Generation',
        description: 'Fraction of expected key concepts and terms that appeared in the generated answer.',
        good: 0.8, mid: 0.5,
    },
    deflection_accuracy: {
        label: 'Deflection Accuracy',
        group: 'Generation',
        description: 'How often the system correctly refused off-topic or unanswerable questions rather than hallucinating an answer.',
        good: 0.9, mid: 0.7,
    },
    avg_citation_density: {
        label: 'Citation Density',
        group: 'Generation',
        description: 'Average number of [Source N] citations per sentence. A well-cited answer typically has 0.5-2 per sentence.',
        isRaw: true,
    },
    avg_latency_ms: {
        label: 'Avg Latency',
        group: 'Performance',
        description: 'Average end-to-end time per query including retrieval, reranking, and LLM generation.',
        isRaw: true, suffix: 'ms',
    },
    total_queries: {
        label: 'Total Queries',
        group: 'Performance',
        description: 'Number of benchmark queries evaluated.',
        isRaw: true,
    },
    // RAGTruth
    case_level_accuracy: {
        label: 'Case Accuracy',
        group: 'Hallucination Detection',
        description: 'Binary classification accuracy — correctly identifying whether a response contains hallucinations or not.',
        good: 0.9, mid: 0.7,
    },
    hallucination_rate: {
        label: 'Hallucination Rate',
        group: 'Hallucination Detection',
        description: 'Percentage of generated responses that were flagged as containing hallucinated content. Lower is better.',
        good: 0.1, mid: 0.3, invert: true,
    },
    avg_span_f1: {
        label: 'Span F1',
        group: 'Hallucination Detection',
        description: 'Token-level F1 between predicted and ground-truth hallucination spans. Measures how precisely hallucinated text is identified.',
        good: 0.7, mid: 0.4,
    },
}

function getScoreColor(value, def) {
    if (!def || def.isRaw) return 'text-gray-100'
    if (def.invert) {
        if (value <= def.good) return 'text-emerald-400'
        if (value <= def.mid) return 'text-amber-400'
        return 'text-red-400'
    }
    if (value >= def.good) return 'text-emerald-400'
    if (value >= def.mid) return 'text-amber-400'
    return 'text-red-400'
}

function getBarColor(value, def) {
    if (!def || def.isRaw) return '#6b7280'
    if (def.invert) {
        if (value <= def.good) return '#4ade80'
        if (value <= def.mid) return '#fbbf24'
        return '#f87171'
    }
    if (value >= def.good) return '#4ade80'
    if (value >= def.mid) return '#fbbf24'
    return '#f87171'
}

function formatValue(value, def) {
    if (typeof value !== 'number') return value
    if (def?.suffix === 'ms') return `${Math.round(value).toLocaleString()}ms`
    if (def?.isRaw) return value % 1 === 0 ? value : value.toFixed(2)
    return `${(value * 100).toFixed(1)}%`
}

// ── Metric card component ─────────────────────────────────────────────────────
function MetricCard({ metricKey, value }) {
    const def = METRIC_DEFS[metricKey]
    const [showTip, setShowTip] = useState(false)
    const label = def?.label || metricKey.replace(/_/g, ' ')
    const color = getScoreColor(value, def)
    const barPct = def?.isRaw ? 0 : Math.min(value * 100, 100)
    const barColor = getBarColor(value, def)

    return (
        <div
            className="bg-gray-800/80 border border-gray-700 rounded-xl p-3 relative cursor-help"
            onMouseEnter={() => setShowTip(true)}
            onMouseLeave={() => setShowTip(false)}
        >
            <div className="text-xs text-gray-500 mb-1 capitalize">{label}</div>
            <div className={`text-lg font-bold ${color}`}>
                {formatValue(value, def)}
            </div>
            {!def?.isRaw && (
                <div className="mt-1.5 h-1 bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all" style={{ width: `${barPct}%`, background: barColor }} />
                </div>
            )}
            {showTip && def?.description && (
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50 bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-xs text-gray-300 w-64 shadow-xl pointer-events-none">
                    {def.description}
                </div>
            )}
        </div>
    )
}

// ── Topic breakdown row ───────────────────────────────────────────────────────
function TopicRow({ topic, data }) {
    const mrr = data.retrieval_mrr ?? 0
    const hitRate = data.retrieval_hit_rate ?? 0
    const citPrec = data.avg_citation_precision ?? 0
    const completeness = data.avg_completeness ?? 0
    const count = data.total_queries ?? 0

    return (
        <tr className="border-b border-gray-800 hover:bg-gray-800/50 text-xs">
            <td className="py-2 px-3 text-gray-300 capitalize font-medium">{topic.replace(/_/g, ' ')}</td>
            <td className="py-2 px-3 text-gray-500 text-center">{count}</td>
            <td className={`py-2 px-3 text-center ${getScoreColor(mrr, METRIC_DEFS.retrieval_mrr)}`}>{(mrr * 100).toFixed(0)}%</td>
            <td className={`py-2 px-3 text-center ${getScoreColor(hitRate, METRIC_DEFS.retrieval_hit_rate)}`}>{(hitRate * 100).toFixed(0)}%</td>
            <td className={`py-2 px-3 text-center ${getScoreColor(citPrec, METRIC_DEFS.avg_citation_precision)}`}>{(citPrec * 100).toFixed(0)}%</td>
            <td className={`py-2 px-3 text-center ${getScoreColor(completeness, METRIC_DEFS.avg_completeness)}`}>{(completeness * 100).toFixed(0)}%</td>
        </tr>
    )
}

// ── Try-it panel ──────────────────────────────────────────────────────────────
function TryItPanel({ examples }) {
    const [selectedIdx, setSelectedIdx] = useState(null)
    const [running, setRunning] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [filterTopic, setFilterTopic] = useState('')
    const [filterType, setFilterType] = useState('')

    const topics = [...new Set(examples.map(e => e.topic))].sort()
    const types = [...new Set(examples.map(e => e.query_type))].sort()

    const filtered = examples.filter(e => {
        if (filterTopic && e.topic !== filterTopic) return false
        if (filterType && e.query_type !== filterType) return false
        if (e.should_deflect) return false // skip deflection queries for try-it
        return true
    })

    const selected = selectedIdx !== null ? filtered[selectedIdx] : null

    const handleTry = async () => {
        if (!selected) return
        setRunning(true)
        setError(null)
        setResult(null)
        try {
            const data = await queryRAG(selected.question, 5)
            setResult(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setRunning(false)
        }
    }

    const diffColor = { easy: 'text-emerald-400', medium: 'text-amber-400', hard: 'text-red-400' }

    return (
        <div>
            {/* Filters */}
            <div className="flex items-center gap-3 mb-3">
                <select
                    value={filterTopic}
                    onChange={e => { setFilterTopic(e.target.value); setSelectedIdx(null); setResult(null) }}
                    className="bg-gray-800 border border-gray-700 rounded-md text-xs text-gray-300 px-2 py-1.5"
                >
                    <option value="">All topics</option>
                    {topics.map(t => <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>)}
                </select>
                <select
                    value={filterType}
                    onChange={e => { setFilterType(e.target.value); setSelectedIdx(null); setResult(null) }}
                    className="bg-gray-800 border border-gray-700 rounded-md text-xs text-gray-300 px-2 py-1.5"
                >
                    <option value="">All types</option>
                    {types.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
                <span className="text-xs text-gray-600">{filtered.length} examples</span>
            </div>

            {/* Question list */}
            <div className="max-h-64 overflow-y-auto space-y-1 mb-4">
                {filtered.map((ex, i) => (
                    <button
                        key={ex.id}
                        onClick={() => { setSelectedIdx(i); setResult(null); setError(null) }}
                        className={`w-full text-left px-3 py-2 rounded-lg text-xs transition flex items-center gap-2 ${
                            selectedIdx === i
                                ? 'bg-blue-600/20 border border-blue-500/40 text-gray-200'
                                : 'bg-gray-800/50 border border-transparent hover:border-gray-700 text-gray-400'
                        }`}
                    >
                        <span className={`font-mono text-[10px] ${diffColor[ex.difficulty] || 'text-gray-500'}`}>
                            {ex.difficulty[0].toUpperCase()}
                        </span>
                        <span className="truncate flex-1">{ex.question}</span>
                        <span className="text-gray-600 text-[10px] shrink-0">{ex.topic.replace(/_/g, ' ')}</span>
                    </button>
                ))}
            </div>

            {/* Selected example details + run */}
            {selected && (
                <div className="bg-gray-800/60 border border-gray-700 rounded-xl p-4">
                    <div className="flex items-start justify-between mb-3">
                        <div>
                            <p className="text-sm text-gray-200 font-medium mb-1">{selected.question}</p>
                            <div className="flex gap-2 text-[10px]">
                                <span className="bg-gray-700 text-gray-400 px-1.5 py-0.5 rounded">{selected.query_type}</span>
                                <span className="bg-gray-700 text-gray-400 px-1.5 py-0.5 rounded">{selected.topic.replace(/_/g, ' ')}</span>
                                <span className={`px-1.5 py-0.5 rounded ${
                                    selected.difficulty === 'easy' ? 'bg-emerald-900/40 text-emerald-400' :
                                    selected.difficulty === 'medium' ? 'bg-amber-900/40 text-amber-400' :
                                    'bg-red-900/40 text-red-400'
                                }`}>{selected.difficulty}</span>
                            </div>
                        </div>
                        <button
                            onClick={handleTry}
                            disabled={running}
                            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 shrink-0"
                        >
                            {running ? <><Spinner size={12} /> Running...</> : <><SearchIcon size={12} /> Try it</>}
                        </button>
                    </div>

                    {/* Ground truth */}
                    <div className="mb-3">
                        <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Expected</div>
                        <div className="flex flex-wrap gap-1.5">
                            <span className="text-xs text-gray-400">
                                Sources: {selected.expected_sources.map(s => (
                                    <span key={s} className="font-mono text-blue-400 mr-1">{s}</span>
                                ))}
                            </span>
                            {selected.expected_answer_contains.length > 0 && (
                                <span className="text-xs text-gray-500 ml-2">
                                    Keywords: {selected.expected_answer_contains.map(k => (
                                        <span key={k} className="bg-gray-700 text-gray-400 px-1 py-0.5 rounded text-[10px] mr-1">{k}</span>
                                    ))}
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Error */}
                    {error && (
                        <div className="flex items-center gap-2 text-xs text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2 mb-3">
                            <AlertIcon size={14} />
                            <span>{error}</span>
                        </div>
                    )}

                    {/* Result */}
                    {result && (() => {
                        // Normalize arxiv IDs: "arxiv_1706_03762" → "1706.03762"
                        const normalizeId = (raw) => {
                            if (!raw) return ''
                            let id = raw
                            if (id.toLowerCase().startsWith('arxiv_')) id = id.slice(6)
                            id = id.replace(/_/g, '.')
                            return id
                        }

                        // Compute evaluation metrics
                        const sources = result.sources || []
                        const retrievedPapers = sources.map(s => normalizeId(s.paper_id)).filter(Boolean)
                        const expectedPapers = (selected.expected_sources || []).map(normalizeId)
                        const expectedKeywords = selected.expected_answer_contains || []
                        const answerText = (result.answer || '').toLowerCase()

                        // Retrieval: did we find the expected paper?
                        const retrievalHit = expectedPapers.some(ep => retrievedPapers.includes(ep))
                        const expectedPosition = expectedPapers.length > 0
                            ? retrievedPapers.findIndex(p => expectedPapers.includes(p)) + 1
                            : 0

                        // Citations: extract [Source N] from the answer text before the Sources: block
                        const answerBody = result.answer?.split('\n\nSources:')[0] || ''
                        const citedNums = [...answerBody.matchAll(/\[Source\s+(\d+)\]/g)].map(m => parseInt(m[1]))
                        const citedPapers = [...new Set(citedNums)].map(n => retrievedPapers[n - 1]).filter(Boolean)
                        const citationCorrect = expectedPapers.some(ep => citedPapers.includes(ep))
                        const citationPrecision = citedPapers.length > 0
                            ? citedPapers.filter(p => expectedPapers.includes(p)).length / citedPapers.length
                            : 0
                        const citationRecall = expectedPapers.length > 0
                            ? expectedPapers.filter(p => citedPapers.includes(p)).length / expectedPapers.length
                            : 0

                        // Completeness: keyword check
                        const foundKeywords = expectedKeywords.filter(k => answerText.includes(k.toLowerCase()))
                        const completeness = expectedKeywords.length > 0
                            ? foundKeywords.length / expectedKeywords.length
                            : 1.0

                        return (
                        <div className="space-y-3">
                            {/* Evaluation scorecard */}
                            <div className="bg-gray-900/80 border border-gray-700 rounded-lg p-3">
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Evaluation Scorecard</div>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                    <div className="text-xs">
                                        <span className="text-gray-500 block mb-0.5">Retrieval</span>
                                        <span className={`font-bold ${retrievalHit ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {retrievalHit ? `Hit (pos ${expectedPosition})` : 'Miss'}
                                        </span>
                                    </div>
                                    <div className="text-xs">
                                        <span className="text-gray-500 block mb-0.5">Citation</span>
                                        <span className={`font-bold ${citationCorrect ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {citationCorrect ? 'Correct' : citedPapers.length > 0 ? 'Wrong source' : 'None cited'}
                                        </span>
                                    </div>
                                    <div className="text-xs">
                                        <span className="text-gray-500 block mb-0.5">Cit. Precision</span>
                                        <span className={`font-bold ${citationPrecision >= 0.5 ? 'text-emerald-400' : citationPrecision > 0 ? 'text-amber-400' : 'text-red-400'}`}>
                                            {(citationPrecision * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="text-xs">
                                        <span className="text-gray-500 block mb-0.5">Completeness</span>
                                        <span className={`font-bold ${completeness >= 0.8 ? 'text-emerald-400' : completeness >= 0.5 ? 'text-amber-400' : 'text-red-400'}`}>
                                            {(completeness * 100).toFixed(0)}%
                                            <span className="font-normal text-gray-600 ml-1">({foundKeywords.length}/{expectedKeywords.length})</span>
                                        </span>
                                    </div>
                                </div>
                                {/* Keyword detail */}
                                {expectedKeywords.length > 0 && (
                                    <div className="mt-2 flex flex-wrap gap-1">
                                        {expectedKeywords.map(k => {
                                            const found = answerText.includes(k.toLowerCase())
                                            return (
                                                <span key={k} className={`text-[10px] px-1.5 py-0.5 rounded ${
                                                    found ? 'bg-emerald-900/30 text-emerald-400' : 'bg-red-900/30 text-red-400'
                                                }`}>
                                                    {found ? '\u2713' : '\u2717'} {k}
                                                </span>
                                            )
                                        })}
                                    </div>
                                )}
                            </div>

                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Generated Answer</div>
                                <div className="text-xs text-gray-300 bg-gray-900/60 rounded-lg p-3 max-h-48 overflow-y-auto leading-relaxed">
                                    {formatAnswer(result.answer)}
                                </div>
                            </div>

                            {/* Sources retrieved */}
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Retrieved Sources</div>
                                <div className="space-y-1">
                                    {sources.map((src, i) => {
                                        const isExpected = expectedPapers.includes(normalizeId(src.paper_id))
                                        const isCited = citedNums.includes(i + 1)
                                        return (
                                            <div key={i} className={`text-xs px-2 py-1.5 rounded flex items-center gap-2 ${
                                                isExpected ? 'bg-emerald-900/20 border border-emerald-800/40' : 'bg-gray-900/40'
                                            }`}>
                                                <span className={`font-mono w-5 ${isCited ? 'text-blue-400' : 'text-gray-500'}`}>{i + 1}</span>
                                                <span className={`font-mono text-[10px] ${isExpected ? 'text-emerald-400' : 'text-gray-500'}`}>
                                                    {normalizeId(src.paper_id) || '?'}
                                                </span>
                                                <span className="text-gray-400 truncate flex-1">{src.title || src.text_preview?.slice(0, 80)}</span>
                                                <span className="text-gray-600 font-mono text-[10px]">{src.score?.toFixed(3)}</span>
                                                {isExpected && <span className="text-emerald-400 text-[10px]">expected</span>}
                                                {isCited && <span className="text-blue-400 text-[10px]">cited</span>}
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>

                            {/* Quick metrics */}
                            <div className="flex gap-4 text-xs">
                                <span className="text-gray-500">Latency: <span className="text-gray-300">{Math.round(result.latency_ms)}ms</span></span>
                                <span className="text-gray-500">Deflected: <span className="text-gray-300">{result.deflected ? 'Yes' : 'No'}</span></span>
                                <span className="text-gray-500">Backend: <span className="text-gray-300">{result.model}</span></span>
                            </div>
                        </div>
                        )
                    })()}
                </div>
            )}
        </div>
    )
}


// ── RAGTruth try-it panel ────────────────────────────────────────────────────
// Runs a question through the live RAG pipeline, then checks the generated
// answer for hallucinations against the retrieved source passages.
function RagtruthTryItPanel() {
    const [question, setQuestion] = useState('')
    const [running, setRunning] = useState(false)
    const [step, setStep] = useState('')  // 'querying' | 'detecting'
    const [pipelineResult, setPipelineResult] = useState(null)
    const [detection, setDetection] = useState(null)
    const [error, setError] = useState(null)

    const SAMPLE_QUESTIONS = [
        'What is the Transformer architecture?',
        'How does LoRA reduce the number of trainable parameters?',
        'Explain the attention mechanism in neural networks.',
        'What are the key contributions of the BERT paper?',
        'How does RLHF work for language model alignment?',
        'What is FlashAttention and why is it faster?',
    ]

    const handleRun = async (q) => {
        const query = q || question
        if (!query.trim()) return
        setRunning(true)
        setStep('querying')
        setError(null)
        setPipelineResult(null)
        setDetection(null)

        try {
            // Step 1: Run the RAG pipeline
            const result = await queryRAG(query, 5)
            setPipelineResult(result)

            // Step 2: Run hallucination detection on answer vs retrieved sources
            setStep('detecting')
            const sourceContext = (result.sources || [])
                .map((s, i) => `[Source ${i + 1}]: ${s.text_preview || ''}`)
                .join('\n\n')
            const det = await detectHallucination(sourceContext, result.answer || '')
            setDetection(det)
        } catch (err) {
            setError(err.message)
        } finally {
            setRunning(false)
            setStep('')
        }
    }

    // Highlight detected spans in the answer text
    const renderWithHighlights = (text, spans) => {
        if (!spans?.length) return <span>{text}</span>

        const markers = []
        for (const s of spans) {
            const spanText = s.text || s
            const idx = text.indexOf(spanText)
            if (idx >= 0) markers.push({ start: idx, end: idx + spanText.length })
        }
        markers.sort((a, b) => a.start - b.start)

        const parts = []
        let pos = 0
        for (const m of markers) {
            if (m.start > pos) parts.push({ text: text.slice(pos, m.start), flagged: false })
            if (m.start >= pos) {
                parts.push({ text: text.slice(m.start, m.end), flagged: true })
                pos = m.end
            }
        }
        if (pos < text.length) parts.push({ text: text.slice(pos), flagged: false })

        return parts.map((p, i) =>
            p.flagged
                ? <span key={i} className="bg-amber-900/40 text-amber-300 border-b border-amber-500" title="Potentially unsupported claim">{p.text}</span>
                : <span key={i}>{p.text}</span>
        )
    }

    return (
        <div>
            {/* Custom question input */}
            <div className="flex gap-2 mb-3">
                <input
                    type="text"
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleRun()}
                    placeholder="Ask a question about AI/ML research..."
                    className="flex-1 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-200 px-3 py-2 placeholder-gray-600 focus:outline-none focus:border-blue-500"
                />
                <button
                    onClick={() => handleRun()}
                    disabled={running || !question.trim()}
                    className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs px-4 py-2 rounded-lg transition flex items-center gap-1.5 shrink-0"
                >
                    {running ? <><Spinner size={12} /> {step === 'querying' ? 'Querying...' : 'Checking...'}</> : <><SearchIcon size={12} /> Run &amp; Check</>}
                </button>
            </div>

            {/* Sample questions */}
            <div className="flex flex-wrap gap-1.5 mb-4">
                {SAMPLE_QUESTIONS.map(q => (
                    <button
                        key={q}
                        onClick={() => { setQuestion(q); handleRun(q) }}
                        disabled={running}
                        className="text-[10px] bg-gray-800/60 border border-gray-700 hover:border-gray-600 text-gray-400 hover:text-gray-300 px-2 py-1 rounded-lg transition disabled:opacity-50"
                    >
                        {q}
                    </button>
                ))}
            </div>

            {error && (
                <div className="flex items-center gap-2 text-xs text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2 mb-3">
                    <AlertIcon size={14} />
                    <span>{error}</span>
                </div>
            )}

            {/* Pipeline result + detection */}
            {pipelineResult && (
                <div className="bg-gray-800/60 border border-gray-700 rounded-xl p-4 space-y-3">
                    {/* Detection scorecard */}
                    {detection && (
                        <div className="bg-gray-900/80 border border-gray-700 rounded-lg p-3">
                            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Hallucination Check</div>
                            <div className="grid grid-cols-3 gap-3">
                                <div className="text-xs">
                                    <span className="text-gray-500 block mb-0.5">Verdict</span>
                                    <span className={`font-bold ${detection.has_hallucination ? 'text-amber-400' : 'text-emerald-400'}`}>
                                        {detection.has_hallucination ? 'Potential issues found' : 'Looks faithful'}
                                    </span>
                                </div>
                                <div className="text-xs">
                                    <span className="text-gray-500 block mb-0.5">Unsupported Spans</span>
                                    <span className={`font-bold ${(detection.flagged_spans?.length || 0) > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                                        {detection.flagged_spans?.length || 0}
                                    </span>
                                </div>
                                <div className="text-xs">
                                    <span className="text-gray-500 block mb-0.5">Latency</span>
                                    <span className="font-bold text-gray-200">
                                        {Math.round(pipelineResult.latency_ms)}ms + {detection.latency_ms}ms
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Answer with highlighted spans */}
                    <div>
                        <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">
                            Generated Answer
                            {detection?.flagged_spans?.length > 0 && (
                                <span className="ml-2 text-amber-400 normal-case">(amber = potentially unsupported)</span>
                            )}
                        </div>
                        <div className="text-xs text-gray-300 bg-gray-900/60 rounded-lg p-3 max-h-48 overflow-y-auto leading-relaxed">
                            {detection?.flagged_spans?.length > 0
                                ? renderWithHighlights(pipelineResult.answer || '', detection.flagged_spans)
                                : formatAnswer(pipelineResult.answer)
                            }
                        </div>
                    </div>

                    {/* Flagged spans detail */}
                    {detection?.flagged_spans?.length > 0 && (
                        <div>
                            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Flagged Spans</div>
                            <div className="space-y-1">
                                {detection.flagged_spans.map((s, i) => (
                                    <div key={i} className="text-xs bg-amber-900/20 border border-amber-800/30 rounded px-2 py-1.5">
                                        <span className="text-amber-400 text-[10px] font-mono mr-2">{s.type}</span>
                                        <span className="text-gray-300">"{s.text.slice(0, 150)}{s.text.length > 150 ? '...' : ''}"</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Sources */}
                    <div>
                        <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Retrieved Sources</div>
                        <div className="space-y-1">
                            {(pipelineResult.sources || []).map((src, i) => (
                                <div key={i} className="text-xs bg-gray-900/40 px-2 py-1.5 rounded flex items-center gap-2">
                                    <span className="font-mono text-gray-500 w-5">{i + 1}</span>
                                    <span className="text-gray-400 truncate flex-1">{src.title || src.text_preview?.slice(0, 80)}</span>
                                    <span className="text-gray-600 font-mono text-[10px]">{src.score?.toFixed(3)}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Quick metrics */}
                    <div className="flex gap-4 text-xs">
                        <span className="text-gray-500">Deflected: <span className="text-gray-300">{pipelineResult.deflected ? 'Yes' : 'No'}</span></span>
                        <span className="text-gray-500">Backend: <span className="text-gray-300">{pipelineResult.model}</span></span>
                    </div>
                </div>
            )}
        </div>
    )
}


// ── Main BenchmarksTab ────────────────────────────────────────────────────────
export function BenchmarksTab() {
    const [activeBenchmark, setActiveBenchmark] = useState('ragbench')
    const [results, setResults] = useState(null)
    const [examples, setExamples] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [showTopics, setShowTopics] = useState(false)
    const [showExamples, setShowExamples] = useState(false)

    // Load results when benchmark changes
    useEffect(() => {
        setLoading(true)
        setError(null)
        setResults(null)
        fetchBenchmarkLatest(activeBenchmark)
            .then(data => setResults(data))
            .catch(err => setError(err.message))
            .finally(() => setLoading(false))
    }, [activeBenchmark])

    // Load examples once
    useEffect(() => {
        fetchBenchmarkExamples()
            .then(data => setExamples(data.examples))
            .catch(() => {})
    }, [])

    const benchmarkInfo = {
        ragbench: {
            name: 'RAG-Bench',
            fullName: 'End-to-End RAG Pipeline Evaluation',
            description: 'Tests the complete pipeline against 77 hand-crafted ground-truth queries on the AI/ML research paper corpus. Measures retrieval accuracy, citation quality, answer completeness, and deflection of off-topic questions.',
            source: 'Custom ground-truth dataset (77 entries across 15 ML topics)',
        },
        ragtruth: {
            name: 'RAGTruth',
            fullName: 'RAG Hallucination Corpus (ACL 2024)',
            description: 'Published benchmark that evaluates hallucination detection at case and span level. Given a context and generated response, measures whether the system can identify unsupported claims.',
            source: 'ParticleMedia/RAGTruth (ACL 2024)',
        },
    }

    const info = benchmarkInfo[activeBenchmark]

    // Determine which metric keys to show
    const summaryKeys = results?.summary
        ? Object.keys(results.summary).filter(k => METRIC_DEFS[k] && typeof results.summary[k] === 'number')
        : []

    // Group metrics
    const groups = {}
    for (const k of summaryKeys) {
        const group = METRIC_DEFS[k]?.group || 'Other'
        if (!groups[group]) groups[group] = []
        groups[group].push(k)
    }

    return (
        <div className="flex-1 overflow-y-auto">
            <div className="max-w-5xl mx-auto px-4 py-6">
                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 bg-blue-600/10 border border-blue-600/20 rounded-xl flex items-center justify-center">
                        <BarChartIcon size={20} />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-gray-100">Benchmark Results</h2>
                        <p className="text-xs text-gray-500">Pipeline performance against standardized evaluation datasets</p>
                    </div>
                </div>

                {/* Benchmark selector */}
                <div className="flex gap-2 mb-6">
                    {Object.entries(benchmarkInfo).map(([key, bm]) => (
                        <button
                            key={key}
                            onClick={() => { setActiveBenchmark(key); setShowTopics(false); setShowExamples(false) }}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                                activeBenchmark === key
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-700'
                            }`}
                        >
                            {bm.name}
                        </button>
                    ))}
                </div>

                {/* Benchmark info card */}
                <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4 mb-6">
                    <h3 className="text-sm font-semibold text-gray-200">{info.fullName}</h3>
                    <p className="text-xs text-gray-500 mt-0.5 mb-2">{info.source}</p>
                    <p className="text-xs text-gray-400">{info.description}</p>
                </div>

                {/* Loading */}
                {loading && (
                    <div className="flex items-center justify-center py-16">
                        <Spinner size={24} />
                        <span className="ml-3 text-sm text-gray-400">Loading results...</span>
                    </div>
                )}

                {/* Error */}
                {error && !loading && (
                    <div className="flex flex-col items-center justify-center py-16 text-center">
                        <div className="w-16 h-16 bg-gray-800/50 rounded-full flex items-center justify-center mb-4">
                            <BarChartIcon size={28} />
                        </div>
                        <h3 className="text-sm font-medium text-gray-400 mb-2">No results available</h3>
                        <p className="text-xs text-gray-600 max-w-sm">
                            Run the benchmark evaluation from the command line to generate results:
                        </p>
                        <code className="mt-2 text-xs bg-gray-800 text-gray-300 px-3 py-1.5 rounded-lg border border-gray-700">
                            {activeBenchmark === 'ragbench' ? 'python -m rag_bench.eval.run' : 'python -m rag_bench.eval.ragtruth'}
                        </code>
                    </div>
                )}

                {/* Results */}
                {results && !loading && (
                    <div>
                        {/* Metric groups */}
                        {Object.entries(groups).map(([groupName, keys]) => (
                            <div key={groupName} className="mb-6">
                                <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">{groupName}</h4>
                                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                                    {keys.map(k => (
                                        <MetricCard key={k} metricKey={k} value={results.summary[k]} />
                                    ))}
                                </div>
                            </div>
                        ))}

                        {/* RAGTruth extra summary info */}
                        {activeBenchmark === 'ragtruth' && results.summary && (
                            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4 mb-6">
                                <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Confusion Matrix</h4>
                                <div className="grid grid-cols-2 gap-3 max-w-sm">
                                    {[
                                        ['True Negatives', results.summary.true_negatives, 'text-emerald-400'],
                                        ['True Positives', results.summary.true_positives, 'text-emerald-400'],
                                        ['False Positives', results.summary.false_positives, 'text-red-400'],
                                        ['False Negatives', results.summary.false_negatives, 'text-red-400'],
                                    ].map(([label, val, color]) => val !== undefined && (
                                        <div key={label} className="text-xs">
                                            <span className="text-gray-500">{label}: </span>
                                            <span className={`font-bold ${color}`}>{val}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Topic breakdown (RAG-Bench only) */}
                        {activeBenchmark === 'ragbench' && results.by_topic && (
                            <div className="mb-6">
                                <button
                                    onClick={() => setShowTopics(!showTopics)}
                                    className="flex items-center gap-2 text-sm text-gray-300 hover:text-gray-100 transition mb-2"
                                >
                                    {showTopics ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                    Performance by Topic
                                </button>
                                {showTopics && (
                                    <div className="bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
                                        <table className="w-full">
                                            <thead>
                                                <tr className="text-[10px] text-gray-500 uppercase tracking-wider border-b border-gray-700">
                                                    <th className="text-left py-2 px-3">Topic</th>
                                                    <th className="text-center py-2 px-3">Queries</th>
                                                    <th className="text-center py-2 px-3">MRR</th>
                                                    <th className="text-center py-2 px-3">Hit Rate</th>
                                                    <th className="text-center py-2 px-3">Cit. Prec</th>
                                                    <th className="text-center py-2 px-3">Complete</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {Object.entries(results.by_topic)
                                                    .sort((a, b) => (b[1].retrieval_mrr || 0) - (a[1].retrieval_mrr || 0))
                                                    .map(([topic, data]) => (
                                                        <TopicRow key={topic} topic={topic} data={data} />
                                                    ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Timestamp */}
                        {results.timestamp && (
                            <p className="text-xs text-gray-600 mb-4">
                                Last evaluated: {new Date(results.timestamp).toLocaleString()}
                            </p>
                        )}
                        {results._source_file && (
                            <p className="text-xs text-gray-600 mb-4">
                                Source: {results._source_file}
                                {results.summary?.total_queries && ` (${results.summary.total_queries} queries)`}
                            </p>
                        )}
                    </div>
                )}

                {/* Try it section (RAG-Bench) */}
                {activeBenchmark === 'ragbench' && examples && (
                    <div className="mt-6 border-t border-gray-800 pt-6">
                        <button
                            onClick={() => setShowExamples(!showExamples)}
                            className="flex items-center gap-2 text-sm font-medium text-gray-200 hover:text-gray-100 transition mb-4"
                        >
                            {showExamples ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                            <SearchIcon size={14} />
                            Try a Benchmark Example
                        </button>
                        <p className="text-xs text-gray-500 mb-4">
                            Pick a question from the benchmark and run it through the pipeline to see how retrieval, generation, and citation work in real time.
                        </p>
                        {showExamples && <TryItPanel examples={examples} />}
                    </div>
                )}

                {/* Try it section (RAGTruth) */}
                {activeBenchmark === 'ragtruth' && (
                    <div className="mt-6 border-t border-gray-800 pt-6">
                        <button
                            onClick={() => setShowExamples(!showExamples)}
                            className="flex items-center gap-2 text-sm font-medium text-gray-200 hover:text-gray-100 transition mb-4"
                        >
                            {showExamples ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                            <SearchIcon size={14} />
                            Try Hallucination Detection
                        </button>
                        <p className="text-xs text-gray-500 mb-4">
                            Ask a question about AI/ML research. The pipeline generates an answer, then the hallucination detector checks each sentence against the retrieved sources.
                        </p>
                        {showExamples && <RagtruthTryItPanel />}
                    </div>
                )}

                {/* ── Eval Trends ── */}
                <TrendsPanel />

                {/* ── Auto-Eval Schedule ── */}
                <AutoEvalPanel />
            </div>
        </div>
    )
}


// ── Trend chart (lightweight SVG sparkline) ──

function TrendChart({ data, series, title, yFormat, height = 140 }) {
    if (!data || data.length < 2) return null

    const W = 560, H = height, PAD_L = 48, PAD_R = 12, PAD_T = 6, PAD_B = 20
    const chartW = W - PAD_L - PAD_R
    const chartH = H - PAD_T - PAD_B

    let maxVal = 0
    for (const s of series) {
        for (const d of data) {
            const v = d[s.key]
            if (v != null && v > maxVal) maxVal = v
        }
    }
    if (maxVal === 0) maxVal = 1

    const yMax = Math.ceil(maxVal * 10) / 10 || 1
    function xPos(i) { return PAD_L + (i / (data.length - 1)) * chartW }
    function yPos(v) { return PAD_T + chartH - (Math.min(v, yMax) / yMax) * chartH }

    const fmtY = yFormat || (v => `${v.toFixed(2)}`)

    // Y-axis ticks
    const yTicks = [0, yMax / 2, yMax]

    function linePath(s) {
        return data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xPos(i)},${yPos(d[s.key] || 0)}`).join('')
    }

    return (
        <div className="mt-3">
            <div className="rounded-lg overflow-hidden" style={{ background: '#181b28', border: '1px solid #2a2f3e' }}>
                <div className="px-3 pt-2.5 pb-1">
                    <span className="text-[11px] font-medium" style={{ color: '#8b8fa3' }}>{title}</span>
                </div>
                <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ background: '#111217', display: 'block' }}>
                    {yTicks.map((v, i) => (
                        <g key={i}>
                            <line x1={PAD_L} y1={yPos(v)} x2={W - PAD_R} y2={yPos(v)} stroke="#1e2130" strokeWidth="1" strokeDasharray="4,3" />
                            <text x={PAD_L - 6} y={yPos(v) + 3} textAnchor="end" fill="#6c7183" fontSize="8" fontFamily="ui-monospace,monospace">{fmtY(v)}</text>
                        </g>
                    ))}
                    {series.map((s, idx) => (
                        <path key={idx} d={linePath(s)} fill="none" stroke={s.color} strokeWidth="1.5" strokeLinejoin="round" />
                    ))}
                    {/* Dots on last point */}
                    {series.map((s, idx) => {
                        const last = data[data.length - 1]
                        return <circle key={idx} cx={xPos(data.length - 1)} cy={yPos(last[s.key] || 0)} r="3" fill={s.color} />
                    })}
                    {/* X-axis labels (first and last) */}
                    <text x={xPos(0)} y={H - 4} textAnchor="start" fill="#6c7183" fontSize="8" fontFamily="ui-monospace,monospace">
                        {data[0].timestamp?.slice(5, 10) || ''}
                    </text>
                    <text x={xPos(data.length - 1)} y={H - 4} textAnchor="end" fill="#6c7183" fontSize="8" fontFamily="ui-monospace,monospace">
                        {data[data.length - 1].timestamp?.slice(5, 10) || ''}
                    </text>
                </svg>
                <div className="flex items-center gap-4 px-3 py-1.5 justify-center flex-wrap" style={{ borderTop: '1px solid #1e2130' }}>
                    {series.map(s => {
                        const last = data[data.length - 1]
                        const prev = data.length >= 2 ? data[data.length - 2] : null
                        const val = last[s.key] || 0
                        const delta = prev ? val - (prev[s.key] || 0) : 0
                        const isUp = delta > 0.001
                        const isDown = delta < -0.001
                        return (
                            <span key={s.label} className="flex items-center gap-1.5 text-[10px]" style={{ color: '#b3b8c8' }}>
                                <span className="inline-block w-3 h-0.5 rounded" style={{ backgroundColor: s.color }} />
                                {s.label}
                                <span className="font-mono" style={{ color: s.color }}>{fmtY(val)}</span>
                                {isUp && <span className="text-green-400">+{delta.toFixed(3)}</span>}
                                {isDown && <span className="text-red-400">{delta.toFixed(3)}</span>}
                            </span>
                        )
                    })}
                </div>
            </div>
        </div>
    )
}


function TrendsPanel() {
    const [trends, setTrends] = useState(null)
    const [loading, setLoading] = useState(true)
    const [runType, setRunType] = useState('production')

    useEffect(() => {
        setLoading(true)
        fetchBenchmarkTrends(runType)
            .then(data => setTrends(data?.trends || []))
            .catch(() => setTrends([]))
            .finally(() => setLoading(false))
    }, [runType])

    if (loading) return <div className="text-xs text-gray-500 mt-6"><Spinner size={14} /> Loading trends...</div>
    if (!trends || trends.length < 2) return null

    return (
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 mt-6">
            <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                    <BarChartIcon size={16} /> Eval Trends
                </h3>
                <div className="flex gap-1 text-xs">
                    {['production', 'manual', 'all'].map(t => (
                        <button
                            key={t}
                            onClick={() => setRunType(t)}
                            className={`px-2 py-0.5 rounded transition-colors ${
                                runType === t
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-700 text-gray-400 hover:text-gray-200'
                            }`}
                        >
                            {t.charAt(0).toUpperCase() + t.slice(1)}
                        </button>
                    ))}
                </div>
            </div>
            <p className="text-xs text-gray-500 mb-3">
                Metrics across {trends.length} {runType !== 'all' ? runType + ' ' : ''}evaluation runs.
            </p>

            <TrendChart
                data={trends}
                title="Retrieval Quality"
                series={[
                    { key: 'retrieval_mrr', color: '#38bdf8', label: 'MRR' },
                    { key: 'retrieval_ndcg_at_5', color: '#4ade80', label: 'NDCG@5' },
                    { key: 'retrieval_hit_rate', color: '#facc15', label: 'Hit Rate' },
                ]}
            />

            <TrendChart
                data={trends}
                title="Generation Quality"
                series={[
                    { key: 'avg_citation_precision', color: '#a78bfa', label: 'Citation Prec' },
                    { key: 'avg_completeness', color: '#34d399', label: 'Completeness' },
                    { key: 'avg_faithfulness', color: '#f87171', label: 'Faithfulness' },
                ]}
            />
        </div>
    )
}


function AutoEvalPanel() {
    const [schedule, setSchedule] = useState(null)
    const [loading, setLoading] = useState(true)
    const [toggling, setToggling] = useState(false)

    useEffect(() => {
        fetchEvalSchedule()
            .then(setSchedule)
            .catch(() => setSchedule(null))
            .finally(() => setLoading(false))
    }, [])

    async function handleToggle() {
        if (!schedule) return
        setToggling(true)
        try {
            const updated = await updateEvalSchedule(!schedule.enabled, schedule.interval_hours)
            setSchedule(updated)
        } catch { /* silent */ }
        finally { setToggling(false) }
    }

    if (loading) return null
    if (!schedule) return null

    return (
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 mt-6">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                    <ZapIcon size={16} /> Scheduled Auto-Eval
                </h3>
                <button
                    onClick={handleToggle}
                    disabled={toggling}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${schedule.enabled ? 'bg-green-600' : 'bg-gray-600'}`}
                >
                    <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${schedule.enabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
                </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                <div className="bg-gray-900 rounded-lg p-2.5">
                    <div className="text-gray-500 mb-0.5">Status</div>
                    <div className={schedule.enabled ? 'text-green-400' : 'text-gray-400'}>
                        {schedule.enabled ? 'Active' : 'Disabled'}
                    </div>
                </div>
                <div className="bg-gray-900 rounded-lg p-2.5">
                    <div className="text-gray-500 mb-0.5">Interval</div>
                    <div className="text-gray-200">Every {schedule.interval_hours}h</div>
                </div>
                <div className="bg-gray-900 rounded-lg p-2.5">
                    <div className="text-gray-500 mb-0.5">Last Run</div>
                    <div className="text-gray-200">{schedule.last_run ? new Date(schedule.last_run).toLocaleDateString() : '—'}</div>
                </div>
            </div>

            {schedule.last_run_summary && Object.keys(schedule.last_run_summary).length > 0 && (
                <div className="mt-3 text-xs text-gray-400">
                    Last: MRR {schedule.last_run_summary.retrieval_mrr?.toFixed(3) || '—'} · Hit Rate {schedule.last_run_summary.retrieval_hit_rate?.toFixed(3) || '—'} · Latency {Math.round(schedule.last_run_summary.avg_latency_ms || 0)}ms
                </div>
            )}
        </div>
    )
}
