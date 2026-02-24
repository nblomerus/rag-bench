import React, { useState, useEffect, useRef } from 'react'
import ReactDOM from 'react-dom'
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
    if (!def || def.isRaw) return 'var(--apple-text-primary)'
    if (def.invert) {
        if (value <= def.good) return 'var(--apple-green)'
        if (value <= def.mid) return 'var(--apple-yellow)'
        return 'var(--apple-red)'
    }
    if (value >= def.good) return 'var(--apple-green)'
    if (value >= def.mid) return 'var(--apple-yellow)'
    return 'var(--apple-red)'
}

function getBarColor(value, def) {
    if (!def || def.isRaw) return 'var(--apple-text-quaternary)'
    if (def.invert) {
        if (value <= def.good) return 'var(--apple-green)'
        if (value <= def.mid) return 'var(--apple-yellow)'
        return 'var(--apple-red)'
    }
    if (value >= def.good) return 'var(--apple-green)'
    if (value >= def.mid) return 'var(--apple-yellow)'
    return 'var(--apple-red)'
}

function formatValue(value, def) {
    if (typeof value !== 'number') return value
    if (def?.suffix === 'ms') return `${Math.round(value).toLocaleString()}ms`
    if (def?.isRaw) return value % 1 === 0 ? value : value.toFixed(2)
    return `${(value * 100).toFixed(1)}%`
}

// ── Metric card component with ring gauge ────────────────────────────────────
function MetricCard({ metricKey, value }) {
    const def = METRIC_DEFS[metricKey]
    const [tipPos, setTipPos] = useState(null)
    const cardRef = useRef(null)
    const label = def?.label || metricKey.replace(/_/g, ' ')
    const scoreColor = getScoreColor(value, def)
    const barColor = getBarColor(value, def)
    const isPercent = !def?.isRaw

    // Ring gauge geometry
    const R = 28, STROKE = 4
    const C = 2 * Math.PI * R
    const pct = isPercent ? Math.min(value, 1) : 0
    const offset = C * (1 - pct)

    const showTip = () => {
        if (cardRef.current) {
            const r = cardRef.current.getBoundingClientRect()
            setTipPos({ top: r.top, left: r.left + r.width / 2 })
        }
    }

    return (
        <div
            ref={cardRef}
            className="glass-card p-4 cursor-help"
            onMouseEnter={showTip}
            onMouseLeave={() => setTipPos(null)}
        >
            {isPercent ? (
                <div className="flex items-center gap-3">
                    {/* Ring gauge */}
                    <svg width="64" height="64" viewBox="0 0 64 64" className="shrink-0">
                        <circle cx="32" cy="32" r={R} fill="none" stroke="var(--apple-bg-tertiary)" strokeWidth={STROKE} />
                        <circle cx="32" cy="32" r={R} fill="none" stroke={barColor} strokeWidth={STROKE}
                            strokeDasharray={C} strokeDashoffset={offset}
                            strokeLinecap="round" transform="rotate(-90 32 32)"
                            style={{ transition: 'stroke-dashoffset 0.6s ease' }} />
                        <text x="32" y="34" textAnchor="middle" dominantBaseline="central"
                            style={{ fill: scoreColor, fontSize: '13px', fontWeight: 700, fontFamily: 'ui-monospace, monospace' }}>
                            {(value * 100).toFixed(0)}
                        </text>
                    </svg>
                    <div className="min-w-0">
                        <div className="text-xs capitalize" style={{ color: 'var(--apple-text-tertiary)' }}>{label}</div>
                        <div className="text-sm font-bold mt-0.5" style={{ color: scoreColor }}>
                            {formatValue(value, def)}
                        </div>
                    </div>
                </div>
            ) : (
                <div>
                    <div className="text-xs mb-1 capitalize" style={{ color: 'var(--apple-text-tertiary)' }}>{label}</div>
                    <div className="text-lg font-bold" style={{ color: scoreColor }}>
                        {formatValue(value, def)}
                    </div>
                </div>
            )}
            {tipPos && def?.description && ReactDOM.createPortal(
                <div style={{
                    position: 'fixed',
                    top: tipPos.top - 6,
                    left: tipPos.left,
                    transform: 'translateX(-50%) translateY(-100%)',
                    background: 'var(--apple-bg-elevated)',
                    color: 'var(--apple-text-primary)',
                    fontSize: '12px',
                    fontWeight: 400,
                    lineHeight: 1.4,
                    padding: '8px 12px',
                    borderRadius: '10px',
                    border: '1px solid var(--apple-border-primary)',
                    whiteSpace: 'normal',
                    width: 'max-content',
                    maxWidth: '280px',
                    zIndex: 9999,
                    pointerEvents: 'none',
                    boxShadow: 'var(--apple-shadow-md)',
                }}>
                    {def.description}
                </div>,
                document.body
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
        <tr className="hover:opacity-80 text-xs" style={{ borderBottom: '1px solid var(--apple-divider)' }}>
            <td className="py-2 px-3 capitalize font-medium" style={{ color: 'var(--apple-text-primary)' }}>{topic.replace(/_/g, ' ')}</td>
            <td className="py-2 px-3 text-center" style={{ color: 'var(--apple-text-tertiary)' }}>{count}</td>
            <td className="py-2 px-3 text-center" style={{ color: getScoreColor(mrr, METRIC_DEFS.retrieval_mrr) }}>{(mrr * 100).toFixed(0)}%</td>
            <td className="py-2 px-3 text-center" style={{ color: getScoreColor(hitRate, METRIC_DEFS.retrieval_hit_rate) }}>{(hitRate * 100).toFixed(0)}%</td>
            <td className="py-2 px-3 text-center" style={{ color: getScoreColor(citPrec, METRIC_DEFS.avg_citation_precision) }}>{(citPrec * 100).toFixed(0)}%</td>
            <td className="py-2 px-3 text-center" style={{ color: getScoreColor(completeness, METRIC_DEFS.avg_completeness) }}>{(completeness * 100).toFixed(0)}%</td>
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

    const diffColor = { easy: 'var(--apple-green)', medium: 'var(--apple-yellow)', hard: 'var(--apple-red)' }

    return (
        <div>
            {/* Filters */}
            <div className="flex items-center gap-3 mb-3">
                <select
                    value={filterTopic}
                    onChange={e => { setFilterTopic(e.target.value); setSelectedIdx(null); setResult(null) }}
                    className="rounded-md text-xs px-2 py-1.5"
                    style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-primary)', borderColor: 'var(--apple-border-primary)', border: '1px solid var(--apple-border-primary)' }}
                >
                    <option value="">All topics</option>
                    {topics.map(t => <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>)}
                </select>
                <select
                    value={filterType}
                    onChange={e => { setFilterType(e.target.value); setSelectedIdx(null); setResult(null) }}
                    className="rounded-md text-xs px-2 py-1.5"
                    style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-primary)', borderColor: 'var(--apple-border-primary)', border: '1px solid var(--apple-border-primary)' }}
                >
                    <option value="">All types</option>
                    {types.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
                <span className="text-xs" style={{ color: 'var(--apple-text-quaternary)' }}>{filtered.length} examples</span>
            </div>

            {/* Question list */}
            <div className="max-h-64 overflow-y-auto space-y-1 mb-4">
                {filtered.map((ex, i) => (
                    <button
                        key={ex.id}
                        onClick={() => { setSelectedIdx(i); setResult(null); setError(null) }}
                        className={`w-full text-left px-3 py-2 rounded-lg text-xs transition flex items-center gap-2 ${
                            selectedIdx === i
                                ? 'border text-white'
                                : 'border border-transparent hover:opacity-80'
                        }`}
                        style={selectedIdx === i
                            ? { background: 'color-mix(in srgb, var(--apple-accent) 20%, transparent)', borderColor: 'color-mix(in srgb, var(--apple-accent) 40%, transparent)', color: 'var(--apple-text-primary)' }
                            : { background: 'var(--apple-glass-bg)', color: 'var(--apple-text-secondary)' }
                        }
                    >
                        <span className="font-mono text-[10px]" style={{ color: diffColor[ex.difficulty] || 'var(--apple-text-tertiary)' }}>
                            {ex.difficulty[0].toUpperCase()}
                        </span>
                        <span className="truncate flex-1">{ex.question}</span>
                        <span className="text-[10px] shrink-0" style={{ color: 'var(--apple-text-quaternary)' }}>{ex.topic.replace(/_/g, ' ')}</span>
                    </button>
                ))}
            </div>

            {/* Selected example details + run */}
            {selected && (
                <div className="glass-card p-5">
                    <div className="flex items-start justify-between mb-3">
                        <div>
                            <p className="text-sm font-medium mb-1" style={{ color: 'var(--apple-text-primary)' }}>{selected.question}</p>
                            <div className="flex gap-2 text-[10px]">
                                <span className="px-1.5 py-0.5 rounded" style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-secondary)' }}>{selected.query_type}</span>
                                <span className="px-1.5 py-0.5 rounded" style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-secondary)' }}>{selected.topic.replace(/_/g, ' ')}</span>
                                <span className="px-1.5 py-0.5 rounded" style={
                                    selected.difficulty === 'easy' ? { background: 'var(--apple-green-bg)', color: 'var(--apple-green)' } :
                                    selected.difficulty === 'medium' ? { background: 'var(--apple-yellow-bg)', color: 'var(--apple-yellow)' } :
                                    { background: 'var(--apple-red-bg)', color: 'var(--apple-red)' }
                                }>{selected.difficulty}</span>
                            </div>
                        </div>
                        <button
                            onClick={handleTry}
                            disabled={running}
                            className="disabled:opacity-50 text-white text-xs px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 shrink-0"
                            style={{ background: 'var(--apple-accent)' }}
                        >
                            {running ? <><Spinner size={12} /> Running...</> : <><SearchIcon size={12} /> Try it</>}
                        </button>
                    </div>

                    {/* Ground truth */}
                    <div className="mb-3">
                        <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>Expected</div>
                        <div className="flex flex-wrap gap-1.5">
                            <span className="text-xs" style={{ color: 'var(--apple-text-secondary)' }}>
                                Sources: {selected.expected_sources.map(s => (
                                    <span key={s} className="font-mono mr-1" style={{ color: 'var(--apple-accent)' }}>{s}</span>
                                ))}
                            </span>
                            {selected.expected_answer_contains.length > 0 && (
                                <span className="text-xs ml-2" style={{ color: 'var(--apple-text-tertiary)' }}>
                                    Keywords: {selected.expected_answer_contains.map(k => (
                                        <span key={k} className="px-1 py-0.5 rounded text-[10px] mr-1" style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-secondary)' }}>{k}</span>
                                    ))}
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Error */}
                    {error && (
                        <div className="flex items-center gap-2 text-xs rounded-lg px-3 py-2 mb-3" style={{ color: 'var(--apple-red)', background: 'var(--apple-red-bg)', border: '1px solid var(--apple-red-border)' }}>
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
                            <div className="rounded-xl p-3" style={{ background: 'var(--apple-bg-tertiary)', border: '1px solid var(--apple-border-primary)' }}>
                                <div className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--apple-text-tertiary)' }}>Evaluation Scorecard</div>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                    <div className="text-xs">
                                        <span className="block mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Retrieval</span>
                                        <span className="font-bold" style={{ color: retrievalHit ? 'var(--apple-green)' : 'var(--apple-red)' }}>
                                            {retrievalHit ? `Hit (pos ${expectedPosition})` : 'Miss'}
                                        </span>
                                    </div>
                                    <div className="text-xs">
                                        <span className="block mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Citation</span>
                                        <span className="font-bold" style={{ color: citationCorrect ? 'var(--apple-green)' : 'var(--apple-red)' }}>
                                            {citationCorrect ? 'Correct' : citedPapers.length > 0 ? 'Wrong source' : 'None cited'}
                                        </span>
                                    </div>
                                    <div className="text-xs">
                                        <span className="block mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Cit. Precision</span>
                                        <span className="font-bold" style={{ color: citationPrecision >= 0.5 ? 'var(--apple-green)' : citationPrecision > 0 ? 'var(--apple-yellow)' : 'var(--apple-red)' }}>
                                            {(citationPrecision * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="text-xs">
                                        <span className="block mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Completeness</span>
                                        <span className="font-bold" style={{ color: completeness >= 0.8 ? 'var(--apple-green)' : completeness >= 0.5 ? 'var(--apple-yellow)' : 'var(--apple-red)' }}>
                                            {(completeness * 100).toFixed(0)}%
                                            <span className="font-normal ml-1" style={{ color: 'var(--apple-text-quaternary)' }}>({foundKeywords.length}/{expectedKeywords.length})</span>
                                        </span>
                                    </div>
                                </div>
                                {/* Keyword detail */}
                                {expectedKeywords.length > 0 && (
                                    <div className="mt-2 flex flex-wrap gap-1">
                                        {expectedKeywords.map(k => {
                                            const found = answerText.includes(k.toLowerCase())
                                            return (
                                                <span key={k} className="text-[10px] px-1.5 py-0.5 rounded" style={
                                                    found
                                                        ? { background: 'var(--apple-green-bg)', color: 'var(--apple-green)' }
                                                        : { background: 'var(--apple-red-bg)', color: 'var(--apple-red)' }
                                                }>
                                                    {found ? '\u2713' : '\u2717'} {k}
                                                </span>
                                            )
                                        })}
                                    </div>
                                )}
                            </div>

                            <div>
                                <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>Generated Answer</div>
                                <div className="text-xs rounded-xl p-3 max-h-48 overflow-y-auto leading-relaxed" style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-primary)' }}>
                                    {formatAnswer(result.answer)}
                                </div>
                            </div>

                            {/* Sources retrieved */}
                            <div>
                                <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>Retrieved Sources</div>
                                <div className="space-y-1">
                                    {sources.map((src, i) => {
                                        const isExpected = expectedPapers.includes(normalizeId(src.paper_id))
                                        const isCited = citedNums.includes(i + 1)
                                        return (
                                            <div key={i} className="text-xs px-2 py-1.5 rounded flex items-center gap-2" style={
                                                isExpected
                                                    ? { background: 'var(--apple-green-bg)', border: '1px solid var(--apple-green-border)' }
                                                    : { background: 'var(--apple-bg-tertiary)' }
                                            }>
                                                <span className="font-mono w-5" style={{ color: isCited ? 'var(--apple-accent)' : 'var(--apple-text-tertiary)' }}>{i + 1}</span>
                                                <span className="font-mono text-[10px]" style={{ color: isExpected ? 'var(--apple-green)' : 'var(--apple-text-tertiary)' }}>
                                                    {normalizeId(src.paper_id) || '?'}
                                                </span>
                                                <span className="truncate flex-1" style={{ color: 'var(--apple-text-secondary)' }}>{src.title || src.text_preview?.slice(0, 80)}</span>
                                                <span className="font-mono text-[10px]" style={{ color: 'var(--apple-text-quaternary)' }}>{src.score?.toFixed(3)}</span>
                                                {isExpected && <span className="text-[10px]" style={{ color: 'var(--apple-green)' }}>expected</span>}
                                                {isCited && <span className="text-[10px]" style={{ color: 'var(--apple-accent)' }}>cited</span>}
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>

                            {/* Quick metrics */}
                            <div className="flex gap-4 text-xs">
                                <span style={{ color: 'var(--apple-text-tertiary)' }}>Latency: <span style={{ color: 'var(--apple-text-primary)' }}>{Math.round(result.latency_ms)}ms</span></span>
                                <span style={{ color: 'var(--apple-text-tertiary)' }}>Deflected: <span style={{ color: 'var(--apple-text-primary)' }}>{result.deflected ? 'Yes' : 'No'}</span></span>
                                <span style={{ color: 'var(--apple-text-tertiary)' }}>Backend: <span style={{ color: 'var(--apple-text-primary)' }}>{result.model}</span></span>
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
                ? <span key={i} className="border-b" style={{ background: 'var(--apple-yellow-bg)', color: 'var(--apple-yellow)', borderColor: 'var(--apple-yellow)' }} title="Potentially unsupported claim">{p.text}</span>
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
                    className="flex-1 rounded-lg text-xs px-3 py-2 focus:outline-none"
                    style={{ background: 'var(--apple-bg-tertiary)', border: '1px solid var(--apple-border-primary)', color: 'var(--apple-text-primary)', '--tw-placeholder-color': 'var(--apple-text-quaternary)' }}
                />
                <button
                    onClick={() => handleRun()}
                    disabled={running || !question.trim()}
                    className="disabled:opacity-50 text-white text-xs px-4 py-2 rounded-lg transition flex items-center gap-1.5 shrink-0"
                    style={{ background: 'var(--apple-accent)' }}
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
                        className="text-[10px] px-2 py-1 rounded-lg transition disabled:opacity-50 hover:opacity-80"
                        style={{ background: 'var(--apple-glass-bg)', border: '1px solid var(--apple-border-primary)', color: 'var(--apple-text-secondary)' }}
                    >
                        {q}
                    </button>
                ))}
            </div>

            {error && (
                <div className="flex items-center gap-2 text-xs rounded-lg px-3 py-2 mb-3" style={{ color: 'var(--apple-red)', background: 'var(--apple-red-bg)', border: '1px solid var(--apple-red-border)' }}>
                    <AlertIcon size={14} />
                    <span>{error}</span>
                </div>
            )}

            {/* Pipeline result + detection */}
            {pipelineResult && (
                <div className="glass-card p-5 space-y-3">
                    {/* Detection scorecard */}
                    {detection && (
                        <div className="rounded-xl p-3" style={{ background: 'var(--apple-bg-tertiary)', border: '1px solid var(--apple-border-primary)' }}>
                            <div className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--apple-text-tertiary)' }}>Hallucination Check</div>
                            <div className="grid grid-cols-3 gap-3">
                                <div className="text-xs">
                                    <span className="block mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Verdict</span>
                                    <span className="font-bold" style={{ color: detection.has_hallucination ? 'var(--apple-yellow)' : 'var(--apple-green)' }}>
                                        {detection.has_hallucination ? 'Potential issues found' : 'Looks faithful'}
                                    </span>
                                </div>
                                <div className="text-xs">
                                    <span className="block mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Unsupported Spans</span>
                                    <span className="font-bold" style={{ color: (detection.flagged_spans?.length || 0) > 0 ? 'var(--apple-yellow)' : 'var(--apple-green)' }}>
                                        {detection.flagged_spans?.length || 0}
                                    </span>
                                </div>
                                <div className="text-xs">
                                    <span className="block mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Latency</span>
                                    <span className="font-bold" style={{ color: 'var(--apple-text-primary)' }}>
                                        {Math.round(pipelineResult.latency_ms)}ms + {detection.latency_ms}ms
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Answer with highlighted spans */}
                    <div>
                        <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>
                            Generated Answer
                            {detection?.flagged_spans?.length > 0 && (
                                <span className="ml-2 normal-case" style={{ color: 'var(--apple-yellow)' }}>(amber = potentially unsupported)</span>
                            )}
                        </div>
                        <div className="text-xs rounded-xl p-3 max-h-48 overflow-y-auto leading-relaxed" style={{ background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-primary)' }}>
                            {detection?.flagged_spans?.length > 0
                                ? renderWithHighlights(pipelineResult.answer || '', detection.flagged_spans)
                                : formatAnswer(pipelineResult.answer)
                            }
                        </div>
                    </div>

                    {/* Flagged spans detail */}
                    {detection?.flagged_spans?.length > 0 && (
                        <div>
                            <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>Flagged Spans</div>
                            <div className="space-y-1">
                                {detection.flagged_spans.map((s, i) => (
                                    <div key={i} className="text-xs rounded px-2 py-1.5" style={{ background: 'var(--apple-yellow-bg)', border: '1px solid var(--apple-yellow-border)' }}>
                                        <span className="text-[10px] font-mono mr-2" style={{ color: 'var(--apple-yellow)' }}>{s.type}</span>
                                        <span style={{ color: 'var(--apple-text-primary)' }}>"{s.text.slice(0, 150)}{s.text.length > 150 ? '...' : ''}"</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Sources */}
                    <div>
                        <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--apple-text-tertiary)' }}>Retrieved Sources</div>
                        <div className="space-y-1">
                            {(pipelineResult.sources || []).map((src, i) => (
                                <div key={i} className="text-xs px-2 py-1.5 rounded flex items-center gap-2" style={{ background: 'var(--apple-bg-tertiary)' }}>
                                    <span className="font-mono w-5" style={{ color: 'var(--apple-text-tertiary)' }}>{i + 1}</span>
                                    <span className="truncate flex-1" style={{ color: 'var(--apple-text-secondary)' }}>{src.title || src.text_preview?.slice(0, 80)}</span>
                                    <span className="font-mono text-[10px]" style={{ color: 'var(--apple-text-quaternary)' }}>{src.score?.toFixed(3)}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Quick metrics */}
                    <div className="flex gap-4 text-xs">
                        <span style={{ color: 'var(--apple-text-tertiary)' }}>Deflected: <span style={{ color: 'var(--apple-text-primary)' }}>{pipelineResult.deflected ? 'Yes' : 'No'}</span></span>
                        <span style={{ color: 'var(--apple-text-tertiary)' }}>Backend: <span style={{ color: 'var(--apple-text-primary)' }}>{pipelineResult.model}</span></span>
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
                {/* Benchmark selector — segmented control */}
                <div className="inline-flex rounded-full p-0.5 mb-5" style={{ background: 'var(--apple-bg-tertiary)', border: '1px solid var(--apple-border-secondary)' }}>
                    {Object.entries(benchmarkInfo).map(([key, bm]) => (
                        <button
                            key={key}
                            onClick={() => { setActiveBenchmark(key); setShowExamples(false) }}
                            className={`px-5 py-1.5 rounded-full text-xs font-medium transition-all ${
                                activeBenchmark === key ? 'shadow-sm' : 'hover:opacity-80'
                            }`}
                            style={activeBenchmark === key
                                ? { background: 'var(--apple-accent)', color: '#fff' }
                                : { background: 'transparent', color: 'var(--apple-text-secondary)' }
                            }
                        >
                            {bm.name}
                        </button>
                    ))}
                </div>

                {/* Benchmark info */}
                <div className="mb-6">
                    <h3 className="text-sm font-medium" style={{ color: 'var(--apple-text-primary)' }}>{info.fullName}</h3>
                    <p className="text-xs mt-1 leading-relaxed" style={{ color: 'var(--apple-text-tertiary)' }}>
                        {info.description}
                        <span className="mx-1.5" style={{ color: 'var(--apple-text-quaternary)' }}>&mdash;</span>
                        <span style={{ color: 'var(--apple-text-quaternary)' }}>{info.source}</span>
                    </p>
                </div>

                {/* Loading */}
                {loading && (
                    <div className="flex items-center justify-center py-16">
                        <Spinner size={24} />
                        <span className="ml-3 text-sm" style={{ color: 'var(--apple-text-secondary)' }}>Loading results...</span>
                    </div>
                )}

                {/* Error */}
                {error && !loading && (
                    <div className="flex flex-col items-center justify-center py-16 text-center">
                        <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4" style={{ background: 'var(--apple-glass-bg)' }}>
                            <BarChartIcon size={28} />
                        </div>
                        <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--apple-text-secondary)' }}>No results available</h3>
                        <p className="text-xs max-w-sm" style={{ color: 'var(--apple-text-quaternary)' }}>
                            Run the benchmark evaluation from the command line to generate results:
                        </p>
                        <code className="mt-2 text-xs px-3 py-1.5 rounded-lg" style={{ background: 'var(--apple-bg-secondary)', color: 'var(--apple-text-primary)', border: '1px solid var(--apple-border-primary)' }}>
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
                                <div className="flex items-center gap-2 mb-3">
                                    <div className="w-0.5 h-3.5 rounded-full" style={{ background: 'var(--apple-accent)' }} />
                                    <h4 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--apple-text-tertiary)' }}>{groupName}</h4>
                                </div>
                                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                                    {keys.map(k => (
                                        <MetricCard key={k} metricKey={k} value={results.summary[k]} />
                                    ))}
                                </div>
                            </div>
                        ))}

                        {/* RAGTruth extra summary info */}
                        {activeBenchmark === 'ragtruth' && results.summary && (
                            <div className="glass-card p-5 mb-6">
                                <h4 className="text-xs uppercase tracking-wider mb-2" style={{ color: 'var(--apple-text-tertiary)' }}>Confusion Matrix</h4>
                                <div className="grid grid-cols-2 gap-3 max-w-sm">
                                    {[
                                        ['True Negatives', results.summary.true_negatives, 'var(--apple-green)'],
                                        ['True Positives', results.summary.true_positives, 'var(--apple-green)'],
                                        ['False Positives', results.summary.false_positives, 'var(--apple-red)'],
                                        ['False Negatives', results.summary.false_negatives, 'var(--apple-red)'],
                                    ].map(([label, val, color]) => val !== undefined && (
                                        <div key={label} className="text-xs">
                                            <span style={{ color: 'var(--apple-text-tertiary)' }}>{label}: </span>
                                            <span className="font-bold" style={{ color }}>{val}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Eval Trends (right after metrics) */}
                        <TrendsPanel />

                        {/* Topic breakdown (RAG-Bench only) — always visible */}
                        {activeBenchmark === 'ragbench' && results.by_topic && Object.keys(results.by_topic).length > 0 && (
                            <div className="mb-6">
                                <div className="flex items-center gap-2 mb-3">
                                    <div className="w-0.5 h-3.5 rounded-full" style={{ background: 'var(--apple-accent)' }} />
                                    <h4 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--apple-text-tertiary)' }}>Performance by Topic</h4>
                                </div>
                                <div className="glass-card overflow-hidden">
                                    <table className="w-full">
                                        <thead>
                                            <tr className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--apple-text-tertiary)', borderBottom: '1px solid var(--apple-border-primary)' }}>
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
                            </div>
                        )}

                        {/* Timestamp */}
                        {results.timestamp && (
                            <p className="text-xs mb-4" style={{ color: 'var(--apple-text-quaternary)' }}>
                                Last evaluated: {new Date(results.timestamp).toLocaleString()}
                            </p>
                        )}
                        {results._source_file && (
                            <p className="text-xs mb-4" style={{ color: 'var(--apple-text-quaternary)' }}>
                                Source: {results._source_file}
                                {results.summary?.total_queries && ` (${results.summary.total_queries} queries)`}
                            </p>
                        )}
                    </div>
                )}

                {/* Try it section (RAG-Bench) */}
                {activeBenchmark === 'ragbench' && examples && (
                    <div className="mt-6 pt-6" style={{ borderTop: '1px solid var(--apple-divider)' }}>
                        <button
                            onClick={() => setShowExamples(!showExamples)}
                            className="flex items-center gap-2 text-sm font-medium transition mb-4 hover:opacity-80"
                            style={{ color: 'var(--apple-text-primary)' }}
                        >
                            {showExamples ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                            <SearchIcon size={14} />
                            Try a Benchmark Example
                        </button>
                        <p className="text-xs mb-4" style={{ color: 'var(--apple-text-tertiary)' }}>
                            Pick a question from the benchmark and run it through the pipeline to see how retrieval, generation, and citation work in real time.
                        </p>
                        {showExamples && <TryItPanel examples={examples} />}
                    </div>
                )}

                {/* Try it section (RAGTruth) */}
                {activeBenchmark === 'ragtruth' && (
                    <div className="mt-6 pt-6" style={{ borderTop: '1px solid var(--apple-divider)' }}>
                        <button
                            onClick={() => setShowExamples(!showExamples)}
                            className="flex items-center gap-2 text-sm font-medium transition mb-4 hover:opacity-80"
                            style={{ color: 'var(--apple-text-primary)' }}
                        >
                            {showExamples ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                            <SearchIcon size={14} />
                            Try Hallucination Detection
                        </button>
                        <p className="text-xs mb-4" style={{ color: 'var(--apple-text-tertiary)' }}>
                            Ask a question about AI/ML research. The pipeline generates an answer, then the hallucination detector checks each sentence against the retrieved sources.
                        </p>
                        {showExamples && <RagtruthTryItPanel />}
                    </div>
                )}

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
            <div className="rounded-xl overflow-hidden" style={{ background: 'var(--apple-chart-bg)', border: '1px solid var(--apple-chart-border)' }}>
                <div className="px-3 pt-2.5 pb-1">
                    <span className="text-[11px] font-medium" style={{ color: 'var(--apple-chart-label)' }}>{title}</span>
                </div>
                <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ background: 'var(--apple-chart-canvas)', display: 'block' }}>
                    {yTicks.map((v, i) => (
                        <g key={i}>
                            <line x1={PAD_L} y1={yPos(v)} x2={W - PAD_R} y2={yPos(v)} style={{ stroke: 'var(--apple-chart-grid)' }} strokeWidth="1" strokeDasharray="4,3" />
                            <text x={PAD_L - 6} y={yPos(v) + 3} textAnchor="end" style={{ fill: 'var(--apple-chart-axis)' }} fontSize="8" fontFamily="ui-monospace,monospace">{fmtY(v)}</text>
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
                    <text x={xPos(0)} y={H - 4} textAnchor="start" style={{ fill: 'var(--apple-chart-axis)' }} fontSize="8" fontFamily="ui-monospace,monospace">
                        {data[0].timestamp?.slice(5, 10) || ''}
                    </text>
                    <text x={xPos(data.length - 1)} y={H - 4} textAnchor="end" style={{ fill: 'var(--apple-chart-axis)' }} fontSize="8" fontFamily="ui-monospace,monospace">
                        {data[data.length - 1].timestamp?.slice(5, 10) || ''}
                    </text>
                </svg>
                <div className="flex items-center gap-4 px-3 py-1.5 justify-center flex-wrap" style={{ borderTop: '1px solid var(--apple-chart-grid)' }}>
                    {series.map(s => {
                        const last = data[data.length - 1]
                        const prev = data.length >= 2 ? data[data.length - 2] : null
                        const val = last[s.key] || 0
                        const delta = prev ? val - (prev[s.key] || 0) : 0
                        const isUp = delta > 0.001
                        const isDown = delta < -0.001
                        return (
                            <span key={s.label} className="flex items-center gap-1.5 text-[10px]" style={{ color: 'var(--apple-chart-label)' }}>
                                <span className="inline-block w-3 h-0.5 rounded" style={{ backgroundColor: s.color }} />
                                {s.label}
                                <span className="font-mono" style={{ color: s.color }}>{fmtY(val)}</span>
                                {isUp && <span style={{ color: 'var(--apple-green)' }}>+{delta.toFixed(3)}</span>}
                                {isDown && <span style={{ color: 'var(--apple-red)' }}>{delta.toFixed(3)}</span>}
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

    if (loading) return <div className="text-xs mb-6" style={{ color: 'var(--apple-text-tertiary)' }}><Spinner size={14} /> Loading trends...</div>
    if (!trends || trends.length < 2) return null

    return (
        <div className="glass-card p-5 mb-6">
            <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                    <BarChartIcon size={16} /> Eval Trends
                </h3>
                <div className="flex gap-1 text-xs">
                    {['production', 'manual', 'all'].map(t => (
                        <button
                            key={t}
                            onClick={() => setRunType(t)}
                            className={`px-2 py-0.5 rounded transition-colors ${
                                runType === t ? 'text-white' : 'hover:opacity-80'
                            }`}
                            style={runType === t
                                ? { background: 'var(--apple-accent)' }
                                : { background: 'var(--apple-bg-tertiary)', color: 'var(--apple-text-secondary)' }
                            }
                        >
                            {t.charAt(0).toUpperCase() + t.slice(1)}
                        </button>
                    ))}
                </div>
            </div>
            <p className="text-xs mb-3" style={{ color: 'var(--apple-text-tertiary)' }}>
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
        <div className="glass-card p-5 mt-6">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                    <ZapIcon size={16} /> Scheduled Auto-Eval
                </h3>
                <button
                    onClick={handleToggle}
                    disabled={toggling}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors`}
                    style={{ background: schedule.enabled ? 'var(--apple-green)' : 'var(--apple-text-quaternary)' }}
                >
                    <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${schedule.enabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
                </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                <div className="rounded-xl p-2.5" style={{ background: 'var(--apple-bg-tertiary)' }}>
                    <div className="mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Status</div>
                    <div style={{ color: schedule.enabled ? 'var(--apple-green)' : 'var(--apple-text-secondary)' }}>
                        {schedule.enabled ? 'Active' : 'Disabled'}
                    </div>
                </div>
                <div className="rounded-xl p-2.5" style={{ background: 'var(--apple-bg-tertiary)' }}>
                    <div className="mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Interval</div>
                    <div style={{ color: 'var(--apple-text-primary)' }}>Every {schedule.interval_hours}h</div>
                </div>
                <div className="rounded-xl p-2.5" style={{ background: 'var(--apple-bg-tertiary)' }}>
                    <div className="mb-0.5" style={{ color: 'var(--apple-text-tertiary)' }}>Last Run</div>
                    <div style={{ color: 'var(--apple-text-primary)' }}>{schedule.last_run ? new Date(schedule.last_run).toLocaleDateString() : '\u2014'}</div>
                </div>
            </div>

            {schedule.last_run_summary && Object.keys(schedule.last_run_summary).length > 0 && (
                <div className="mt-3 text-xs" style={{ color: 'var(--apple-text-secondary)' }}>
                    Last: MRR {schedule.last_run_summary.retrieval_mrr?.toFixed(3) || '\u2014'} · Hit Rate {schedule.last_run_summary.retrieval_hit_rate?.toFixed(3) || '\u2014'} · Latency {Math.round(schedule.last_run_summary.avg_latency_ms || 0)}ms
                </div>
            )}
        </div>
    )
}
