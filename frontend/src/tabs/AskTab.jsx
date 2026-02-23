import React, { useState, useRef, useEffect, useCallback } from 'react'
import { queryRAGStream, fetchStats, runEvalAPI, fetchPaper, API_BASE } from '../utils/api'
import { Message } from '../components/Message'
import { LoadingDots } from '../components/LoadingDots'
import { StatsPanel } from '../components/StatsPanel'
import { EvalPanel } from '../components/EvalPanel'
import { PaperViewer } from '../components/PaperViewer'
import { SendIcon, BookIcon, DatabaseIcon, BarChartIcon, AlertIcon, Spinner } from '../components/Icons'

const EXAMPLES = [
    "What is the scaled dot-product attention formula?",
    "How does LoRA reduce trainable parameters?",
    "Compare RAG and RLHF approaches to factual accuracy",
    "What is the forward diffusion process in DDPM?",
    "What is the Mamba architecture's selective state space mechanism?",
]

export function AskTab({ ready, serverOffline }) {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [stats, setStats] = useState(null)
    const [showStats, setShowStats] = useState(false)
    const [showEval, setShowEval] = useState(false)
    const [evalResults, setEvalResults] = useState(null)
    const [evalRunning, setEvalRunning] = useState(false)
    const [error, setError] = useState(null)

    // Paper viewer state
    const [paperViewerData, setPaperViewerData] = useState(null)
    const [paperLoading, setPaperLoading] = useState(false)
    const paperCache = useRef({})

    const endRef = useRef(null)
    const inputRef = useRef(null)

    useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, isLoading])

    const handleSubmit = useCallback(async (override) => {
        const q = override || input.trim()
        if (!q || isLoading) return
        setInput(''); setError(null)

        setMessages(prev => [...prev, { role: 'user', content: q }])

        setMessages(prev => [...prev, {
            role: 'assistant', content: '', streaming: true,
            data: { sources: [], deflected: false, deflection_reason: '', latency_ms: 0, backend: '', model: '' },
        }])
        setIsLoading(true)

        let streamedText = ''

        try {
            await queryRAGStream(q, {
                onSources: (sources) => {
                    setMessages(prev => {
                        const updated = [...prev]
                        const msg = updated[updated.length - 1]
                        if (msg?.role === 'assistant') {
                            msg.data = { ...msg.data, sources }
                        }
                        return updated
                    })
                },
                onToken: (token) => {
                    streamedText += token
                    setMessages(prev => {
                        const updated = [...prev]
                        const msg = updated[updated.length - 1]
                        if (msg?.role === 'assistant') {
                            msg.content = streamedText
                        }
                        return [...updated]
                    })
                },
                onDone: (evt) => {
                    setMessages(prev => {
                        const updated = [...prev]
                        const msg = updated[updated.length - 1]
                        if (msg?.role === 'assistant') {
                            msg.content = evt.answer || streamedText
                            msg.streaming = false
                            msg.data = {
                                ...msg.data,
                                deflected: evt.deflected || false,
                                deflection_reason: evt.reason || '',
                                latency_ms: evt.latency_ms || 0,
                                backend: evt.backend || '',
                                model: evt.model || '',
                                quality: evt.quality || null,
                            }
                        }
                        return [...updated]
                    })
                },
                onError: (errMsg) => {
                    setError(errMsg)
                    setMessages(prev => {
                        const updated = [...prev]
                        const msg = updated[updated.length - 1]
                        if (msg?.role === 'assistant') {
                            msg.content = `Error: ${errMsg}`
                            msg.streaming = false
                        }
                        return [...updated]
                    })
                },
            })
        } catch (err) {
            setError(err.message)
            setMessages(prev => {
                const updated = [...prev]
                const msg = updated[updated.length - 1]
                if (msg?.role === 'assistant') {
                    msg.content = `Error: ${err.message}`
                    msg.streaming = false
                }
                return [...updated]
            })
        } finally {
            setIsLoading(false)
            inputRef.current?.focus()
        }
    }, [input, isLoading])

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() }
    }

    const handleRunEval = async (runAll) => {
        setEvalRunning(true)
        try { setEvalResults(await runEvalAPI(runAll)) }
        catch (err) { setError(`Eval failed: ${err.message}`) }
        finally { setEvalRunning(false) }
    }

    const handleShowStats = async () => {
        if (!showStats) {
            try { setStats(await fetchStats()) }
            catch (err) { setError(`Stats: ${err.message}`) }
        }
        setShowStats(!showStats)
    }

    const handleViewSource = useCallback(async (source) => {
        if (!source.paper_id) return
        setPaperLoading(true)
        try {
            let paper = paperCache.current[source.paper_id]
            if (!paper) {
                paper = await fetchPaper(source.paper_id)
                paperCache.current[source.paper_id] = paper
            }
            setPaperViewerData({
                paper,
                highlightChunkId: source.chunk_id || '',
                highlightText: source.text_preview || '',
            })
        } catch (err) {
            setError(`Paper: ${err.message}`)
        } finally {
            setPaperLoading(false)
        }
    }, [])

    const isEmpty = messages.length === 0

    return (
        <div className="flex flex-1 overflow-hidden">
            {/* Main content area */}
            <div className="flex-1 flex flex-col overflow-hidden">
                <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-4 flex flex-col">
                    <div className="flex items-center gap-2 mb-3">
                        <button onClick={handleShowStats}
                            className={`text-xs px-3 py-1.5 rounded-md transition ${showStats ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}>
                            <span className="inline-flex items-center gap-1"><DatabaseIcon size={12} />Stats</span>
                        </button>
                        <button onClick={() => setShowEval(!showEval)}
                            className={`text-xs px-3 py-1.5 rounded-md transition ${showEval ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}>
                            <span className="inline-flex items-center gap-1"><BarChartIcon size={12} />Eval</span>
                        </button>
                    </div>

                    {showStats && <StatsPanel stats={stats} onClose={() => setShowStats(false)} />}
                    {showEval && <EvalPanel evalResults={evalResults} isRunning={evalRunning} onRun={handleRunEval} onClose={() => setShowEval(false)} />}

                    <div className="flex-1 overflow-y-auto pb-4">
                        {serverOffline && (
                            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
                                <div className="w-16 h-16 bg-red-600/10 border border-red-600/30 rounded-2xl flex items-center justify-center mb-4">
                                    <span style={{ fontSize: '28px' }}>&#x1F50C;</span>
                                </div>
                                <h2 className="text-lg font-semibold text-gray-100 mb-2">Server Offline</h2>
                                <p className="text-sm text-gray-400 max-w-sm mb-1">
                                    The RAG-Bench API is not reachable right now.
                                </p>
                                <p className="text-xs text-gray-500 max-w-sm mb-4">
                                    This service runs on a spare personal machine — it may be temporarily offline or restarting. Retrying automatically...
                                </p>
                                <p className="text-xs text-yellow-500 flex items-center gap-2">
                                    <Spinner size={12} /> Waiting for server...
                                </p>
                            </div>
                        )}
                        {!serverOffline && isEmpty && !isLoading && (
                            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
                                <div className="w-16 h-16 bg-blue-600/10 border border-blue-600/20 rounded-2xl flex items-center justify-center mb-4">
                                    <BookIcon size={28} />
                                </div>
                                <h2 className="text-lg font-semibold text-gray-100 mb-3">RAG-Bench</h2>
                                <div className="bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 mb-4 max-w-md text-left space-y-2">
                                    <p className="text-sm text-gray-300">
                                        Ask questions about AI/ML research. Answers are grounded in <span className="text-white font-medium">{stats?.total_papers || '500+'} papers</span> with inline <span className="text-blue-400 font-medium">[Source N]</span> citations — no hallucinations.
                                    </p>
                                    <p className="text-xs text-gray-400">Uses BM25 + semantic search + cross-encoder reranking for precision retrieval.</p>
                                    <p className="text-xs text-amber-400">&#9888; First response may take 10–30 s while the model loads.</p>
                                    <p className="text-xs text-gray-500">Hosted on a spare personal machine — uptime is best-effort and the service may occasionally be offline.</p>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-w-lg w-full">
                                    {EXAMPLES.map((q, i) => (
                                        <button key={i} onClick={() => handleSubmit(q)} disabled={!ready}
                                            className="text-left text-xs bg-gray-800/50 hover:bg-gray-800 border border-gray-700/50 hover:border-gray-600 rounded-lg p-3 text-gray-400 hover:text-gray-200 transition disabled:opacity-50 disabled:cursor-not-allowed">
                                            {q}
                                        </button>
                                    ))}
                                </div>
                                {!ready && (
                                    <p className="text-xs text-yellow-500 mt-4 flex items-center gap-2">
                                        <Spinner size={12} /> Loading pipeline...
                                    </p>
                                )}
                            </div>
                        )}

                        {messages.map((msg, i) => <Message key={i} message={msg} onViewSource={handleViewSource} />)}
                        {isLoading && <LoadingDots />}
                        <div ref={endRef} />
                    </div>

                    {/* Input */}
                    <div className="sticky bottom-0 pt-2 pb-2 bg-gray-950">
                        {error && (
                            <div className="text-xs text-red-400 mb-2 flex items-center gap-1">
                                <AlertIcon size={12} />{error}
                            </div>
                        )}
                        <div className="flex items-end gap-2">
                            <textarea ref={inputRef} value={input}
                                onChange={e => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={ready ? "Ask a question about AI/ML research papers..." : "Pipeline loading..."}
                                disabled={!ready || isLoading} rows={1}
                                className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 disabled:opacity-50"
                                style={{ minHeight: '48px', maxHeight: '120px' }}
                                onInput={e => { e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px' }}
                            />
                            <button onClick={() => handleSubmit()}
                                disabled={!input.trim() || isLoading || !ready}
                                className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white p-3 rounded-xl transition flex-shrink-0">
                                {isLoading ? <Spinner size={18} /> : <SendIcon size={18} />}
                            </button>
                        </div>
                        <p className="text-[10px] text-gray-600 text-center mt-2">
                            Answers grounded in research papers. Always verify against original sources.
                        </p>
                    </div>
                </main>

                {paperLoading && (
                    <div className="fixed bottom-6 left-6 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 flex items-center gap-2 z-50 shadow-lg">
                        <Spinner size={14} /> <span className="text-xs text-gray-300">Loading paper...</span>
                    </div>
                )}
            </div>

            {/* Paper Viewer — persistent right panel */}
            <PaperViewer
                paper={paperViewerData?.paper}
                highlightChunkId={paperViewerData?.highlightChunkId}
                highlightText={paperViewerData?.highlightText}
                onClose={() => setPaperViewerData(null)}
                isEmpty={!paperViewerData}
            />
        </div>
    )
}
