import React, { useState, useRef, useEffect, useCallback } from 'react'
import { queryRAGStream, fetchPaper, API_BASE } from '../utils/api'
import { Message } from '../components/Message'
import { LoadingDots } from '../components/LoadingDots'
import { PaperViewer } from '../components/PaperViewer'
import { SendIcon, AlertIcon, Spinner } from '../components/Icons'

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
            data: { sources: [], deflected: false, deflection_reason: '', latency_ms: 0, backend: '', model: '', question: q },
        }])
        setIsLoading(true)

        let streamedText = ''

        try {
            await queryRAGStream(q, {
                onPipeline: (stage) => {
                    setMessages(prev => {
                        const updated = [...prev]
                        const msg = updated[updated.length - 1]
                        if (msg?.role === 'assistant') {
                            const stages = [...(msg.data?.pipelineStages || [])]
                            const existing = stages.findIndex(s => s.stage === stage.stage)
                            if (existing >= 0) stages[existing] = stage
                            else stages.push(stage)
                            msg.data = { ...msg.data, pipelineStages: stages }
                        }
                        return [...updated]
                    })
                },
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
                                pipeline: evt.pipeline || null,
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
                    <div className="flex-1 overflow-y-auto pb-4">
                        {serverOffline && (
                            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
                                <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                                    style={{ background: 'var(--apple-red-bg)', border: '1px solid var(--apple-red-border)' }}>
                                    <span style={{ fontSize: '28px' }}>&#x1F50C;</span>
                                </div>
                                <h2 className="text-lg font-semibold mb-2" style={{ color: 'var(--apple-text-primary)' }}>Server Offline</h2>
                                <p className="text-sm max-w-sm mb-1" style={{ color: 'var(--apple-text-secondary)' }}>
                                    The RAG-Bench API is not reachable right now.
                                </p>
                                <p className="text-xs max-w-sm mb-4" style={{ color: 'var(--apple-text-tertiary)' }}>
                                    This runs on a spare machine at home — might be down or restarting. Retrying...
                                </p>
                                <p className="text-xs flex items-center gap-2" style={{ color: 'var(--apple-yellow)' }}>
                                    <Spinner size={12} /> Waiting for server...
                                </p>
                            </div>
                        )}
                        {!serverOffline && isEmpty && !isLoading && (
                            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
                                <div className="glass-card px-5 py-4 mb-4 max-w-md text-left space-y-2">
                                    <p className="text-sm" style={{ color: 'var(--apple-text-primary)' }}>
                                        Ask anything about AI/ML research. Answers come from <span style={{ color: 'var(--apple-text-primary)', fontWeight: 500 }}>500+ papers</span> with <span className="font-medium" style={{ color: 'var(--apple-accent)' }}>[Source N]</span> citations so you can check the originals.
                                    </p>
                                    <p className="text-xs" style={{ color: 'var(--apple-text-secondary)' }}>Search uses BM25 + embeddings + reranking under the hood.</p>
                                    <p className="text-xs" style={{ color: 'var(--apple-yellow)' }}>&#9888; First question might take 10–30s to warm up.</p>
                                    <p className="text-xs" style={{ color: 'var(--apple-text-tertiary)' }}>Runs on a spare machine at home, so it might go down sometimes.</p>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-w-lg w-full">
                                    {EXAMPLES.map((q, i) => (
                                        <button key={i} onClick={() => handleSubmit(q)} disabled={!ready}
                                            className="text-left text-xs rounded-[16px] p-4 transition disabled:opacity-50 disabled:cursor-not-allowed"
                                            style={{ background: 'var(--apple-glass-bg)', border: '1px solid var(--apple-glass-border)', color: 'var(--apple-text-secondary)' }}>
                                            {q}
                                        </button>
                                    ))}
                                </div>
                                {!ready && (
                                    <p className="text-xs mt-4 flex items-center gap-2" style={{ color: 'var(--apple-yellow)' }}>
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
                    <div className="sticky bottom-0 pt-2 pb-2" style={{ background: 'var(--apple-bg-primary)' }}>
                        {error && (
                            <div className="text-xs mb-2 flex items-center gap-1" style={{ color: 'var(--apple-red)' }}>
                                <AlertIcon size={12} />{error}
                            </div>
                        )}
                        <div className="flex items-end gap-2">
                            <textarea ref={inputRef} value={input}
                                onChange={e => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={ready ? "Ask a question about AI/ML research papers..." : "Pipeline loading..."}
                                disabled={!ready || isLoading} rows={1}
                                className="flex-1 rounded-[16px] px-5 py-3.5 text-sm resize-none focus:outline-none disabled:opacity-50 placeholder-gray-500"
                                style={{ background: 'var(--apple-input-bg)', border: '1px solid var(--apple-input-border)', color: 'var(--apple-text-primary)', backdropFilter: 'blur(20px)', minHeight: '48px', maxHeight: '120px' }}
                                onInput={e => { e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px' }}
                            />
                            <button onClick={() => handleSubmit()}
                                disabled={!input.trim() || isLoading || !ready}
                                className="text-white p-3 rounded-[14px] transition flex-shrink-0 disabled:opacity-50"
                                style={{ background: input.trim() && !isLoading && ready ? 'var(--apple-accent)' : 'var(--apple-bg-tertiary)', color: input.trim() && !isLoading && ready ? '#fff' : 'var(--apple-text-quaternary)' }}>
                                {isLoading ? <Spinner size={18} /> : <SendIcon size={18} />}
                            </button>
                        </div>
                        <p className="text-[10px] text-center mt-2" style={{ color: 'var(--apple-text-quaternary)' }}>
                            Answers pulled from papers — always double-check the sources.
                        </p>
                    </div>
                </main>

                {paperLoading && (
                    <div className="fixed bottom-6 left-6 glass rounded-xl px-4 py-2.5 flex items-center gap-2 z-50 shadow-apple-md">
                        <Spinner size={14} /> <span className="text-xs" style={{ color: 'var(--apple-text-primary)' }}>Loading paper...</span>
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
